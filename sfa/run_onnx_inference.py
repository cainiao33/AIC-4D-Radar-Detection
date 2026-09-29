"""
使用ONNX Runtime进行模型推理
支持在验证集上进行完整的性能评估
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime未安装")
    print("安装命令: pip install onnxruntime")
    sys.exit(1)

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.kitti_dataloader import create_test_dataloader
from utils.evaluation_utils import decode_numpy, post_processing_numpy, convert_det_to_real_values
from data_process import kitti_data_utils
from utils.evaluation_utils import post_processing
from utils.misc import make_folder, time_synchronized
from easydict import EasyDict as edict

CLASS_ID_TO_NAME = {idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())


def parse_configs():
    parser = argparse.ArgumentParser(description='ONNX模型推理')
    parser.add_argument('--onnx_model', type=str, required=True,
                       help='ONNX模型路径')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='数据集目录')
    parser.add_argument('--imagesets-dir', type=str, default=None,
                       help='ImageSets目录')
    parser.add_argument('--test-subdir', type=str, default='training',
                       help='测试子目录')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='处理样本数 (None=全部)')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--peak_thresh', type=float, default=0.25)
    parser.add_argument('--nms_thresh', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='./results/onnx_inference')

    cfg = edict(vars(parser.parse_args()))

    # 模型配置
    cfg.input_size = (608, 608)
    cfg.hm_size = (152, 152)
    cfg.down_ratio = 4
    cfg.max_objects = 50
    cfg.num_classes = 3
    cfg.num_center_offset = 2
    cfg.num_z = 1
    cfg.num_dim = 3
    cfg.num_direction = 2

    cfg.heads = {
        'hm_cen': cfg.num_classes,
        'cen_offset': cfg.num_center_offset,
        'direction': cfg.num_direction,
        'z_coor': cfg.num_z,
        'dim': cfg.num_dim
    }

    # ONNX只支持CPU
    cfg.no_cuda = True
    cfg.gpu_idx = 0
    cfg.pin_memory = True
    cfg.distributed = False
    cfg.num_input_features = 4
    cfg.imagenet_pretrained = False
    cfg.head_conv = 64

    # 处理路径
    if not os.path.isabs(cfg.dataset_dir):
        cfg.dataset_dir = os.path.abspath(cfg.dataset_dir)
    if cfg.imagesets_dir and not os.path.isabs(cfg.imagesets_dir):
        cfg.imagesets_dir = os.path.abspath(cfg.imagesets_dir)

    return cfg


def sigmoid_numpy(x):
    """NumPy版本的sigmoid"""
    return 1 / (1 + np.exp(-x))


def main():
    cfg = parse_configs()

    print("="*80)
    print("ONNX Runtime模型推理")
    print("="*80)
    print(f"ONNX模型: {cfg.onnx_model}")
    print(f"数据集目录: {cfg.dataset_dir}")
    print(f"ImageSets目录: {cfg.imagesets_dir}")
    print(f"输出目录: {cfg.output_dir}")
    print(f"样本数量: {cfg.num_samples if cfg.num_samples else '全部'}")
    print(f"超激进参数: peak_thresh={cfg.peak_thresh}, nms_thresh={cfg.nms_thresh}")
    print("="*80)

    # 创建输出目录
    make_folder(cfg.output_dir)

    # 检查ONNX模型文件
    if not os.path.isfile(cfg.onnx_model):
        print(f"[ERROR] ONNX模型文件不存在: {cfg.onnx_model}")
        return False

    # 加载ONNX模型
    print("\n正在加载ONNX模型...")
    try:
        session = ort.InferenceSession(cfg.onnx_model)
        print(f"[OK] ONNX模型加载成功")

        # 显示模型信息
        input_info = session.get_inputs()[0]
        print(f"\n模型信息:")
        print(f"   输入名称: {input_info.name}")
        print(f"   输入形状: {input_info.shape}")
        print(f"   输入类型: {input_info.type}")

        print(f"\n输出节点:")
        for i, output in enumerate(session.get_outputs()):
            print(f"   {i+1}. {output.name}: {output.shape}")

        model_size = os.path.getsize(cfg.onnx_model) / (1024 * 1024)
        print(f"\n   文件大小: {model_size:.2f} MB")

    except Exception as e:
        print(f"[ERROR] ONNX模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建数据加载器
    print("\n正在创建数据加载器...")
    try:
        dataloader = create_test_dataloader(cfg)
        dataset = dataloader.dataset
        print(f"[OK] 数据加载器创建成功，数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"[ERROR] 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 推理统计
    class_stats = {name: {'count': 0, 'scores': []} for name in CLASS_NAMES}
    total_detections = 0
    processed = 0
    inference_times = []

    total_samples = cfg.num_samples if cfg.num_samples else len(dataset)

    print(f"\n开始推理 (处理 {total_samples} 个样本)...")
    print("-"*80)

    input_name = session.get_inputs()[0].name

    for i, (metadatas, bev_maps, img_rgb) in enumerate(dataloader):
        if cfg.num_samples and i >= cfg.num_samples:
            break

        try:
            # 准备输入数据 (转换为NumPy)
            bev_input = bev_maps.numpy().astype(np.float32)

            # ONNX Runtime推理
            t1 = time_synchronized()
            ort_outputs = session.run(None, {input_name: bev_input})
            t2 = time_synchronized()
            inference_time = (t2 - t1) * 1000  # ms
            inference_times.append(inference_time)

            # 提取输出
            hm_cen = sigmoid_numpy(ort_outputs[0])
            cen_offset = sigmoid_numpy(ort_outputs[1])
            direction = ort_outputs[2]
            z_coor = ort_outputs[3]
            dim = ort_outputs[4]

            # 后处理使用numpy版本
            detections = decode_numpy(hm_cen, cen_offset, direction, z_coor, dim, K=cfg.K)
            detections = post_processing_numpy(detections, cfg.num_classes, cfg.down_ratio,
                                              cfg.peak_thresh, inter_class_nms=True,
                                              nms_thresh=cfg.nms_thresh)

            # 转换为KITTI格式并保存结果
            batch_size = len(metadatas[list(metadatas.keys())[0]])
            for j in range(batch_size):
                metainfo = {k: v[j] if isinstance(v, list) else v for k, v in metadatas.items()}
                img_path = metainfo['img_path']
                sample_id = Path(img_path).stem
                frame_id = int(sample_id)  # 从文件名提取frame_id

                # 转换检测结果的坐标系统
                detections_kitti = convert_det_to_real_values(detections[j], cfg.num_classes)

                # 保存为KITTI格式
                if len(detections_kitti) > 0:
                    # 保存检测结果
                    save_dir = os.path.join(cfg.output_dir, 'data')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'{frame_id:06d}.txt')

                    with open(save_path, 'w') as f:
                        for det in detections_kitti:
                            cls_id, x, y, z, h, w, l, yaw = det
                            cls_name = CLASS_ID_TO_NAME.get(int(cls_id), 'DontCare')

                            # 创建KITTI格式的检测行
                            det_line = f"{cls_name} -1 -1 -1 {h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {yaw:.4f} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
                            f.write(det_line)

                # 统计检测结果
                for cls_id in range(cfg.num_classes):
                    if len(detections_kitti) > 0 and cls_id in detections_kitti[:, 0]:
                        class_mask = detections_kitti[:, 0] == cls_id
                        class_dets = detections_kitti[class_mask]
                        class_name = CLASS_ID_TO_NAME.get(cls_id, f'Class_{cls_id}')

                        class_stats[class_name]['count'] += len(class_dets)
                        # 使用检测框的置信度（如果有）
                        if len(class_dets) > 0:
                            class_stats[class_name]['scores'].extend([0.5] * len(class_dets))  # 默认置信度

                total_detections += len(detections_kitti)

            processed += 1

            # 打印进度
            if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                avg_time = np.mean(inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                print(f"[{i+1}/{total_samples}] "
                      f"推理时间: {inference_time:.2f}ms | "
                      f"平均: {avg_time:.2f}ms | "
                      f"FPS: {fps:.2f}")

        except Exception as e:
            print(f"[ERROR] 样本 {i+1} 推理失败: {e}")
            print(f"metadatas type: {type(metadatas)}")
            print(f"metadatas keys: {list(metadatas.keys()) if hasattr(metadatas, 'keys') else 'No keys'}")
            import traceback
            traceback.print_exc()
            continue

    # 最终统计
    print("\n" + "="*80)
    print("推理完成 - 统计结果")
    print("="*80)

    print(f"\n成功处理样本数: {processed}/{total_samples}")

    if inference_times:
        print(f"\n推理性能统计:")
        print("-"*80)
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)
        fps = 1000.0 / avg_time

        print(f"平均推理时间: {avg_time:.2f} ms")
        print(f"最小推理时间: {min_time:.2f} ms")
        print(f"最大推理时间: {max_time:.2f} ms")
        print(f"标准差: {std_time:.2f} ms")
        print(f"推理FPS: {fps:.2f}")

    print("\n" + "="*80)
    print("ONNX Runtime推理完成")
    print("="*80)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
