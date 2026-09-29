"""
163轮量化模型推理脚本 - 基于run_163_working.py修改
用于测试INT8量化模型在验证集上的性能
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.kitti_dataloader import create_test_dataloader
from models.fpn_resnet import get_pose_net
from utils.evaluation_utils import decode, post_processing
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid

CLASS_ID_TO_NAME = {idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())


def parse_configs():
    parser = argparse.ArgumentParser(description='163轮量化模型推理')
    parser.add_argument('--saved_fn', type=str, default='163_quantized', metavar='FN')
    parser.add_argument('--arch', type=str, default='fpn_resnet_18', metavar='ARCH')
    parser.add_argument('--quantized_model', type=str,
                        default='../quantized_models/Model_163_quantized_static_20251102_205000.pth',
                        metavar='PATH', help='Path to quantized model')
    parser.add_argument('--dataset-dir', type=str, required=True, metavar='PATH')
    parser.add_argument('--test-subdir', type=str, default='training', metavar='DIR')
    parser.add_argument('--imagesets-dir', type=str, default=None, metavar='DIR')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process (None=all)')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--peak_thresh', type=float, default=0.25)  # 超激进
    parser.add_argument('--nms_thresh', type=float, default=0.2)  # 超激进NMS
    parser.add_argument('--output-dir', type=str, default='../results/163_quantized')

    cfg = edict(vars(parser.parse_args()))
    cfg.pin_memory = True
    cfg.distributed = False

    # 量化模型强制使用CPU
    cfg.no_cuda = True
    cfg.gpu_idx = 0

    # 模型配置
    cfg.input_size = (608, 608)
    cfg.hm_size = (152, 152)
    cfg.down_ratio = 4
    cfg.max_objects = 50
    cfg.imagenet_pretrained = False
    cfg.head_conv = 64
    cfg.num_classes = 3
    cfg.num_center_offset = 2
    cfg.num_z = 1
    cfg.num_dim = 3
    cfg.num_direction = 2
    cfg.num_input_features = 4

    cfg.heads = {
        'hm_cen': cfg.num_classes,
        'cen_offset': cfg.num_center_offset,
        'direction': cfg.num_direction,
        'z_coor': cfg.num_z,
        'dim': cfg.num_dim
    }

    # 设备配置 - 量化模型只支持CPU
    cfg.device = torch.device('cpu')

    # 处理路径
    if not os.path.isabs(cfg.dataset_dir):
        cfg.dataset_dir = os.path.abspath(cfg.dataset_dir)
    if cfg.imagesets_dir and not os.path.isabs(cfg.imagesets_dir):
        cfg.imagesets_dir = os.path.abspath(cfg.imagesets_dir)

    return cfg


def load_quantized_model(cfg):
    """加载量化模型"""
    print("\n正在加载量化模型...")

    try:
        # 尝试直接加载量化模型
        checkpoint = torch.load(cfg.quantized_model, map_location='cpu')

        if 'model' in checkpoint:
            # 完整模型格式
            model = checkpoint['model']
            print(f"[OK] 加载完整量化模型")
            if 'quantization_method' in checkpoint:
                print(f"     量化方法: {checkpoint['quantization_method']}")

            # 量化模型已经是推理模式，不需要调用eval()
            # 直接返回
        else:
            # 需要重建模型结构
            print("[INFO] 检测到state_dict格式，正在重建模型...")
            model = get_pose_net(
                num_layers=18,
                heads=cfg.heads,
                head_conv=cfg.head_conv,
                imagenet_pretrained=False
            )

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            print(f"[OK] 模型结构重建完成")

        # 获取模型大小
        model_size = os.path.getsize(cfg.quantized_model) / (1024 * 1024)
        print(f"[OK] 量化模型加载成功")
        print(f"     模型文件大小: {model_size:.2f} MB")

        return model

    except Exception as e:
        print(f"[ERROR] 量化模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    cfg = parse_configs()

    print("=" * 80)
    print("SFA3D-Modified 163轮量化模型推理")
    print("=" * 80)
    print(f"量化模型路径: {cfg.quantized_model}")
    print(f"数据集目录: {cfg.dataset_dir}")
    print(f"测试子目录: {cfg.test_subdir}")
    print(f"ImageSets目录: {cfg.imagesets_dir}")
    print(f"输出目录: {cfg.output_dir}")
    print(f"样本数量: {cfg.num_samples if cfg.num_samples else '全部'}")
    print(f"设备: {cfg.device} (量化模型仅支持CPU)")
    print(f"超激进参数: peak_thresh={cfg.peak_thresh}, nms_thresh={cfg.nms_thresh}")
    print("=" * 80)

    # 创建输出目录
    make_folder(cfg.output_dir)

    # 检查模型文件
    if not os.path.isfile(cfg.quantized_model):
        print(f"[ERROR] 量化模型文件不存在: {cfg.quantized_model}")
        return False

    # 加载量化模型
    model = load_quantized_model(cfg)
    if model is None:
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

    # 实际处理的样本数
    total_samples = cfg.num_samples if cfg.num_samples else len(dataset)

    print(f"\n开始推理 (处理 {total_samples} 个样本)...")
    print("-" * 80)

    with torch.no_grad():
        for i, (metadatas, bev_maps, img_rgb) in enumerate(dataloader):
            if cfg.num_samples and i >= cfg.num_samples:
                break

            # 数据准备
            bev_maps = bev_maps.to(cfg.device).float()

            # 模型推理
            t1 = time_synchronized()
            try:
                outputs = model(bev_maps)
                outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
                t2 = time_synchronized()
                inference_time = (t2 - t1) * 1000  # ms
                inference_times.append(inference_time)

                # 解码和后处理
                detections = decode(outputs['hm_cen'],
                                  outputs['cen_offset'],
                                  outputs['direction'],
                                  outputs['z_coor'],
                                  outputs['dim'],
                                  K=cfg.K)
                detections = detections.cpu().numpy().astype(np.float32)
                detections = post_processing(detections, cfg.num_classes, cfg.down_ratio,
                                           cfg.peak_thresh, nms_thresh=cfg.nms_thresh)

                # 统计检测结果
                sample_detections = 0
                for j in range(cfg.num_classes):
                    class_name = CLASS_NAMES[j]
                    if len(detections[j]) > 0:
                        num_dets = len(detections[j])
                        class_stats[class_name]['count'] += num_dets
                        class_stats[class_name]['scores'].extend(detections[j][:, 0].tolist())
                        sample_detections += num_dets

                total_detections += sample_detections
                processed += 1

                # 打印进度
                if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                    avg_time = np.mean(inference_times)
                    fps = 1000.0 / avg_time if avg_time > 0 else 0
                    print(f"[{i+1}/{total_samples}] "
                          f"推理时间: {inference_time:.2f}ms | "
                          f"平均: {avg_time:.2f}ms | "
                          f"FPS: {fps:.2f} | "
                          f"本样本检测: {sample_detections}")

            except Exception as e:
                print(f"[ERROR] 样本 {i+1} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 最终统计
    print("\n" + "=" * 80)
    print("推理完成 - 统计结果")
    print("=" * 80)

    print(f"\n成功处理样本数: {processed}/{total_samples}")
    print(f"总检测目标数: {total_detections}")

    print(f"\n各类别检测统计:")
    print("-" * 80)
    for class_name in CLASS_NAMES:
        stats = class_stats[class_name]
        count = stats['count']
        if count > 0:
            scores = stats['scores']
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            print(f"{class_name:10s}: {count:5d} 个 | "
                  f"平均置信度: {avg_score:.3f} | "
                  f"范围: [{min_score:.3f}, {max_score:.3f}]")
        else:
            print(f"{class_name:10s}: {count:5d} 个")

    # 推理性能统计
    if inference_times:
        print(f"\n推理性能统计:")
        print("-" * 80)
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

    print("\n" + "=" * 80)
    print("量化模型推理完成")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
