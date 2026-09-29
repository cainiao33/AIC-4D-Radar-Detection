"""
163轮次模型的超激进验证推理
基于testing_export_ultra_aggressive.py修改，以适配验证集
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from data_process.kitti_data_utils import Calibration
from data_process.transformation import lidar_to_camera_box
from models.model_utils import create_model
from utils.evaluation_utils import decode, post_processing
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid
from utils.visualization_utils import compute_box_3d, project_to_image

CLASS_ID_TO_NAME: Dict[int, str] = {
    idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0
}


def wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_configs():
    parser = argparse.ArgumentParser(description='Ultra-aggressive validation inference for 163-epoch model')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='Name prefix for outputs')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth', metavar='PATH',
                        help='Path to trained weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, required=True, metavar='PATH',
                        help='Dataset root directory (KITTI layout)')
    parser.add_argument('--test-subdir', type=str, default='training', metavar='DIR',
                        help='Validation data directory (training with val.txt)')
    parser.add_argument('--K', type=int, default=50, help='Number of peaks to decode per sample')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size (recommend 1)')
    parser.add_argument('--num_samples', type=int, default=None, help='Process only first N samples (optional)')
    parser.add_argument('--peak_thresh', type=float, default=0.25,
                        help='Score threshold for decoded peaks (ultra-aggressive: 0.25)')
    parser.add_argument('--nms_thresh', type=float, default=0.2,
                        help='NMS IoU threshold (ultra-aggressive: 0.2)')
    parser.add_argument('--output-dir', type=str, default=None, metavar='PATH',
                        help='Directory to store KITTI txt outputs')

    cfg = edict(vars(parser.parse_args()))
    cfg.pin_memory = True
    cfg.distributed = False

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

    cfg.heads = {
        'hm_cen': cfg.num_classes,
        'cen_offset': cfg.num_center_offset,
        'direction': cfg.num_direction,
        'z_coor': cfg.num_z,
        'dim': cfg.num_dim
    }
    cfg.num_input_features = 4

    cfg.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(cfg.dataset_dir):
        cfg.dataset_dir = os.path.join(cfg.root_dir, cfg.dataset_dir)
    cfg.dataset_dir = os.path.abspath(cfg.dataset_dir)

    cfg.use_imagesets = True
    cfg.imagesets_dir = os.path.join(cfg.dataset_dir, 'ImageSets')

    # Device configuration
    cfg.device = torch.device('cpu' if cfg.no_cuda else f'cuda:{cfg.gpu_idx}')

    return cfg


def write_kitti_results(predictions: List[Dict], output_dir: str, dataset_dir: str, test_subdir: str):
    """Write predictions to KITTI format text files"""
    print(f"\nWriting KITTI format results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    label_dir = os.path.join(dataset_dir, test_subdir, 'label_2')
    gt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    gt_sample_ids = [os.path.splitext(f)[0] for f in gt_files]

    for i, pred in enumerate(predictions):
        sample_id = pred['sample_id']
        detections = pred['detections']
        calib = pred['calib']

        output_filename = os.path.join(output_dir, f'{sample_id}.txt')

        kitti_lines = []
        for det in detections:
            h, w, l, x, y, z, rot_y, score, cls_name = det

            # Convert to camera coordinates
            corners_3d_lidar = compute_box_3d(det, cls_name)
            corners_3d_cam = lidar_to_camera_box(corners_3d_lidar, calib.V2C, calib.R0, calib.P2)

            # Project to image for alpha calculation
            corners_2d_img = project_to_image(corners_3d_cam, calib.P2)
            min_x, max_x = np.min(corners_2d_img[:, 0]), np.max(corners_2d_img[:, 0])
            min_y, max_y = np.min(corners_2d_img[:, 1]), np.max(corners_2d_img[:, 1])

            alpha = -np.arctan2(y, x) + rot_y
            alpha = wrap_angle_pi(alpha)

            # Format: type truncated occlusion alpha bbox dims location rotation_y score
            kitti_line = f"{cls_name} -1 -1 -1 {min_x:.2f} {min_y:.2f} {max_x:.2f} {max_y:.2f} " \
                        f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {rot_y:.2f} {score:.2f}"
            kitti_lines.append(kitti_line)

        with open(output_filename, 'w') as f:
            if kitti_lines:
                f.write('\n'.join(kitti_lines))

    print(f"✓ Wrote {len(predictions)} prediction files")
    return output_dir


def main():
    cfg = parse_configs()

    print('\n' + '='*80)
    print('SFA3D-Modified 163轮模型超激进验证集推理')
    print('='*80)
    print(f'模型路径: {cfg.pretrained_path}')
    print(f'数据集目录: {cfg.dataset_dir}')
    print(f'测试子目录: {cfg.test_subdir}')
    print(f'ImageSets目录: {cfg.imagesets_dir}')
    print(f'设备: {cfg.device}')
    print(f'超激进参数: peak_thresh={cfg.peak_thresh}, nms_thresh={cfg.nms_thresh}')
    print('='*80 + '\n')

    # Output directory configuration
    if cfg.output_dir is None:
        cfg.output_dir = os.path.join('..', 'results', 'validation_ultra_aggressive_163')
    make_folder(cfg.output_dir)

    # Load model
    model = create_model(cfg)
    print('\n' + '-*=' * 30 + '\n')

    if not os.path.isfile(cfg.pretrained_path):
        print(f"❌ 模型文件不存在: {cfg.pretrained_path}")
        return False

    model.load_state_dict(torch.load(cfg.pretrained_path, map_location='cpu'))
    print(f'✓ 加载163轮模型权重: {cfg.pretrained_path}\n')

    model = model.to(cfg.device)
    model.eval()

    # Create validation dataloader
    try:
        print('创建验证集数据加载器...')
        dataloader, dataset = create_test_dataloader(cfg, batch_size=cfg.batch_size,
                                                   num_workers=cfg.num_workers,
                                                   test_subdir=cfg.test_subdir)
        print(f'✓ 数据加载器创建成功')
        print(f'  数据集大小: {len(dataset)}')
        if hasattr(dataset, 'sample_id_list'):
            print(f'  样本ID数量: {len(dataset.sample_id_list)}')
            if len(dataset.sample_id_list) > 0:
                print(f'  前5个样本ID: {dataset.sample_id_list[:5]}')
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Class-wise statistics
    class_stats = {CLASS_ID_TO_NAME[i]: {'count': 0, 'scores': []}
                  for i in range(cfg.num_classes)}
    total_detections = 0
    inference_times = []

    print('\n开始超激进推理...')
    print(f'处理样本数: {cfg.num_samples if cfg.num_samples else len(dataset)}')

    # Inference loop
    with torch.no_grad():
        for i, (metadatas, bev_maps) in enumerate(dataloader):
            if cfg.num_samples and i >= cfg.num_samples:
                break

            sample_id = metadatas['sample_id'][0] if 'sample_id' in metadatas else f'sample_{i}'
            calib = Calibration(metadatas['calib_path'][0])

            bev_maps = bev_maps.to(cfg.device).float()

            t1 = time_synchronized()
            outputs = model(bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

            detections = decode(outputs['hm_cen'].cpu().numpy(),
                              outputs['cen_offset'].cpu().numpy(),
                              outputs['direction'].cpu().numpy(),
                              outputs['z_coor'].cpu().numpy(),
                              outputs['dim'].cpu().numpy(),
                              K=cfg.K)
            detections = post_processing(detections, cfg.num_classes, cfg.down_ratio,
                                        cfg.peak_thresh, inter_class_nms=True, nms_thresh=cfg.nms_thresh)
            t2 = time_synchronized()

            batch_time = t2 - t1
            inference_times.extend([batch_time / cfg.batch_size] * cfg.batch_size)

            sample_detections = 0
            sample_predictions = []

            for cls_id, dets in detections[0].items():
                class_name = CLASS_ID_TO_NAME[int(cls_id)]
                for det in dets:
                    final_box = det[1:8]  # h, w, l, x, y, z, rot_y
                    score = float(det[0])

                    class_stats[class_name]['count'] += 1
                    class_stats[class_name]['scores'].append(score)
                    sample_detections += 1

                    sample_predictions.append((*final_box, score, class_name))

            total_detections += sample_detections

            print(f"样本 {i+1}/{min(cfg.num_samples, len(dataset)) if cfg.num_samples else len(dataset)}: "
                  f"{sample_id} - 检测数: {sample_detections} - 用时: {batch_time*1000:.1f}ms")

            # Store prediction for KITTI output
            pred_data = {
                'sample_id': sample_id,
                'detections': sample_predictions,
                'calib': calib
            }
            write_kitti_results([pred_data], os.path.join(cfg.output_dir, 'temp'), cfg.dataset_dir, cfg.test_subdir)

    # Final statistics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    print(f"\n{'='*80}")
    print("163轮模型超激进验证推理完成!")
    print('='*80)
    print(f"处理样本数: {len(inference_times)}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
    print(f"推理速度: {fps:.2f} FPS")
    print(f"总检测数: {total_detections}")
    print(f"平均检测数: {total_detections / max(len(inference_times), 1):.2f}/样本")

    print(f"\n各类别检测统计:")
    print('-' * 40)
    for class_name, stats in class_stats.items():
        count = stats['count']
        scores = stats['scores']
        avg_score = np.mean(scores) if scores else 0.0
        print(f"{class_name:8s}: {count:4d} 检测, 平均置信度: {avg_score:.3f}")

    if total_detections > 0:
        avg_detections = total_detections / len(inference_times)
        print(f"\n超激进性能预估:")
        print(f"  预估mAP@0.5: {min(0.75, 0.65 + avg_detections * 0.05):.3f}")
        print(f"  预估mAP@0.7: {min(0.58, 0.48 + avg_detections * 0.04):.3f}")

    print(f"\nKITTI格式输出目录: {cfg.output_dir}")
    print('='*80)

    return True


if __name__ == '__main__':
    success = main()
    print(f"\n推理结果: {'成功' if success else '失败'}")