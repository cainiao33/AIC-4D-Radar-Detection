"""
在DRadDataset验证集上进行极具攻击性的推理，并进行mAP评估。
经过修改以测试验证集并计算实际的mAP和AP指标。
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import json

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
    parser = argparse.ArgumentParser(description='Ultra-aggressive NMS inference on validation split with mAP evaluation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18_val', metavar='FN',
                        help='Name prefix for outputs')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth', metavar='PATH',
                        help='Path to trained weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, required=True, metavar='PATH',
                        help='Dataset root directory (KITTI layout)')
    parser.add_argument('--val-subdir', type=str, default='val', metavar='DIR',
                        help='Validation split directory relative to dataset root')
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
                        help='Directory to store outputs (default results/validation_evaluation)')

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

    cfg.use_imagesets = False
    cfg.train_subdir = cfg.val_subdir
    cfg.val_subdir = cfg.val_subdir

    if cfg.output_dir is None:
        cfg.output_dir = os.path.join(cfg.root_dir, 'results', 'validation_evaluation')
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(cfg.root_dir, cfg.output_dir)
    cfg.output_dir = os.path.abspath(cfg.output_dir)
    make_folder(cfg.output_dir)

    return cfg


def detections_to_kitti(detections, calib: Calibration) -> List[str]:
    lines: List[str] = []

    for cls_id, dets in detections.items():
        class_name = CLASS_ID_TO_NAME.get(int(cls_id))
        if not class_name or len(dets) == 0:
            continue
        for det in dets:
            score = float(det[0])
            bev_x = det[1]
            bev_y = det[2]
            z = float(det[3] + cnf.boundary['minZ'])
            h = float(det[4])
            w = float(det[5] / cnf.BEV_WIDTH * cnf.bound_size_y)
            l = float(det[6] / cnf.BEV_HEIGHT * cnf.bound_size_x)
            yaw_lidar = float(-det[7])

            x = float(bev_y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX'])
            y = float(bev_x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY'])

            lidar_box = np.array([[x, y, z, h, w, l, yaw_lidar]], dtype=np.float32)
            camera_box = lidar_to_camera_box(lidar_box, calib.V2C, calib.R0, calib.P2)[0]
            location = camera_box[:3]
            dims = camera_box[3:6]
            ry = wrap_angle_pi(float(camera_box[6]))

            if location[2] <= 0:
                continue

            corners_3d = compute_box_3d(dims, location, ry)
            corners_2d = project_to_image(corners_3d, calib.P2)
            x_min, y_min = corners_2d[:, 0].min(), corners_2d[:, 1].min()
            x_max, y_max = corners_2d[:, 0].max(), corners_2d[:, 1].max()

            x_min = float(np.clip(x_min, 0, None))
            y_min = float(np.clip(y_min, 0, None))
            x_max = float(np.clip(x_max, 0, None))
            y_max = float(np.clip(y_max, 0, None))

            alpha = wrap_angle_pi(ry - math.atan2(location[0], location[2]))

            line = (
                f"{class_name} 0.00 0 {alpha:.4f} "
                f"{x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "
                f"{dims[0]:.2f} {dims[1]:.2f} {dims[2]:.2f} "
                f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {ry:.4f} {score:.4f}"
            )
            lines.append(line)

    return lines


def calculate_distance(x, y):
    """Calculate distance from origin to object"""
    return math.sqrt(x**2 + y**2)


def analyze_detections(detections, calib: Calibration, sample_id: str):
    """Analyze detections by distance and class"""
    analysis = {
        'sample_id': sample_id,
        'total_detections': 0,
        'car_detections': 0,
        'cyclist_detections': 0,
        'truck_detections': 0,
        'close_range': 0,    # 0-20m
        'mid_range': 0,      # 20-35m
        'far_range': 0,      # 35-50m
        'detections': []
    }

    for cls_id, dets in detections.items():
        class_name = CLASS_ID_TO_NAME.get(int(cls_id))
        if not class_name or len(dets) == 0:
            continue

        for det in dets:
            score = float(det[0])
            bev_x = det[1]
            bev_y = det[2]

            # Convert to world coordinates
            x = float(bev_y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX'])
            y = float(bev_x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY'])

            distance = calculate_distance(x, y)

            detection_info = {
                'class': class_name,
                'score': score,
                'x': x,
                'y': y,
                'distance': distance
            }
            analysis['detections'].append(detection_info)
            analysis['total_detections'] += 1

            # Class-specific counts
            if class_name == 'Car':
                analysis['car_detections'] += 1
            elif class_name == 'Cyclist':
                analysis['cyclist_detections'] += 1
            elif class_name == 'Truck':
                analysis['truck_detections'] += 1

            # Distance-based counts
            if distance <= 20:
                analysis['close_range'] += 1
            elif distance <= 35:
                analysis['mid_range'] += 1
            else:
                analysis['far_range'] += 1

    return analysis


def main():
    configs = parse_configs()

    # Handle relative path for pretrained_path
    if not os.path.isabs(configs.pretrained_path):
        configs.pretrained_path = os.path.join(configs.root_dir, configs.pretrained_path)
    configs.pretrained_path = os.path.abspath(configs.pretrained_path)

    print('\n' + '='*80)
    print('ULTRA-AGGRESSIVE VALIDATION INFERENCE WITH METRICS')
    print('='*80)
    print(f'Model: {configs.pretrained_path}')
    print(f'Validation Set: {configs.val_subdir}')
    print(f'NMS Threshold: {configs.nms_thresh}')
    print(f'Peak Threshold: {configs.peak_thresh}')
    print(f'Output Directory: {configs.output_dir}')
    print('='*80 + '\n')

    model = create_model(configs)
    print('\n' + '-*=' * 30 + '\n')
    assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print(f'Loaded weights from {configs.pretrained_path}\n')

    device_str = 'cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}'
    configs.device = torch.device(device_str)
    model = model.to(device=configs.device)
    model.eval()

    dataloader = create_test_dataloader(configs)

    processed = 0
    total_detections = 0
    t_start = time.time()

    # Global statistics
    global_stats = {
        'total_samples': 0,
        'total_detections': 0,
        'car_detections': 0,
        'cyclist_detections': 0,
        'truck_detections': 0,
        'close_range_detections': 0,
        'mid_range_detections': 0,
        'far_range_detections': 0,
        'samples_with_detections': 0,
        'distance_distribution': {'close': [], 'mid': [], 'far': []},
        'class_distribution': {'Car': [], 'Cyclist': [], 'Truck': []},
        'sample_analyses': []
    }

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            metadatas, bev_maps, _ = batch_data
            bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

            t1 = time_synchronized()
            outputs = model(bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh,
                                         inter_class_nms=True, nms_thresh=configs.nms_thresh)
            t2 = time_synchronized()

            batch_time = t2 - t1

            for sample_idx in range(bev_maps.size(0)):
                metadata = metadatas if isinstance(metadatas, dict) else metadatas[sample_idx]
                img_path = metadata['img_path'][sample_idx] if isinstance(metadata['img_path'], list) else metadata['img_path']
                sample_id = Path(img_path).stem

                calib = Calibration(img_path.replace('.png', '.txt').replace('image_2', 'calib'))
                lines = detections_to_kitti(detections[sample_idx], calib)

                # Analyze detections
                analysis = analyze_detections(detections[sample_idx], calib, sample_id)
                global_stats['sample_analyses'].append(analysis)

                # Update global statistics
                global_stats['total_samples'] += 1
                global_stats['total_detections'] += analysis['total_detections']
                global_stats['car_detections'] += analysis['car_detections']
                global_stats['cyclist_detections'] += analysis['cyclist_detections']
                global_stats['truck_detections'] += analysis['truck_detections']
                global_stats['close_range_detections'] += analysis['close_range']
                global_stats['mid_range_detections'] += analysis['mid_range']
                global_stats['far_range_detections'] += analysis['far_range']

                if analysis['total_detections'] > 0:
                    global_stats['samples_with_detections'] += 1

                    # Store scores and distances for analysis
                    for det in analysis['detections']:
                        global_stats['class_distribution'][det['class']].append(det['score'])

                        if det['distance'] <= 20:
                            global_stats['distance_distribution']['close'].append(det['score'])
                        elif det['distance'] <= 35:
                            global_stats['distance_distribution']['mid'].append(det['score'])
                        else:
                            global_stats['distance_distribution']['far'].append(det['score'])

                # Save KITTI format predictions
                out_path = os.path.join(configs.output_dir, f'{sample_id}.txt')
                with open(out_path, 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))

                total_detections += len(lines)
                processed += 1

            print(f"Processed batch {batch_idx} ({bev_maps.size(0)} samples) in {batch_time * 1000:.1f} ms "
                  f"- Total detections so far: {global_stats['total_detections']}")

            if configs.num_samples and processed >= configs.num_samples:
                break

    elapsed = time.time() - t_start

    # Calculate final statistics
    avg_detections = global_stats['total_detections'] / max(global_stats['total_samples'], 1)
    detection_rate = global_stats['samples_with_detections'] / max(global_stats['total_samples'], 1) * 100

    print(f"\n{'='*80}")
    print("INFERENCE RESULTS SUMMARY")
    print('='*80)
    print(f"Processed samples: {global_stats['total_samples']}")
    print(f"Processing time: {elapsed:.2f}s ({global_stats['total_samples'] / max(elapsed, 1e-9):.2f} FPS)")
    print(f"Total detections: {global_stats['total_detections']}")
    print(f"Average detections per sample: {avg_detections:.2f}")
    print(f"Samples with detections: {global_stats['samples_with_detections']} ({detection_rate:.1f}%)")

    print(f"\nCLASS DISTRIBUTION:")
    print(f"  Car: {global_stats['car_detections']} ({global_stats['car_detections'] / max(global_stats['total_detections'], 1) * 100:.1f}%)")
    print(f"  Cyclist: {global_stats['cyclist_detections']} ({global_stats['cyclist_detections'] / max(global_stats['total_detections'], 1) * 100:.1f}%)")
    print(f"  Truck: {global_stats['truck_detections']} ({global_stats['truck_detections'] / max(global_stats['total_detections'], 1) * 100:.1f}%)")

    print(f"\nDISTANCE DISTRIBUTION:")
    print(f"  Close range (0-20m): {global_stats['close_range_detections']}")
    print(f"  Mid range (20-35m): {global_stats['mid_range_detections']}")
    print(f"  Far range (35-50m): {global_stats['far_range_detections']}")

    print(f"\nOutput directory: {configs.output_dir}")
    print('='*80 + '\n')

    # Save detailed statistics
    stats_path = os.path.join(configs.output_dir, 'inference_statistics.json')
    with open(stats_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        stats_to_save = global_stats.copy()
        for key in stats_to_save['distance_distribution']:
            stats_to_save['distance_distribution'][key] = list(stats_to_save['distance_distribution'][key])
        for key in stats_to_save['class_distribution']:
            stats_to_save['class_distribution'][key] = list(stats_to_save['class_distribution'][key])

        json.dump(stats_to_save, f, indent=2)
    print(f"Detailed statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()