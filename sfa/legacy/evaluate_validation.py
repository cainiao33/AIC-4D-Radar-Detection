"""
修改后的评估脚本，用于测试163轮模型在验证集上的性能。
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.kitti_dataset import KittiDataset
from data_process.kitti_data_utils import Calibration
from data_process.transformation import lidar_to_camera_box
from models.model_utils import create_model
from utils.evaluation_utils import decode, post_processing
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid

CLASS_ID_TO_NAME = {idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())


def wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_configs():
    parser = argparse.ArgumentParser(description='Evaluate 163-epoch model on validation set')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='Name for checkpoints/logs/results folders')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth', metavar='PATH',
                        help='Path to trained weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, required=True, metavar='PATH',
                        help='Dataset root directory (KITTI layout)')
    parser.add_argument('--train-subdir', type=str, default='training', metavar='DIR',
                        help='Training split relative path')
    parser.add_argument('--val-subdir', type=str, default='training', metavar='DIR',
                        help='Validation split (use training with ImageSets)')
    parser.add_argument('--test-subdir', type=str, default='testing', metavar='DIR',
                        help='Testing split relative path (unused here)')
    parser.add_argument('--imagesets-dir', type=str, default=None, metavar='DIR',
                        help='Optional ImageSets directory (relative if not absolute)')
    parser.add_argument('--K', type=int, default=50, help='Number of decoded peaks per sample')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of samples for quick evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size (recommend 1)')
    parser.add_argument('--peak_thresh', type=float, default=0.25, help='Score threshold for decoded peaks')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for matching (default 0.5)')

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

    cfg.root_dir = '../'
    if not os.path.isabs(cfg.dataset_dir):
        cfg.dataset_dir = os.path.join(cfg.root_dir, cfg.dataset_dir)
    cfg.dataset_dir = os.path.abspath(cfg.dataset_dir)

    return cfg


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute 3D IoU between two boxes"""
    # Simplified 2D BEV IoU calculation
    center1 = box1[:2]
    center2 = box2[:2]
    dims1 = box1[3:5][::-1]  # l, w -> w, l for BEV
    dims2 = box2[3:5][::-1]

    # Create rectangle corners
    def get_corners(center, dims, angle):
        l, w = dims
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return corners @ rot.T + center

    corners1 = get_corners(center1, dims1, box1[6])
    corners2 = get_corners(center2, dims2, box2[6])

    # Compute BEV IoU
    def polygon_area(corners):
        x, y = corners[:, 0], corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def intersection_area(corners1, corners2):
        # Simple intersection using axis-aligned bounding boxes
        min_x1, max_x1 = corners1[:, 0].min(), corners1[:, 0].max()
        min_y1, max_y1 = corners1[:, 1].min(), corners1[:, 1].max()
        min_x2, max_x2 = corners2[:, 0].min(), corners2[:, 0].max()
        min_y2, max_y2 = corners2[:, 1].min(), corners2[:, 1].max()

        inter_min_x = max(min_x1, min_x2)
        inter_max_x = min(max_x1, max_x2)
        inter_min_y = max(min_y1, min_y2)
        inter_max_y = min(max_y1, max_y2)

        if inter_min_x < inter_max_x and inter_min_y < inter_max_y:
            return (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        return 0.0

    area1 = polygon_area(corners1)
    area2 = polygon_area(corners2)
    inter_area = intersection_area(corners1, corners2)

    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-10)


def evaluate_predictions(pred_records: Dict[str, List[Dict]],
                        gt_records: Dict[str, Dict],
                        iou_thresh: float = 0.5) -> Dict:
    """Evaluate predictions against ground truth"""

    results = {}

    for class_name in CLASS_NAMES:
        if class_name not in pred_records or class_name not in gt_records:
            results[class_name] = {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}
            continue

        preds = pred_records[class_name]
        gts = gt_records[class_name]

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = set()

        # Match predictions to ground truth
        for i, pred in enumerate(preds):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_list in enumerate(gts.values()):
                if j in gt_matched:
                    continue

                for gt in gt_list:
                    pred_box = np.array([pred['x'], pred['y'], pred['z'],
                                       pred['l'], pred['w'], pred['h'], pred['yaw']])
                    gt_box = np.array([gt['x'], gt['y'], gt['z'],
                                      gt['l'], gt['w'], gt['h'], gt['yaw']])

                    iou = compute_iou_3d(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= iou_thresh:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1

        # Calculate precision and recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)
        recalls = tp_cumsum / max(len(gts), 1)

        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                ap += precisions[mask].max()
        ap /= 11.0

        final_precision = precisions[-1] if len(precisions) > 0 else 0.0
        final_recall = recalls[-1] if len(recalls) > 0 else 0.0

        total_gt = sum(len(gt_list) for gt_list in gts.values())

        results[class_name] = {
            'ap': ap,
            'precision': final_precision,
            'recall': final_recall,
            'tp': int(tp.sum()),
            'fp': int(fp.sum()),
            'fn': total_gt - len(gt_matched)
        }

    return results


def main():
    configs = parse_configs()

    print('\n' + '='*80)
    print('SFA3D-Modified 163轮模型验证集评估')
    print('='*80)
    print(f'模型: {configs.pretrained_path}')
    print(f'数据集: {configs.dataset_dir}')
    print(f'验证集: {configs.val_subdir}')
    print(f'ImageSets: {configs.imagesets_dir}')
    print(f'IoU阈值: {configs.iou_thresh}')
    print(f'峰值阈值: {configs.peak_thresh}')
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

    # Use validation mode with ImageSets
    dataset = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=False,
                            pin_memory=configs.pin_memory, num_workers=configs.num_workers)

    print(f'数据集大小: {len(dataset)}')
    print(f'批处理大小: {configs.batch_size}')
    print()

    pred_records = {cls: [] for cls in CLASS_NAMES}
    gt_records = {cls: defaultdict(list) for cls in CLASS_NAMES}

    processed = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            metadatas, bev_maps, targets = batch_data
            if isinstance(metadatas, dict):
                img_paths = metadatas['img_path']
            else:
                img_paths = [m['img_path'] for m in metadatas]

            bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

            t1 = time_synchronized()
            outputs = model(bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio,
                                        configs.peak_thresh, inter_class_nms=True, nms_thresh=0.3)
            t2 = time_synchronized()

            batch_time = t2 - t1

            # Process predictions and ground truth
            for sample_idx in range(bev_maps.size(0)):
                img_path = img_paths[sample_idx] if isinstance(img_paths, list) else img_paths[sample_idx]
                sample_id = Path(img_path).stem

                # Store predictions
                for cls_id, dets in detections[sample_idx].items():
                    class_name = CLASS_ID_TO_NAME.get(int(cls_id))
                    if not class_name:
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

                        pred_records[class_name].append({
                            'sample_id': sample_id,
                            'score': score,
                            'x': x, 'y': y, 'z': z,
                            'l': l, 'w': w, 'h': h,
                            'yaw': yaw_lidar
                        })

                # Store ground truth (from targets)
                if targets is not None:
                    for gt in targets[sample_idx]:
                        if gt.sum() == 0:
                            continue

                        class_name = CLASS_ID_TO_NAME.get(int(gt[0]))
                        if not class_name:
                            continue

                        gt_records[class_name][sample_id].append({
                            'x': float(gt[1]),
                            'y': float(gt[2]),
                            'z': float(gt[3]),
                            'l': float(gt[4]),
                            'w': float(gt[5]),
                            'h': float(gt[6]),
                            'yaw': float(gt[7])
                        })

            processed += bev_maps.size(0)
            print(f"Processed batch {batch_idx+1} ({bev_maps.size(0)} samples) in {batch_time*1000:.1f}ms - Total: {processed}")

    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print("评估完成")
    print('='*80)
    print(f"处理时间: {elapsed:.2f}s")
    print(f"处理样本: {processed}")
    print(f"平均速度: {processed / max(elapsed, 1e-9):.2f} FPS")
    print()

    # Evaluate results
    print("计算性能指标...")
    results = evaluate_predictions(pred_records, gt_records, configs.iou_thresh)

    print(f"\n{'='*80}")
    print(f"评估结果 (IoU阈值: {configs.iou_thresh})")
    print('='*80)
    print(f"{'类别':<12} {'AP':<8} {'Precision':<12} {'Recall':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print('-' * 70)

    aps = []
    for class_name in CLASS_NAMES:
        if class_name in results:
            r = results[class_name]
            print(f"{class_name:<12} {r['ap']:<8.3f} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6}")
            aps.append(r['ap'])

    map_score = np.mean(aps) if aps else 0.0
    print('-' * 70)
    print(f"{'mAP':<12} {map_score:<8.3f}")
    print('='*80 + '\n')

    return results


if __name__ == '__main__':
    main()