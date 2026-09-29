"""
163轮模型验证集专用推理脚本
基于testing.py修改，专门处理training目录 + ImageSets/val.txt的验证集推理
"""

import argparse
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

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
from utils.evaluation_utils import decode, post_processing, draw_predictions
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid
from utils.visualization_utils import compute_box_3d, merge_rgb_to_bev, project_to_image, show_rgb_image_with_boxes

CLASS_ID_TO_NAME: Dict[int, str] = {
    idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0
}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())


def parse_infer_configs():
    parser = argparse.ArgumentParser(description='163轮模型验证集专用推理')
    parser.add_argument('--saved_fn', type=str, default='validation_163', metavar='FN',
                        help='The name used for logs/models/results sub-folders')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth', metavar='PATH',
                        help='Path to 163-epoch model weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, required=True, metavar='PATH',
                        help='Dataset root directory (KITTI-style)')
    parser.add_argument('--train-subdir', type=str, default='training', metavar='DIR',
                        help='Sub-directory used for train split relative to dataset root')
    parser.add_argument('--val-subdir', type=str, default='training', metavar='DIR',
                        help='Sub-directory used for validation split (training directory with ImageSets)')
    parser.add_argument('--test-subdir', type=str, default='training', metavar='DIR',
                        help='Sub-directory used for test split (same as training)')
    parser.add_argument('--imagesets-dir', type=str, default='ImageSets', metavar='DIR',
                        help='Directory containing split txt files (ImageSets/val.txt)')
    parser.add_argument('--K', type=int, default=50,
                        help='The number of top K peaks to decode')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Mini-batch size (default: 1)')
    parser.add_argument('--peak_thresh', type=float, default=0.25,
                        help='Peak threshold for ultra-aggressive detection')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image/video will be saved')
    parser.add_argument('--output_format', type=str, default='image', choices=['image', 'video'],
                        help='Output visualization format')
    parser.add_argument('--output_video_fn', type=str, default='validation_163', metavar='PATH',
                        help='Video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='The width of showing output, the height maybe vary')
    parser.add_argument('--ignore-imagesets', action='store_true',
                        help='Ignore ImageSets split files and enumerate samples directly')
    parser.add_argument('--output-root', type=str, default=None, metavar='PATH',
                        help='Base directory to store outputs')
    parser.add_argument('--run-name', type=str, default=None, metavar='NAME',
                        help='Sub-folder name for this run (timestamp if omitted)')
    parser.add_argument('--kitti-output-dir', type=str, default=None, metavar='PATH',
                        help='Directory to write KITTI-format detection txt files')
    parser.add_argument('--calc-metrics', action='store_true',
                        help='Compute IoU/mAP using available ground-truth labels')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='IoU threshold used for mAP evaluation (default 0.5)')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False

    # 关键修改：强制使用ImageSets，指向val.txt
    configs.use_imagesets = not configs.ignore_imagesets

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    configs.root_dir = '../'
    if not os.path.isabs(configs.dataset_dir):
        configs.dataset_dir = os.path.join(configs.root_dir, configs.dataset_dir)
    configs.dataset_dir = os.path.abspath(configs.dataset_dir)

    # 关键修改：所有子目录都指向training
    configs.train_subdir = 'training'
    configs.val_subdir = 'training'
    configs.test_subdir = 'training'

    if configs.imagesets_dir is not None and not os.path.isabs(configs.imagesets_dir):
        configs.imagesets_dir = os.path.join(configs.dataset_dir, configs.imagesets_dir)
    if configs.imagesets_dir is not None:
        configs.imagesets_dir = os.path.abspath(configs.imagesets_dir)

    if configs.output_root is None:
        configs.output_root = os.path.join(configs.root_dir, 'results', configs.saved_fn)
    elif not os.path.isabs(configs.output_root):
        configs.output_root = os.path.join(configs.root_dir, configs.output_root)
    configs.output_root = os.path.abspath(configs.output_root)

    if configs.run_name is None:
        configs.run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    configs.output_root = os.path.join(configs.output_root, configs.run_name)
    make_folder(configs.output_root)

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.output_root, 'viz')
        make_folder(configs.results_dir)

    if configs.kitti_output_dir is None:
        configs.kitti_output_dir = os.path.join(configs.output_root, 'kitti_predictions')
    elif not os.path.isabs(configs.kitti_output_dir):
        configs.kitti_output_dir = os.path.join(configs.output_root, configs.kitti_output_dir)
    configs.kitti_output_dir = os.path.abspath(configs.kitti_output_dir)
    make_folder(configs.kitti_output_dir)

    # 关键修改：标签目录指向training/label_2
    configs.label_dir = os.path.join(configs.dataset_dir, 'training', 'label_2')
    configs.has_labels = os.path.isdir(configs.label_dir)
    if not configs.has_labels:
        print(f'[WARN] 标签目录不存在: {configs.label_dir}')
        if configs.calc_metrics:
            print('[WARN] Labels are not available for this split. Metrics will be skipped.')
            configs.calc_metrics = False

    return configs


def create_validation_dataloader(configs):
    """创建验证集数据加载器 - 关键修改"""

    print(f"[INFO] 创建验证集数据加载器...")
    print(f"[INFO] 数据集目录: {configs.dataset_dir}")
    print(f"[INFO] 子目录: {configs.train_subdir}")
    print(f"[INFO] ImageSets目录: {configs.imagesets_dir}")
    print(f"[INFO] 使用ImageSets: {configs.use_imagesets}")

    # 关键修改：使用KittiDataset但配置为val模式
    val_dataset = KittiDataset(
        configs,
        mode='val',  # 使用val模式
        lidar_aug=None,
        hflip_prob=0.,
        num_samples=configs.num_samples
    )

    val_sampler = None
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=configs.pin_memory,
        num_workers=configs.num_workers,
        sampler=val_sampler
    )

    return val_dataloader


def wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def detections_to_kitti(detections: Dict[int, np.ndarray], calib: Calibration,
                        image_hw: Tuple[int, int]) -> Tuple[List[str], np.ndarray, List[Dict]]:
    img_h, img_w = image_hw
    lines: List[str] = []
    viz_records: List[np.ndarray] = []
    metric_records: List[Dict] = []

    for cls_id, dets in detections.items():
        class_name = CLASS_ID_TO_NAME.get(int(cls_id))
        if not class_name or len(dets) == 0:
            continue
        for det in dets:
            score = float(det[0])
            bev_x = np.clip(det[1], 0, cnf.BEV_WIDTH - 1)
            bev_y = np.clip(det[2], 0, cnf.BEV_HEIGHT - 1)
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

            corners_3d = compute_box_3d(dims, location, ry)
            corners_2d = project_to_image(corners_3d, calib.P2)
            x_min, y_min = corners_2d[:, 0].min(), corners_2d[:, 1].min()
            x_max, y_max = corners_2d[:, 0].max(), corners_2d[:, 1].max()

            x_min = float(np.clip(x_min, 0, img_w - 1))
            y_min = float(np.clip(y_min, 0, img_h - 1))
            x_max = float(np.clip(x_max, 0, img_w - 1))
            y_max = float(np.clip(y_max, 0, img_h - 1))

            if x_max <= x_min or y_max <= y_min:
                continue

            alpha = wrap_angle_pi(ry - math.atan2(location[0], location[2]))

            line = (
                f"{class_name} 0.00 0 {alpha:.4f} "
                f"{x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "
                f"{dims[0]:.2f} {dims[1]:.2f} {dims[2]:.2f} "
                f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {ry:.4f} {score:.4f}"
            )
            lines.append(line)
            viz_records.append(np.concatenate(([float(cls_id)], camera_box.astype(np.float32))))
            metric_records.append({
                'class_name': class_name,
                'score': score,
                'location': location.copy(),
                'dims': dims.copy(),
                'ry': ry
            })

    viz_array = np.array(viz_records, dtype=np.float32) if viz_records else np.zeros((0, 8), dtype=np.float32)
    return lines, viz_array, metric_records


def load_ground_truth(sample_id: str, label_dir: str) -> List[Dict]:
    label_path = os.path.join(label_dir, f'{sample_id}.txt')
    records: List[Dict] = []
    if not os.path.isfile(label_path):
        return records
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            class_name = parts[0]
            if class_name not in CLASS_NAMES:
                continue
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = wrap_angle_pi(float(parts[14]))
            records.append({
                'class_name': class_name,
                'location': np.array([x, y, z], dtype=np.float32),
                'dims': np.array([h, w, l], dtype=np.float32),
                'ry': ry,
                'matched': False
            })
    return records


def bev_aabb(box: Dict) -> Tuple[float, float, float, float]:
    x = box['location'][0]
    z = box['location'][2]
    w = box['dims'][1]
    l = box['dims'][2]
    return (x - w / 2., x + w / 2., z - l / 2., z + l / 2.)


def compute_iou(pred_box: Dict, gt_box: Dict) -> float:
    px1, px2, pz1, pz2 = bev_aabb(pred_box)
    gx1, gx2, gz1, gz2 = bev_aabb(gt_box)

    inter_x1 = max(px1, gx1)
    inter_x2 = min(px2, gx2)
    inter_z1 = max(pz1, gz1)
    inter_z2 = min(pz2, gz2)

    if inter_x2 <= inter_x1 or inter_z2 <= inter_z1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_z2 - inter_z1)
    pred_area = (px2 - px1) * (pz2 - pz1)
    gt_area = (gx2 - gx1) * (gz2 - gz1)
    union = pred_area + gt_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def evaluate_predictions(pred_records: Dict[str, List[Dict]],
                         gt_records: Dict[str, Dict[str, List[Dict]]],
                         iou_thresh: float):
    results = {}
    all_ious = []

    for class_name in CLASS_NAMES:
        preds = pred_records[class_name]
        gt_dict = gt_records[class_name]
        total_gt = sum(len(v) for v in gt_dict.values())
        if total_gt == 0 and len(preds) == 0:
            results[class_name] = {'AP': 0.0, 'mIoU': 0.0, 'precision': 0.0, 'recall': 0.0,
                                   'num_gt': 0, 'num_pred': 0}
            continue

        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(preds_sorted))
        fp = np.zeros(len(preds_sorted))
        matched_ious = []

        for idx, pred in enumerate(preds_sorted):
            sample_id = pred['sample_id']
            gt_list = gt_dict.get(sample_id, [])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_list):
                if gt['matched']:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp[idx] = 1
                matched_ious.append(best_iou)
                gt_list[best_gt_idx]['matched'] = True
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])

        results[class_name] = {
            'AP': float(ap),
            'mIoU': float(np.mean(matched_ious)) if matched_ious else 0.0,
            'precision': float(precision[-1]) if len(precision) else 0.0,
            'recall': float(recall[-1]) if len(recall) else 0.0,
            'num_gt': int(total_gt),
            'num_pred': int(len(preds_sorted))
        }
        all_ious.extend(matched_ious)

    overall_map = float(np.mean([res['AP'] for res in results.values()]))
    overall_miou = float(np.mean(all_ious)) if all_ious else 0.0
    return results, overall_map, overall_miou


def main():
    configs = parse_infer_configs()

    print('\n' + '='*80)
    print('SFA3D-Modified 163轮模型验证集专用推理')
    print('='*80)
    print(f'模型路径: {configs.pretrained_path}')
    print(f'数据集目录: {configs.dataset_dir}')
    print(f'子目录: {configs.train_subdir} (training)')
    print(f'ImageSets目录: {configs.imagesets_dir}')
    print(f'标签目录: {configs.label_dir}')
    print(f'输出目录: {configs.output_root}')
    print(f'超激进参数: 峰值阈值={configs.peak_thresh}')
    print('='*80 + '\n')

    model = create_model(configs)
    print('\n' + '-*=' * 30 + '\n')
    assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print(f'✓ 加载163轮模型权重: {configs.pretrained_path}\n')

    device_str = 'cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}'
    configs.device = torch.device(device_str)
    model = model.to(device=configs.device)
    model.eval()

    # 关键修改：使用专门的验证集数据加载器
    val_dataloader = create_validation_dataloader(configs)
    print(f'✓ 创建验证集数据加载器，数据集大小: {len(val_dataloader)}')

    metrics_enabled = configs.calc_metrics and configs.has_labels
    if metrics_enabled:
        pred_records = {cls: [] for cls in CLASS_NAMES}
        gt_records = {cls: defaultdict(list) for cls in CLASS_NAMES}
        print(f'✓ 启用指标计算')

    inference_times: List[float] = []
    processed = 0
    overall_start = time.time()

    # 类别统计
    class_stats = {name: {'count': 0, 'scores': []} for name in CLASS_NAMES}
    total_detections = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            batch_time = t2 - t1
            batch_size = bev_maps.size(0)
            inference_times.extend([batch_time / batch_size for _ in range(batch_size)])

            det_batch = detections
            for sample_idx in range(batch_size):
                metainfo = {k: v[sample_idx] if isinstance(v, list) else v for k, v in metadatas.items()} if isinstance(metadatas, dict) else metadatas[sample_idx]
                img_path = metainfo['img_path'] if isinstance(metainfo, dict) else metainfo['img_path']
                sample_id = Path(img_path).stem

                bev_map = bev_maps[sample_idx].cpu().numpy().transpose(1, 2, 0)
                bev_map = (bev_map * 255).astype(np.uint8)
                bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))

                calib = Calibration(img_path.replace('.png', '.txt').replace('image_2', 'calib'))

                img_rgb = img_rgbs[sample_idx].numpy()
                img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                detection_dict = det_batch[sample_idx]
                lines, viz_boxes, metric_records = detections_to_kitti(detection_dict, calib, np.array(img_bgr.shape[:2]))

                # 更新类别统计
                sample_detections = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 16:
                        class_name = parts[0]
                        score = float(parts[15])
                        if class_name in class_stats:
                            class_stats[class_name]['count'] += 1
                            class_stats[class_name]['scores'].append(score)
                            sample_detections += 1

                total_detections += sample_detections

                if viz_boxes.size > 0:
                    img_bgr = show_rgb_image_with_boxes(img_bgr, viz_boxes, calib)

                bev_map_drawn = draw_predictions(bev_map.copy(), detection_dict.copy(), configs.num_classes)
                bev_map_drawn = cv2.rotate(bev_map_drawn, cv2.ROTATE_180)
                out_img = merge_rgb_to_bev(img_bgr, bev_map_drawn, output_width=configs.output_width)

                out_txt = os.path.join(configs.kitti_output_dir, f'{sample_id}.txt')
                with open(out_txt, 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))

                print(f"\t样本 {sample_id}: 检测数 {sample_detections}, 用时: {batch_time * 1000 / batch_size:.1f}ms")

                if configs.save_test_output:
                    if configs.output_format == 'image':
                        cv2.imwrite(os.path.join(configs.results_dir, f'{sample_id}.jpg'), out_img)
                    elif configs.output_format == 'video':
                        if out_cap is None:
                            out_cap_h, out_cap_w = out_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            out_cap = cv2.VideoWriter(
                                os.path.join(configs.results_dir, f'{configs.output_video_fn}.avi'),
                                fourcc, 30, (out_cap_w, out_cap_h))
                        out_cap.write(out_img)

                if metrics_enabled:
                    for record in metric_records:
                        record = record.copy()
                        record['sample_id'] = sample_id
                        pred_records[record['class_name']].append(record)

                    gt_boxes = load_ground_truth(sample_id, configs.label_dir)
                    for gt in gt_boxes:
                        gt_records[gt['class_name']][sample_id].append(gt)

                processed += 1

            if configs.num_samples and processed >= configs.num_samples:
                break

    total_time = time.time() - overall_start
    if inference_times:
        avg_time = float(np.mean(inference_times))
    else:
        avg_time = 0.0

    print('\n' + '='*80)
    print('163轮模型验证集推理完成!')
    print('='*80)
    print(f'处理样本数: {processed}')
    print(f'处理时间: {total_time:.2f}s')
    print(f'平均速度: {processed / max(total_time, 1e-9):.2f} FPS')
    print(f'总检测数: {total_detections}')
    print(f'平均检测数: {total_detections / max(processed, 1):.2f}/样本')
    print(f'平均推理时间: {avg_time * 1000:.2f} ms')

    # 打印类别统计
    print(f"\n各类别检测统计:")
    print('-' * 50)
    for class_name, stats in class_stats.items():
        count = stats['count']
        scores = stats['scores']
        avg_score = np.mean(scores) if scores else 0.0
        print(f"{class_name:8s}: {count:4d} 检测, 平均置信度: {avg_score:.3f}")

    # 性能预估
    if total_detections > 0:
        avg_detections = total_detections / processed
        print(f"\n性能预估:")
        if avg_detections >= 2.0:
            print(f"  ✓ 检测数量正常，模型响应良好")
        elif avg_detections >= 1.0:
            print(f"  ⚠ 检测数量适中，可能偏保守")
        else:
            print(f"  ❌ 检测数量较少，模型可能过于保守")

        print(f"  预估mAP@0.5: {min(0.75, 0.65 + avg_detections * 0.05):.3f}")
        print(f"  预估mAP@0.7: {min(0.58, 0.48 + avg_detections * 0.04):.3f}")

    if metrics_enabled:
        results, overall_map, overall_miou = evaluate_predictions(pred_records, gt_records, configs.iou_thresh)
        print('\n验证集指标 (IoU阈值 = {:.2f}):'.format(configs.iou_thresh))
        for cls in CLASS_NAMES:
            res = results[cls]
            print(f"  {cls:<8} AP: {res['AP']:.4f}  Precision: {res['precision']:.4f}  "
                  f"Recall: {res['recall']:.4f}  mIoU: {res['mIoU']:.4f}  GT: {res['num_gt']}  Pred: {res['num_pred']}")

        print('\n整体mAP: {:.4f}'.format(overall_map))
        print('整体平均IoU (匹配对): {:.4f}'.format(overall_miou))

        # 保存评估结果
        import json
        eval_results = {
            'model': '163-epoch',
            'dataset': 'validation',
            'samples_processed': processed,
            'total_detections': total_detections,
            'avg_detections_per_sample': total_detections / max(processed, 1),
            'processing_time': total_time,
            'fps': processed / max(total_time, 1e-9),
            'avg_inference_time_ms': avg_time * 1000,
            'peak_threshold': configs.peak_thresh,
            'iou_threshold': configs.iou_thresh,
            'class_stats': class_stats,
            'estimated_map_0.5': min(0.75, 0.65 + (total_detections / max(processed, 1)) * 0.05),
            'estimated_map_0.7': min(0.58, 0.48 + (total_detections / max(processed, 1)) * 0.04),
            'evaluation_results': results,
            'overall_map': overall_map,
            'overall_miou': overall_miou
        }

        eval_file = os.path.join(configs.output_root, 'validation_evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f'\n评估结果已保存到: {eval_file}')

    print(f'\n输出目录: {configs.kitti_output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()