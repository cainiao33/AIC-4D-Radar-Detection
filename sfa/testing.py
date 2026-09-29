"""
增强的推理脚本：生成KITTI格式的输出、可选的指标以及可自定义的输出文件夹。
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
from data_process.kitti_dataloader import create_test_dataloader
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
    parser = argparse.ArgumentParser(description='KITTI inference with visualization, export, and optional metrics')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name used for logs/models/results sub-folders')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='Path to model weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, default=None, metavar='PATH',
                        help='Dataset root directory (KITTI-style)')
    parser.add_argument('--train-subdir', type=str, default='training', metavar='DIR',
                        help='Sub-directory used for train split relative to dataset root')
    parser.add_argument('--val-subdir', type=str, default=None, metavar='DIR',
                        help='Sub-directory used for validation split (defaults to --train-subdir)')
    parser.add_argument('--test-subdir', type=str, default='testing', metavar='DIR',
                        help='Sub-directory used for test split relative to dataset root')
    parser.add_argument('--imagesets-dir', type=str, default=None, metavar='DIR',
                        help='Optional directory containing split txt files; relative to dataset root if not absolute')
    parser.add_argument('--K', type=int, default=50,
                        help='The number of top K peaks to decode')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Mini-batch size (default: 1)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image/video will be saved')
    parser.add_argument('--output_format', type=str, default='image', choices=['image', 'video'],
                        help='Output visualization format')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='Video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='The width of showing output, the height maybe vary')
    parser.add_argument('--ignore-imagesets', action='store_true',
                        help='Ignore ImageSets split files and enumerate samples directly')
    parser.add_argument('--output-root', type=str, default=None, metavar='PATH',
                        help='Base directory to store outputs (results/<saved_fn> if omitted)')
    parser.add_argument('--run-name', type=str, default=None, metavar='NAME',
                        help='Sub-folder name for this run (timestamp if omitted)')
    parser.add_argument('--kitti-output-dir', type=str, default=None, metavar='PATH',
                        help='Directory to write KITTI-format detection txt files (relative to output root if not absolute)')
    parser.add_argument('--calc-metrics', action='store_true',
                        help='Compute IoU/mAP using available ground-truth labels')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='IoU threshold used for mAP evaluation (default 0.5)')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False

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

    configs.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if configs.dataset_dir is None:
        configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    elif not os.path.isabs(configs.dataset_dir):
        configs.dataset_dir = os.path.join(configs.root_dir, configs.dataset_dir)
    configs.dataset_dir = os.path.abspath(configs.dataset_dir)

    if configs.val_subdir is None:
        configs.val_subdir = configs.train_subdir

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

    configs.label_dir = os.path.join(configs.dataset_dir, configs.test_subdir, 'label_2')
    configs.has_labels = os.path.isdir(configs.label_dir)
    if configs.calc_metrics and not configs.has_labels:
        print('[WARN] Labels are not available for this split. Metrics will be skipped.')
        configs.calc_metrics = False

    return configs


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
            # 添加坐标裁剪修复，防止BEV坐标超出合理范围
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

    model = create_model(configs)
    print('\n' + '-*=' * 30 + '\n')
    assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print(f'Loaded weights from {configs.pretrained_path}\n')

    device_str = 'cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}'
    configs.device = torch.device(device_str)
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    test_dataloader = create_test_dataloader(configs)

    metrics_enabled = configs.calc_metrics and configs.has_labels
    if configs.calc_metrics and not configs.has_labels:
        print('[INFO] Metrics disabled because ground-truth labels are not available.')
        metrics_enabled = False

    if metrics_enabled:
        pred_records = {cls: [] for cls in CLASS_NAMES}
        gt_records = {cls: defaultdict(list) for cls in CLASS_NAMES}

    inference_times: List[float] = []
    processed = 0
    overall_start = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
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
            inference_times.extend([batch_time / batch_size] * batch_size)

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

                if viz_boxes.size > 0:
                    img_bgr = show_rgb_image_with_boxes(img_bgr, viz_boxes, calib)

                bev_map_drawn = draw_predictions(bev_map.copy(), detection_dict.copy(), configs.num_classes)
                bev_map_drawn = cv2.rotate(bev_map_drawn, cv2.ROTATE_180)
                out_img = merge_rgb_to_bev(img_bgr, bev_map_drawn, output_width=configs.output_width)

                out_txt = os.path.join(configs.kitti_output_dir, f'{sample_id}.txt')
                with open(out_txt, 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))

                print(f"\tDone testing sample {sample_id}, time: {batch_time * 1000 / batch_size:.1f}ms, "
                      f"speed {1.0 / max(batch_time / batch_size, 1e-9):.2f} FPS")

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
                    else:
                        raise TypeError

                if metrics_enabled:
                    for record in metric_records:
                        record = record.copy()
                        record['sample_id'] = sample_id
                        pred_records[record['class_name']].append(record)

                    gt_boxes = load_ground_truth(sample_id, configs.label_dir)
                    for gt in gt_boxes:
                        gt_records[gt['class_name']][sample_id].append(gt)

                if not configs.save_test_output:
                    cv2.imshow('testing-img', out_img)
                    print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                    if cv2.waitKey(0) & 0xFF == 27:
                        configs.calc_metrics = False
                        configs.save_test_output = False
                        break

                processed += 1

    total_time = time.time() - overall_start
    if inference_times:
        avg_time = float(np.mean(inference_times))
    else:
        avg_time = 0.0

    print('\nInference finished: {} samples in {:.2f}s ({:.2f} FPS)'.format(
        processed, total_time, processed / max(total_time, 1e-9)))
    print('Average per-frame inference time: {:.2f} ms'.format(avg_time * 1000))

    if metrics_enabled:
        results, overall_map, overall_miou = evaluate_predictions(pred_records, gt_records, configs.iou_thresh)
        print('\nClass-wise metrics (IoU threshold = {:.2f}):'.format(configs.iou_thresh))
        for cls in CLASS_NAMES:
            res = results[cls]
            print(f"  {cls:<8} AP: {res['AP']:.4f}  Precision: {res['precision']:.4f}  "
                  f"Recall: {res['recall']:.4f}  mIoU: {res['mIoU']:.4f}  GT: {res['num_gt']}  Pred: {res['num_pred']}")

        print('\nOverall mAP: {:.4f}'.format(overall_map))
        print('Overall mean IoU (matched pairs): {:.4f}'.format(overall_miou))

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


