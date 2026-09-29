"""
Batch KITTI inference script: runs the SFA3D model on a dataset split and exports KITTI-format txt results.
"""

import argparse
import math
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
from data_process.kitti_data_utils import Calibration
from data_process.transformation import lidar_to_camera_box
from models.model_utils import create_model
from utils.evaluation_utils import decode, post_processing
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid
from utils.visualization_utils import compute_box_3d, project_to_image


CLASS_ID_TO_NAME = {idx: name for name, idx in cnf.CLASS_NAME_TO_ID.items() if idx >= 0}


def wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def detections_to_kitti(detections, calib: Calibration, image_hw):
    img_h, img_w = image_hw
    lines = []

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
            ry = float(camera_box[6])

            if location[2] <= 0:
                continue

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

    return lines


def parse_configs():
    parser = argparse.ArgumentParser(description='Batch KITTI-format inference export')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='Name used for output sub-folders')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='Path to trained weights (.pth)')
    parser.add_argument('--dataset-dir', type=str, default=None, metavar='PATH',
                        help='Dataset root directory (KITTI layout)')
    parser.add_argument('--train-subdir', type=str, default='training', metavar='DIR',
                        help='Training split directory relative to dataset root')
    parser.add_argument('--val-subdir', type=str, default=None, metavar='DIR',
                        help='Validation split directory (defaults to --train-subdir)')
    parser.add_argument('--test-subdir', type=str, default='testing', metavar='DIR',
                        help='Testing split directory relative to dataset root')
    parser.add_argument('--imagesets-dir', type=str, default=None, metavar='DIR',
                        help='Optional ImageSets directory (relative if not absolute)')
    parser.add_argument('--K', type=int, default=50, help='Number of top-K peaks to decode')
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU only')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Process only the first N samples')
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader worker threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size')
    parser.add_argument('--peak_thresh', type=float, default=0.2, help='Score threshold for decoded peaks')
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

    cfg.root_dir = '../'

    if cfg.dataset_dir is None:
        cfg.dataset_dir = os.path.join(cfg.root_dir, 'dataset', 'kitti')
    elif not os.path.isabs(cfg.dataset_dir):
        cfg.dataset_dir = os.path.join(cfg.root_dir, cfg.dataset_dir)
    cfg.dataset_dir = os.path.abspath(cfg.dataset_dir)

    if cfg.val_subdir is None:
        cfg.val_subdir = cfg.train_subdir

    if cfg.imagesets_dir is not None and not os.path.isabs(cfg.imagesets_dir):
        cfg.imagesets_dir = os.path.join(cfg.dataset_dir, cfg.imagesets_dir)
    if cfg.imagesets_dir is not None:
        cfg.imagesets_dir = os.path.abspath(cfg.imagesets_dir)

    if cfg.output_dir is None:
        cfg.output_dir = os.path.join(cfg.root_dir, 'results', cfg.saved_fn, 'kitti_predictions_batch')
    make_folder(cfg.output_dir)

    return cfg


def main():
    configs = parse_configs()

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
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
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

            for sample_idx in range(len(metadatas['img_path'])):
                det_dict = detections[sample_idx]
                img_path = metadatas['img_path'][sample_idx]
                img_rgb = img_rgbs[sample_idx].numpy()
                img_h, img_w = img_rgb.shape[0], img_rgb.shape[1]

                calib = Calibration(img_path.replace('.png', '.txt').replace('image_2', 'calib'))
                kitti_lines = detections_to_kitti(det_dict, calib, (img_h, img_w))

                sample_id = Path(img_path).stem
                out_path = os.path.join(configs.output_dir, f'{sample_id}.txt')
                with open(out_path, 'w') as f:
                    f.write('\n'.join(kitti_lines) + ('\n' if kitti_lines else ''))

                processed += 1
            print(f"Processed batch {batch_idx} ({len(metadatas['img_path'])} samples) in {(t2 - t1) * 1000:.1f} ms")

    elapsed = time.time() - t_start
    print(f"\nFinished {processed} samples in {elapsed:.2f}s ({processed / max(elapsed, 1e-6):.2f} FPS)\n")


if __name__ == '__main__':
    main()
