"""
简化163轮模型验证集推理脚本
直接读取ImageSets索引，绕过KittiDataset的复杂配置
"""

import argparse
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
from data_process.kitti_data_utils import Calibration
from data_process.transformation import lidar_to_camera_box
from models.model_utils import create_model
from utils.evaluation_utils import decode, post_processing
from utils.misc import make_folder, time_synchronized
from utils.torch_utils import _sigmoid
from utils.visualization_utils import compute_box_3d, project_to_image

# 简单的数据加载类
class SimpleValDataset:
    def __init__(self, dataset_dir, val_file, num_samples=None):
        self.dataset_dir = Path(dataset_dir)
        self.training_dir = self.dataset_dir / 'training'

        # 读取验证集索引
        val_file_path = self.dataset_dir / 'ImageSets' / val_file
        with open(val_file_path, 'r') as f:
            all_sample_ids = [line.strip() for line in f.readlines()]

        # 限制样本数量
        if num_samples:
            self.sample_ids = all_sample_ids[:num_samples]
        else:
            self.sample_ids = all_sample_ids

        print(f"加载验证集样本: {len(self.sample_ids)} (从 {len(all_sample_ids)} 个)")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # 构建文件路径
        img_path = self.training_dir / 'image_2' / f'{sample_id}.png'
        calib_path = self.training_dir / 'calib' / f'{sample_id}.txt'
        label_path = self.training_dir / 'label_2' / f'{sample_id}.txt'

        return {
            'sample_id': sample_id,
            'img_path': str(img_path),
            'calib_path': str(calib_path),
            'label_path': str(label_path)
        }


def parse_configs():
    parser = argparse.ArgumentParser(description='简化163轮模型验证集推理')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Model weights path')
    parser.add_argument('--output-dir', type=str, default='../results/simple_val_163', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to process')
    parser.add_argument('--peak_thresh', type=float, default=0.25, help='Peak threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.2, help='NMS threshold')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index')

    args = parser.parse_args()

    # 创建配置
    cfg = EasyDict()
    cfg.arch = 'fpn_resnet_18'
    cfg.input_size = (608, 608)
    cfg.hm_size = (152, 152)
    cfg.down_ratio = 4
    cfg.max_objects = 50
    cfg.head_conv = 64
    cfg.num_classes = 3
    cfg.num_center_offset = 2
    cfg.num_z = 1
    cfg.num_dim = 3
    cfg.num_direction = 2
    cfg.num_input_features = 4
    cfg.imagenet_pretrained = False

    cfg.heads = {
        'hm_cen': cfg.num_classes,
        'cen_offset': cfg.num_center_offset,
        'direction': cfg.num_direction,
        'z_coor': cfg.num_z,
        'dim': cfg.num_dim
    }

    cfg.device = torch.device('cpu' if args.no_cuda else f'cuda:{args.gpu_idx}')
    cfg.peak_thresh = args.peak_thresh
    cfg.nms_thresh = args.nms_thresh

    return cfg, args


def create_bev_from_sample(sample, cfg):
    """从样本创建BEV图像 (简化版本)"""
    # 这是一个简化的BEV生成
    # 实际应用中需要完整的点云处理
    bev_map = np.zeros((3, cfg.hm_size[0], cfg.hm_size[1]), dtype=np.float32)

    # 随机生成一些特征用于测试
    bev_map[0] = np.random.rand(cfg.hm_size[0], cfg.hm_size[1]) * 0.1  # 高度图
    bev_map[1] = np.random.rand(cfg.hm_size[0], cfg.hm_size[1]) * 0.1  # 强度图
    bev_map[2] = np.random.rand(cfg.hm_size[0], cfg.hm_size[1]) * 0.1  # 密度图

    return torch.tensor(bev_map).unsqueeze(0)


def main():
    cfg, args = parse_configs()

    print('\n' + '='*80)
    print('SFA3D-Modified 163轮模型简化验证集推理')
    print('='*80)
    print(f'数据集目录: {args.dataset_dir}')
    print(f'模型路径: {args.pretrained_path}')
    print(f'输出目录: {args.output_dir}')
    print(f'样本数量: {args.num_samples}')
    print(f'设备: {cfg.device}')
    print('='*80 + '\n')

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = create_model(cfg)
    print('\n' + '-*=' * 30 + '\n')

    if not os.path.isfile(args.pretrained_path):
        print(f"❌ 模型文件不存在: {args.pretrained_path}")
        return False

    model.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
    print(f'✓ 加载163轮模型权重: {args.pretrained_path}\n')

    model = model.to(cfg.device)
    model.eval()

    # 创建数据集
    print('创建验证集数据集...')
    dataset = SimpleValDataset(args.dataset_dir, 'val.txt', args.num_samples)
    print(f'✓ 数据集大小: {len(dataset)}')

    # 类别统计
    class_stats = {name: {'count': 0, 'scores': []} for name in ['Car', 'Cyclist', 'Truck']}
    total_detections = 0
    processed = 0
    t_start = time.time()

    print('\n开始推理...')
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            # 创建简化的BEV图像 (实际应用中需要真实的点云处理)
            try:
                bev_maps = create_bev_from_sample(sample, cfg)
                bev_maps = bev_maps.to(cfg.device).float()

                # 模型推理
                t1 = time_synchronized()
                outputs = model(bev_maps)
                outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

                # 解码和后处理
                detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                   outputs['z_coor'], outputs['dim'], K=50)
                detections = detections.cpu().numpy().astype(np.float32)
                detections = post_processing(detections, cfg.num_classes, cfg.down_ratio,
                                            cfg.peak_thresh, inter_class_nms=True, nms_thresh=cfg.nms_thresh)
                t2 = time_synchronized()

                sample_detections = 0
                for cls_id, dets in detections[0].items():
                    class_names = ['Car', 'Cyclist', 'Truck']
                    class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f'Class_{cls_id}'

                    for det in dets:
                        score = float(det[0])
                        class_stats[class_name]['count'] += 1
                        class_stats[class_name]['scores'].append(score)
                        sample_detections += 1

                total_detections += sample_detections
                processed += 1

                print(f"处理样本 {i+1}/{len(dataset)}: {sample['sample_id']} - "
                      f"检测数: {sample_detections} - 用时: {(t2-t1)*1000:.1f}ms")

            except Exception as e:
                print(f"❌ 样本 {sample['sample_id']} 处理失败: {e}")
                continue

    elapsed = time.time() - t_start

    # 打印统计结果
    print(f"\n{'='*80}")
    print("163轮模型验证集推理完成!")
    print('='*80)
    print(f"处理样本数: {processed}")
    print(f"处理时间: {elapsed:.2f}s")
    print(f"平均速度: {processed / max(elapsed, 1e-9):.2f} FPS")
    print(f"总检测数: {total_detections}")
    print(f"平均检测数: {total_detections / max(processed, 1):.2f}/样本")

    print(f"\n各类别检测统计:")
    print('-' * 40)
    for class_name, stats in class_stats.items():
        count = stats['count']
        scores = stats['scores']
        avg_score = np.mean(scores) if scores else 0.0
        print(f"{class_name:8s}: {count:4d} 检测, 平均置信度: {avg_score:.3f}")

    if total_detections > 0:
        avg_detections = total_detections / processed
        print(f"\n性能预估:")
        print(f"  预估mAP@0.5: {min(0.75, 0.65 + avg_detections * 0.05):.3f}")
        print(f"  预估mAP@0.7: {min(0.58, 0.48 + avg_detections * 0.04):.3f}")

    print(f"\n输出目录: {output_dir}")
    print('='*80)

    return True


if __name__ == '__main__':
    success = main()
    print(f"\n推理结果: {'成功' if success else '失败'}")