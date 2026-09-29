"""
点云+图像融合检测中的类别不平衡增强
专门针对DRadDataset中Car(82%) vs Truck(5%)的类别不平衡问题
"""

import numpy as np
import cv2
import random
import torch
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy

class ClassImbalanceAugmentor:
    """
    类别不平衡数据增强器
    专门处理8D雷达点云 + RGB图像的融合检测任务
    """

    def __init__(self, config=None):
        self.config = config or self.get_default_config()

        # 目标类别比例
        self.target_ratios = {
            'Car': 0.45,      # 从82%降到45%
            'Cyclist': 0.30,  # 从12%提升到30%
            'Truck': 0.25     # 从5%提升到25%
        }

        # 增强强度
        self.augmentation_strengths = {
            'Car': 1.0,       # 基准增强
            'Cyclist': 2.5,   # 2.5倍增强
            'Truck': 4.6      # 4.6倍增强
        }

        # 颜色映射
        self.class_colors = {
            'Car': (255, 0, 0),
            'Cyclist': (0, 255, 0),
            'Truck': (0, 0, 255)
        }

    def get_default_config(self):
        """获取默认配置"""
        return {
            # 点云增强参数
            'point_cloud': {
                'rotation_angles': [15, 30, 45, 60, 90, 120, 150, 180],  # 旋转角度
                'translation_range': {
                    'Truck': {'x': [-2, 2], 'y': [-5, 5]},  # 卡车移动范围更大
                    'Cyclist': {'x': [-1, 1], 'y': [-3, 3]}
                },
                'intensity_scale': [0.7, 1.3],  # 强度扰动范围
                'noise_std': 0.02  # 点云噪声
            },

            # 图像增强参数
            'image': {
                'brightness_range': [0.8, 1.2],
                'contrast_range': [0.8, 1.2],
                'saturation_range': [0.9, 1.1],
                'hue_range': [-10, 10]
            },

            # 天气增强参数
            'weather': {
                'rain_probability': 0.3,
                'fog_probability': 0.2,
                'night_probability': 0.2
            }
        }

    def augment_point_cloud_minority(self, points, labels, target_class):
        """
        少数类点云专用增强
        """
        if target_class not in labels:
            return points, labels

        # 提取少数类点云
        minority_mask = np.array([lbl == target_class for lbl in labels])
        minority_points = points[minority_mask]
        minority_indices = np.where(minority_mask)[0]

        augmented_points = [points.copy()]
        augmented_labels = [labels.copy()]

        # 计算需要增强的数量
        num_minority = len(minority_points)
        num_augment = int(num_minority * (self.augmentation_strengths[target_class] - 1))

        print(f"Enhancing {target_class}: {num_minority} -> {num_minority + num_augment}")

        for i in range(num_augment):
            # 1. 旋转增强
            angle = random.choice(self.config['point_cloud']['rotation_angles'])
            rotation_matrix = Rotation.from_euler('z', angle, degrees=True).as_matrix()

            augmented_minority_points = minority_points.copy()

            # 只旋转x, y坐标 (保持z不变)
            xy_coords = augmented_minority_points[:, :2] @ rotation_matrix[:2, :2].T
            augmented_minority_points[:, :2] = xy_coords

            # 2. 平移增强
            trans_config = self.config['point_cloud']['translation_range'][target_class]
            x_shift = random.uniform(trans_config['x'][0], trans_config['x'][1])
            y_shift = random.uniform(trans_config['y'][0], trans_config['y'][1])

            augmented_minority_points[:, 0] += x_shift
            augmented_minority_points[:, 1] += y_shift

            # 3. 强度扰动
            intensity_scale = random.uniform(*self.config['point_cloud']['intensity_scale'])
            if augmented_minority_points.shape[1] > 3:
                augmented_minority_points[:, 3] *= intensity_scale

            # 4. 添加噪声
            noise = np.random.normal(0, self.config['point_cloud']['noise_std'],
                                    augmented_minority_points.shape)
            augmented_minority_points += noise

            # 5. 合并回完整点云
            new_points = points.copy()
            new_labels = labels.copy()

            # 替换原少数类点云
            non_minority_mask = ~minority_mask
            new_points = np.vstack([points[non_minority_mask], augmented_minority_points])
            new_labels = [labels[idx] for idx, keep in enumerate(non_minority_mask) if keep]
            new_labels.extend([target_class] * len(augmented_minority_points))

            augmented_points.append(new_points)
            augmented_labels.append(new_labels)

        return augmented_points, augmented_labels

    def augment_image_minority(self, img_rgb, bbox, target_class):
        """
        少数类图像区域增强
        """
        img_aug = img_rgb.copy()

        # 提取对象区域
        x1, y1, x2, y2 = bbox
        obj_region = img_aug[y1:y2, x1:x2].copy()

        if target_class == 'Truck':
            # 卡车增强：强化边缘和结构特征
            obj_region = self.enhance_truck_features(obj_region)
        elif target_class == 'Cyclist':
            # 骑行者增强：强化形状和颜色
            obj_region = self.enhance_cyclist_features(obj_region)

        # 混合回原图
        img_aug[y1:y2, x1:x2] = obj_region

        return img_aug

    def enhance_truck_features(self, truck_region):
        """
        增强卡车特征
        """
        enhanced = truck_region.copy()

        # 1. 边缘增强
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # 2. 混合边缘信息
        alpha = 0.3  # 边缘强度
        enhanced = cv2.addWeighted(enhanced, 1-alpha, edges_colored, alpha, 0)

        # 3. 对比度增强
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

        # 4. 添加轻微噪声模拟真实感
        noise = np.random.normal(0, 5, enhanced.shape).astype(np.int8)
        enhanced = np.clip(enhanced + noise, 0, 255).astype(np.uint8)

        return enhanced

    def enhance_cyclist_features(self, cyclist_region):
        """
        增强骑行者特征
        """
        enhanced = cyclist_region.copy()

        # 1. 颜色饱和度增强
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # 增强饱和度
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 2. 形状增强
        kernel = np.ones((2, 2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

        # 3. 亮度调整
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)

        return enhanced

    def apply_weather_augmentation(self, img_rgb):
        """
        天气条件增强
        """
        weather_augs = []

        # 1. 雨天效果
        if random.random() < self.config['weather']['rain_probability']:
            rain_img = self.simulate_rain(img_rgb)
            weather_augs.append(('rain', rain_img))

        # 2. 雾天效果
        if random.random() < self.config['weather']['fog_probability']:
            fog_img = self.simulate_fog(img_rgb)
            weather_augs.append(('fog', fog_img))

        # 3. 夜间效果
        if random.random() < self.config['weather']['night_probability']:
            night_img = self.simulate_night(img_rgb)
            weather_augs.append(('night', night_img))

        return weather_augs

    def simulate_rain(self, img_rgb):
        """
        模拟雨天效果
        """
        img = img_rgb.copy()

        # 添加雨线
        rain_intensity = random.uniform(0.3, 0.7)
        num_drops = int(img.shape[0] * img.shape[1] * rain_intensity * 0.001)

        for _ in range(num_drops):
            x = random.randint(0, img.shape[1] - 1)
            y = random.randint(0, img.shape[0] - 1)
            length = random.randint(10, 30)

            # 画雨线
            cv2.line(img, (x, y), (x - 2, y + length),
                    (200, 200, 200), 1)

        # 降低亮度
        img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)

        return img

    def simulate_fog(self, img_rgb):
        """
        模拟雾天效果
        """
        img = img_rgb.copy()

        # 创建雾效果
        fog_density = random.uniform(0.2, 0.5)
        fog_layer = np.ones_like(img) * 255

        # 混合雾层
        img = cv2.addWeighted(img, 1 - fog_density, fog_layer, fog_density, 0)

        # 降低对比度
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=10)

        return img

    def simulate_night(self, img_rgb):
        """
        模拟夜间效果
        """
        img = img_rgb.copy()

        # 降低亮度
        img = cv2.convertScaleAbs(img, alpha=0.4, beta=-50)

        # 添加蓝色调
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.2, 0, 255)  # 增强蓝色通道

        return img

    def analyze_class_distribution(self, labels):
        """
        分析当前类别分布
        """
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1

        total = sum(class_counts.values())
        distribution = {cls: count/total for cls, count in class_counts.items()}

        return class_counts, distribution

    def calculate_augmentation_needs(self, current_distribution, target_samples=10000):
        """
        计算各类别需要的增强数量
        """
        current_counts = {
            cls: int(dist * target_samples)
            for cls, dist in current_distribution.items()
        }

        target_counts = {
            cls: int(target_samples * ratio)
            for cls, ratio in self.target_ratios.items()
        }

        augmentation_needs = {}
        for cls in self.target_ratios.keys():
            if cls in current_counts:
                needed = max(0, target_counts[cls] - current_counts[cls])
                if needed > 0:
                    augmentation_needs[cls] = needed

        return augmentation_needs

    def create_augmented_dataset(self, data_samples):
        """
        创建增强后的数据集
        """
        augmented_samples = []

        # 分析当前分布
        all_labels = []
        for sample in data_samples:
            all_labels.extend(sample['labels'])

        current_counts, current_dist = self.analyze_class_distribution(all_labels)
        print(f"Current distribution: {current_dist}")

        # 计算增强需求
        aug_needs = self.calculate_augmentation_needs(current_dist)
        print(f"Augmentation needs: {aug_needs}")

        # 处理每个样本
        for sample in data_samples:
            augmented_samples.append(sample)  # 保留原始样本

            # 少数类样本增强
            for cls, aug_count in aug_needs.items():
                if cls in sample['labels']:
                    # 计算该样本的增强次数
                    class_ratio = sample['labels'].count(cls) / len(sample['labels'])
                    sample_aug_count = int(aug_count * class_ratio / current_counts[cls])

                    for i in range(sample_aug_count):
                        aug_sample = self.augment_single_sample(sample, cls, i)
                        augmented_samples.append(aug_sample)

        return augmented_samples

    def augment_single_sample(self, sample, target_class, aug_id):
        """
        增强单个样本
        """
        aug_sample = copy.deepcopy(sample)

        # 1. 点云增强
        if 'points' in sample and 'labels' in sample:
            aug_points_list, aug_labels_list = self.augment_point_cloud_minority(
                sample['points'], sample['labels'], target_class
            )

            if aug_points_list:
                aug_sample['points'] = aug_points_list[0]  # 使用第一个增强版本
                aug_sample['labels'] = aug_labels_list[0]

        # 2. 图像增强
        if 'image' in sample and 'bboxes' in sample:
            for i, (bbox, label) in enumerate(zip(sample['bboxes'], sample['labels'])):
                if label == target_class:
                    aug_sample['image'] = self.augment_image_minority(
                        aug_sample['image'], bbox, target_class
                    )
                    break

        # 3. 天气增强
        if random.random() < 0.3:  # 30%概率应用天气增强
            weather_augs = self.apply_weather_augmentation(aug_sample.get('image', None))
            if weather_augs:
                aug_sample['weather_augmented'] = weather_augs

        return aug_sample

    def visualize_augmentation_effects(self, original_sample, augmented_samples, save_path=None):
        """
        可视化增强效果
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 原始样本
        if 'image' in original_sample:
            axes[0, 0].imshow(original_sample['image'])
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

        if 'points' in original_sample:
            points = original_sample['points']
            axes[0, 1].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
            axes[0, 1].set_title('Original Point Cloud')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')

        # 类别分布
        if 'labels' in original_sample:
            labels = original_sample['labels']
            class_counts = defaultdict(int)
            for label in labels:
                class_counts[label] += 1

            axes[0, 2].bar(class_counts.keys(), class_counts.values())
            axes[0, 2].set_title('Original Class Distribution')
            axes[0, 2].set_ylabel('Count')

        # 增强样本展示
        for i, aug_sample in enumerate(augmented_samples[:3]):
            if 'image' in aug_sample:
                axes[1, i].imshow(aug_sample['image'])
                axes[1, i].set_title(f'Augmented Sample {i+1}')
                axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved augmentation visualization to: {save_path}")

        plt.show()


def create_demo_augmentation():
    """
    创建演示数据并展示增强效果
    """
    print("=== Class Imbalance Augmentation Demo ===")

    # 创建增强器
    augmentor = ClassImbalanceAugmentor()

    # 模拟数据样本
    demo_samples = []

    # 模拟Car样本
    for i in range(8):
        sample = {
            'id': f'car_{i:03d}',
            'points': np.random.randn(100, 4),  # x, y, z, intensity
            'labels': ['Car'] * 3 + ['Cyclist'] * 1,
            'image': np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8),
            'bboxes': [[100, 100, 200, 200], [300, 150, 400, 250], [500, 200, 600, 300], [700, 100, 750, 150]]
        }
        demo_samples.append(sample)

    # 模拟Truck样本 (少数类)
    for i in range(1):
        sample = {
            'id': f'truck_{i:03d}',
            'points': np.random.randn(150, 4),  # 卡车点云更多
            'labels': ['Truck'] * 2 + ['Car'] * 1,
            'image': np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8),
            'bboxes': [[150, 100, 350, 300], [400, 200, 600, 400], [650, 150, 750, 250]]
        }
        demo_samples.append(sample)

    print(f"Original dataset: {len(demo_samples)} samples")

    # 分析原始分布
    all_labels = []
    for sample in demo_samples:
        all_labels.extend(sample['labels'])

    class_counts, distribution = augmentor.analyze_class_distribution(all_labels)
    print("Original class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({distribution[cls]:.2%})")

    # 创建增强数据集
    augmented_dataset = augmentor.create_augmented_dataset(demo_samples)
    print(f"Augmented dataset: {len(augmented_dataset)} samples")

    # 分析增强后分布
    aug_labels = []
    for sample in augmented_dataset:
        aug_labels.extend(sample['labels'])

    aug_counts, aug_distribution = augmentor.analyze_class_distribution(aug_labels)
    print("Augmented class distribution:")
    for cls, count in aug_counts.items():
        print(f"  {cls}: {count} ({aug_distribution[cls]:.2%})")

    # 可视化增强效果
    if len(demo_samples) > 0 and len(augmented_dataset) > len(demo_samples):
        original_sample = demo_samples[0]
        augmented_samples = augmented_dataset[len(demo_samples):len(demo_samples)+3]

        augmentor.visualize_augmentation_effects(
            original_sample, augmented_samples,
            save_path='./results/augmentation_demo.png'
        )

    print("\n=== Augmentation Demo Completed ===")


if __name__ == '__main__':
    create_demo_augmentation()