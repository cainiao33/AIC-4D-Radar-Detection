"""
Simple Class Imbalance Augmentation Demo
针对DRadDataset的类别不平衡问题 (Car: 82%, Truck: 5%)
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class SimpleClassImbalanceAugmentor:
    """简化的类别不平衡增强器"""

    def __init__(self):
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

    def simulate_point_cloud_augmentation(self, num_points, target_class):
        """
        模拟点云增强
        """
        original_points = np.random.randn(num_points, 4)  # x, y, z, intensity

        augmented_points = [original_points]

        # 根据类别决定增强强度
        aug_count = int(self.augmentation_strengths[target_class] - 1)

        for i in range(aug_count):
            # 旋转
            angle = random.choice([30, 60, 90, 120, 150])
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])

            aug_points = original_points.copy()
            xy = aug_points[:, :2] @ rotation_matrix.T
            aug_points[:, :2] = xy

            # 平移
            if target_class == 'Truck':
                x_shift = random.uniform(-2, 2)
                y_shift = random.uniform(-5, 5)
            else:
                x_shift = random.uniform(-1, 1)
                y_shift = random.uniform(-3, 3)

            aug_points[:, 0] += x_shift
            aug_points[:, 1] += y_shift

            # 强度扰动
            intensity_scale = random.uniform(0.7, 1.3)
            aug_points[:, 3] *= intensity_scale

            augmented_points.append(aug_points)

        return augmented_points

    def simulate_image_augmentation(self, image_shape, target_class):
        """
        模拟图像增强
        """
        original_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
        augmented_images = [original_image]

        aug_count = int(self.augmentation_strengths[target_class] - 1)

        for i in range(aug_count):
            aug_image = original_image.copy()

            if target_class == 'Truck':
                # 卡车增强：边缘增强
                gray = cv2.cvtColor(aug_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                aug_image = cv2.addWeighted(aug_image, 0.7, edges_colored, 0.3, 0)
                aug_image = cv2.convertScaleAbs(aug_image, alpha=1.2, beta=10)

            elif target_class == 'Cyclist':
                # 骑行者增强：颜色增强
                hsv = cv2.cvtColor(aug_image, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
                aug_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                aug_image = cv2.convertScaleAbs(aug_image, alpha=1.1, beta=5)

            # 天气效果
            if random.random() < 0.3:
                # 雨天效果
                rain_intensity = random.uniform(0.2, 0.5)
                aug_image = cv2.convertScaleAbs(aug_image, alpha=0.8, beta=-20)

            augmented_images.append(aug_image)

        return augmented_images

    def analyze_class_distribution(self, labels):
        """分析类别分布"""
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1

        total = sum(class_counts.values())
        distribution = {cls: count/total for cls, count in class_counts.items()}

        return class_counts, distribution

    def calculate_augmentation_plan(self, current_distribution, total_samples=5000):
        """计算增强计划"""
        current_counts = {
            cls: int(dist * total_samples)
            for cls, dist in current_distribution.items()
        }

        target_counts = {
            cls: int(total_samples * ratio)
            for cls, ratio in self.target_ratios.items()
        }

        plan = {}
        for cls in self.target_ratios.keys():
            current = current_counts.get(cls, 0)
            target = target_counts[cls]
            needed = max(0, target - current)

            if needed > 0:
                plan[cls] = {
                    'current': current,
                    'target': target,
                    'needed': needed,
                    'augmentation_factor': needed / max(current, 1)
                }

        return plan

    def create_demo_dataset(self):
        """创建演示数据集"""
        print("=== 创建演示数据集 ===")

        # 模拟原始数据集分布 (接近真实DRadDataset)
        dataset = []

        # Car样本 (82%)
        for i in range(82):
            sample = {
                'id': f'car_{i:03d}',
                'class': 'Car',
                'points': np.random.randn(100, 4),
                'image': np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8),
                'confidence': random.uniform(0.8, 0.95)
            }
            dataset.append(sample)

        # Cyclist样本 (12%)
        for i in range(12):
            sample = {
                'id': f'cyclist_{i:03d}',
                'class': 'Cyclist',
                'points': np.random.randn(50, 4),
                'image': np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8),
                'confidence': random.uniform(0.6, 0.85)
            }
            dataset.append(sample)

        # Truck样本 (5%)
        for i in range(5):
            sample = {
                'id': f'truck_{i:03d}',
                'class': 'Truck',
                'points': np.random.randn(150, 4),
                'image': np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8),
                'confidence': random.uniform(0.5, 0.75)
            }
            dataset.append(sample)

        return dataset

    def apply_augmentation(self, dataset):
        """应用数据增强"""
        print("\n=== 应用数据增强 ===")

        # 分析原始分布
        labels = [sample['class'] for sample in dataset]
        original_counts, original_dist = self.analyze_class_distribution(labels)

        print("原始分布:")
        for cls, count in original_counts.items():
            print(f"  {cls}: {count} ({original_dist[cls]:.1%})")

        # 计算增强计划
        aug_plan = self.calculate_augmentation_plan(original_dist)

        print("\n增强计划:")
        for cls, plan in aug_plan.items():
            print(f"  {cls}: {plan['current']} -> {plan['target']} (需要+{plan['needed']}, 增强倍数: {plan['augmentation_factor']:.1f}x)")

        # 应用增强
        augmented_dataset = dataset.copy()

        for sample in dataset:
            if sample['class'] in aug_plan:
                # 计算该样本需要增强的次数
                cls = sample['class']
                aug_factor = aug_plan[cls]['augmentation_factor']

                # 简化：每个少数类样本增强固定次数
                if cls == 'Truck':
                    aug_times = 4  # 每个Truck样本增强4次
                elif cls == 'Cyclist':
                    aug_times = 2  # 每个Cyclist样本增强2次
                else:
                    aug_times = 1  # Car样本轻度增强

                for i in range(aug_times):
                    aug_sample = sample.copy()
                    aug_sample['id'] = f"{sample['id']}_aug_{i+1}"

                    # 点云增强
                    aug_points = self.simulate_point_cloud_augmentation(
                        len(sample['points']), cls
                    )[0]  # 使用第一个增强版本
                    aug_sample['points'] = aug_points

                    # 图像增强
                    aug_images = self.simulate_image_augmentation(
                        sample['image'].shape, cls
                    )
                    aug_sample['image'] = aug_images[0]

                    # 调整置信度 (增强样本通常置信度稍低)
                    aug_sample['confidence'] *= 0.95

                    augmented_dataset.append(aug_sample)

        return augmented_dataset

    def visualize_results(self, original_dataset, augmented_dataset):
        """可视化结果"""
        print("\n=== 可视化增强效果 ===")

        # 分析增强后分布
        original_labels = [sample['class'] for sample in original_dataset]
        augmented_labels = [sample['class'] for sample in augmented_dataset]

        original_counts, original_dist = self.analyze_class_distribution(original_labels)
        augmented_counts, augmented_dist = self.analyze_class_distribution(augmented_labels)

        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 原始分布柱状图
        classes = list(original_counts.keys())
        original_values = [original_counts[cls] for cls in classes]

        ax1.bar(classes, original_values, color=['red', 'green', 'blue'])
        ax1.set_title('原始类别分布')
        ax1.set_ylabel('样本数量')
        for i, v in enumerate(original_values):
            ax1.text(i, v + 1, f'{v}\n({original_dist[cls]:.1%})', ha='center')

        # 2. 增强后分布柱状图
        augmented_values = [augmented_counts[cls] for cls in classes]

        ax2.bar(classes, augmented_values, color=['red', 'green', 'blue'])
        ax2.set_title('增强后类别分布')
        ax2.set_ylabel('样本数量')
        for i, v in enumerate(augmented_values):
            ax2.text(i, v + 10, f'{v}\n({augmented_dist[cls]:.1%})', ha='center')

        # 3. 原始vs增强对比
        x = np.arange(len(classes))
        width = 0.35

        ax3.bar(x - width/2, original_values, width, label='原始', color=['red', 'green', 'blue'])
        ax3.bar(x + width/2, augmented_values, width, label='增强后', color=['red', 'green', 'blue'], alpha=0.7)
        ax3.set_title('原始 vs 增强后对比')
        ax3.set_ylabel('样本数量')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes)
        ax3.legend()

        # 4. 百分比对比 (饼图)
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 6))

        # 原始分布饼图
        ax5.pie(original_values, labels=classes, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
        ax5.set_title('原始分布百分比')

        # 增强后分布饼图
        ax6.pie(augmented_values, labels=classes, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
        ax6.set_title('增强后分布百分比')

        plt.tight_layout()

        # 保存图表
        save_path = './results/class_imbalance_augmentation_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")

        plt.show()

        # 打印统计信息
        print(f"\n=== 统计摘要 ===")
        print(f"原始数据集: {len(original_dataset)} 样本")
        print(f"增强后数据集: {len(augmented_dataset)} 样本")
        print(f"增强倍数: {len(augmented_dataset) / len(original_dataset):.1f}x")

        print(f"\n类别分布变化:")
        for cls in classes:
            original_pct = original_dist.get(cls, 0) * 100
            augmented_pct = augmented_dist.get(cls, 0) * 100
            change = augmented_pct - original_pct
            print(f"  {cls}: {original_pct:.1f}% -> {augmented_pct:.1f}% ({change:+.1f}%)")

        # 计算不平衡度改进
        original_imbalance = max(original_dist.values()) - min(original_dist.values())
        augmented_imbalance = max(augmented_dist.values()) - min(augmented_dist.values())
        improvement = (original_imbalance - augmented_imbalance) / original_imbalance * 100

        print(f"\n不平衡度改进: {improvement:.1f}%")
        print(f"原始不平衡度: {original_imbalance:.1%}")
        print(f"增强后不平衡度: {augmented_imbalance:.1%}")


def main():
    """主函数"""
    print("=== 类别不平衡数据增强演示 ===")

    # 创建增强器
    augmentor = SimpleClassImbalanceAugmentor()

    # 创建演示数据集
    original_dataset = augmentor.create_demo_dataset()

    # 应用增强
    augmented_dataset = augmentor.apply_augmentation(original_dataset)

    # 可视化结果
    augmentor.visualize_results(original_dataset, augmented_dataset)

    print("\n=== 增强演示完成 ===")


if __name__ == '__main__':
    main()