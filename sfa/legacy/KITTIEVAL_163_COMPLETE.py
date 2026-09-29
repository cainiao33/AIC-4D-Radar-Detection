#!/usr/bin/env python3
"""
KITTI标准完整评测流程 - 163轮模型
1. 运行推理生成预测文件
2. 使用KITTI标准评估mAP
3. 生成详细报告
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ 命令执行成功")
        print("输出:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
        print(f"返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("="*80)
    print("SFA3D-Modified 163轮模型 KITTI标准完整评测")
    print("="*80)
    print(f"评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 设置路径
    project_root = Path(__file__).parent.parent
    sfa_dir = project_root / "sfa"
    model_path = project_root / "checkpoints" / "sfa3d_8d_full_300epochs" / "Model_sfa3d_8d_full_300epochs_epoch_163.pth"
    dataset_dir = project_root / "DRadDataset"
    results_dir = project_root / "results" / "kitti_eval_163"

    # 检查文件
    print(f"\n检查关键文件:")
    print(f"  项目根目录: {project_root}")
    print(f"  模型文件: {model_path} (存在: {model_path.exists()})")
    print(f"  数据集目录: {dataset_dir} (存在: {dataset_dir.exists()})")

    if not model_path.exists():
        print("❌ 模型文件不存在!")
        return False

    # 创建结果目录
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = results_dir / "predictions"

    # Step 1: 运行推理
    print(f"\n{'='*80}")
    print("步骤1: 运行163轮模型推理")
    print('='*80)

    # 使用已验证可工作的命令
    inference_cmd = (
        f'cd "{sfa_dir}" && python testing.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--val-subdir "training" '
        f'--imagesets-dir "{dataset_dir / "ImageSets"}" '
        f'--pretrained_path "{model_path}" '
        f'--calc-metrics '
        f'--iou-thresh 0.5 '
        f'--num_samples 50 '
        f'--peak_thresh 0.25 '
        f'--save_test_output '
        f'--kitti-output-dir "{pred_dir}"'
    )

    if not run_command(inference_cmd, "163轮模型推理"):
        print("❌ 推理失败，请检查模型和数据路径")
        return False

    # Step 2: 检查预测文件
    print(f"\n{'='*80}")
    print("步骤2: 检查预测文件")
    print('='*80)

    pred_files = list(pred_dir.glob("*.txt"))
    print(f"生成预测文件数量: {len(pred_files)}")

    if len(pred_files) == 0:
        print("❌ 没有生成预测文件!")
        return False

    # 统计预测结果
    total_detections = 0
    class_counts = {'Car': 0, 'Cyclist': 0, 'Truck': 0}

    for pred_file in pred_files[:5]:  # 检查前5个文件
        with open(pred_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        class_name = parts[0]
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            total_detections += 1

    print(f"前5个文件检测统计:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} 个")
    print(f"  总计: {total_detections} 个")

    # Step 3: KITTI标准评估
    print(f"\n{'='*80}")
    print("步骤3: KITTI标准mAP评估")
    print('='*80)

    gt_dir = dataset_dir / "split" / "val" / "label_2"
    eval_output = results_dir / "kitti_evaluation.json"

    # 使用我们创建的KITTI评估脚本
    eval_cmd = (
        f'cd "{sfa_dir}" && python kitti_evaluation_163.py '
        f'--pred_dir "{pred_dir}" '
        f'--gt_dir "{gt_dir}" '
        f'--iou_thresh 0.5 0.7 '
        f'--output_file "{eval_output}"'
    )

    if not run_command(eval_cmd, "KITTI标准评估"):
        print("❌ 评估失败")
        return False

    # Step 4: 生成报告
    print(f"\n{'='*80}")
    print("步骤4: 生成评测报告")
    print('='*80)

    if eval_output.exists():
        with open(eval_output, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print("KITTI评测结果:")
        print("-" * 50)

        for iou_key in ['iou_0.5', 'iou_0.7']:
            if iou_key in results:
                iou_val = iou_key.split('_')[1]
                result = results[iou_key]
                print(f"\nmAP@{iou_val}: {result['overall']['map']:.4f}")
                print(f"Overall: P={result['overall']['precision']:.3f}, R={result['overall']['recall']:.3f}, F1={result['overall']['f1']:.3f}")

                print("各类别详细性能:")
                for class_name in ['Car', 'Cyclist', 'Truck']:
                    if class_name in result['classes']:
                        cls_result = result['classes'][class_name]
                        print(f"  {class_name:8s}: AP={cls_result['ap']:.4f}, P={cls_result['precision']:.3f}, R={cls_result['recall']:.3f}")

        # 生成技术方案对比
        print(f"\n{'='*80}")
        print("与技术方案声称值对比")
        print('='*80)

        if 'iou_0.5' in results:
            actual_map = results['iou_0.5']['overall']['map']
            claimed_map = 0.75

            print(f"mAP@0.5 对比:")
            print(f"  技术方案声称: {claimed_map:.3f}")
            print(f"  实际测试结果: {actual_map:.3f}")
            print(f"  差异: {actual_map - claimed_map:+.3f}")

            # 各类别对比
            print(f"\n各类别AP@0.5对比:")
            claimed_aps = {'Car': 0.85, 'Cyclist': 0.68, 'Truck': 0.72}

            for class_name in ['Car', 'Cyclist', 'Truck']:
                if class_name in results['iou_0.5']['classes']:
                    actual_ap = results['iou_0.5']['classes'][class_name]['ap']
                    claimed_ap = claimed_aps[class_name]
                    print(f"  {class_name:8s}: 声称={claimed_ap:.3f}, 实际={actual_ap:.3f}, 差异={actual_ap - claimed_ap:+.3f}")

    print(f"\n{'='*80}")
    print("KITTI标准评测完成!")
    print(f"结果文件:")
    print(f"  预测文件: {pred_dir}")
    print(f"  评估结果: {eval_output}")
    print('='*80)

    return True

if __name__ == "__main__":
    import time
    success = main()
    print(f"\n评测结果: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)