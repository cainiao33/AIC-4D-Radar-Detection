# SFA4D 推理指令文档

## 开始前必读

- ✅ **163 轮权重已随仓库分发**：`./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth`（48.57 MB）
- ❌ **仓库不含数据集**：`./DRadDataset` 需自备（含 `ImageSets/ testing/ training/`）；`sample_data/` 仅 3 个样本供核对格式
- 所有命令在**仓库根目录**执行，conda 环境 `sfa3d`
- ⚠️ **建议始终显式传 `--output-dir`**：不传时代码实际输出到 `<root>/超激进P/`（与参数 help 里写的 `results/<saved_fn>/ultra_aggressive` 不一致，以代码行为为准）

## 概述

本文档包含 SFA4D 的推理命令（超激进 NMS 策略）、参数说明与输出格式。

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+cu118
- CUDA 11.8+
- conda 环境：`sfa3d`

## 激活环境

```bash
source activate sfa3d
```

## 超激进 NMS 推理（推荐）

### 核心推理指令

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_ultra_aggressive --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0 --output-dir ./results/sfa4d_163_ultra_aggressive
```

### 测试推理指令（10 个样本）

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_test --num_samples 10 --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0 --output-dir ./results/sfa4d_163_test
```

### 完整数据集推理

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_full --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0 --batch_size 1 --output-dir ./results/sfa4d_163_full
```

## 参数说明

### 核心参数

- `--pretrained_path`：模型权重路径（推荐 163 轮模型，仓库已附带）
- `--dataset-dir`：数据集根目录（**必填**）
- `--saved_fn`：输出文件名前缀
- `--output-dir`：输出目录（**建议显式指定**，见开头必读）
- `--gpu_idx`：GPU 设备索引

### 超激进 NMS 参数

- `--peak_thresh 0.25`：峰值阈值，更高阈值减少误检
- `--nms_thresh 0.2`：NMS IoU 阈值，更严格去除重复检测（含跨类别 NMS）

### 其他常用参数

- `--num_samples N`：限制推理样本数量（用于快速测试）
- `--batch_size N`：批处理大小（推荐 1）
- `--no_cuda`：强制使用 CPU 推理

## 推理脚本对比

### 1. 超激进 NMS 推理（推荐）

```bash
python sfa/testing_export_ultra_aggressive.py --pretrained_path ... --dataset-dir ./DRadDataset --peak_thresh 0.25 --nms_thresh 0.2 --output-dir ./results/xxx
```

**特点**：`peak_thresh=0.25`（高精度）+ `nms_thresh=0.2`（去重严格），适用于高精度要求场景。

### 2. 标准推理

```bash
python sfa/testing.py --pretrained_path ... --dataset-dir ./DRadDataset
```

**特点**：默认参数，平衡精度和召回率，支持 `--calc-metrics` 计算指标。

### 3. 验证集推理

```bash
python sfa/validation_ultra_aggressive_163.py
```

**特点**：专门用于 163 轮模型验证集评估（路径以其文件内配置为准）。

### 4. ONNX Runtime 推理（CPU 可跑）

```bash
pip install onnxruntime && python sfa/run_onnx_inference.py --onnx_model ./onnx_models/sfa3d_163_fp32.onnx --dataset-dir ./DRadDataset --imagesets-dir ./DRadDataset/ImageSets
```

**特点**：跨平台（无需 GPU 与 PyTorch），CPU 约 5.48 FPS；后处理为纯 NumPy 实现。

## 性能口径说明（三个数字的场景不同，勿混用）

| 数字 | 场景 |
|------|------|
| **110.73 FPS** | GPU 纯推理，RTX 4060 Ti，PyTorch FP32 |
| **~2.8 FPS** | 端到端含 I/O 的日志口径（Windows 验证脚本实测，见下方日志示例） |
| **5.48 FPS** | ONNX Runtime，CPU，1034 样本平均（182 ms/样本） |

## 输出结果

### 目录结构（显式传 `--output-dir ./results/<saved_fn>` 时）

```
results/<saved_fn>/
├── kitti_predictions/    # KITTI 格式检测结果 .txt（每样本一个）
└── viz/                  # 可视化结果（如启用 --save_test_output）
```

不传 `--output-dir` 时输出到 `<root>/超激进P/`（代码实际默认值）。

### 输出格式

- 检测结果：KITTI 格式 `.txt`
- 格式：`[类别] [截断] [遮挡] [角度] [边界框] [维度] [位置] [旋转] [得分]`

## 模型信息

### 163 轮模型（推荐，仓库已附带）

**路径**：`./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth`

- 经过完整 300 轮训练，第 163 轮达到最佳性能
- 针对 8D 毫米波雷达数据优化
- 支持Car、Cyclist、Truck 三类检测

### 自训练模型

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/your_model/Model_your_model_best.pth --dataset-dir ./DRadDataset --saved_fn your_inference --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0 --output-dir ./results/your_inference
```

## 数据处理说明

```
8D 数据 [x,y,z,D,P,R,A,E]
→ 读取 [0,1,2,4] 维度
→ P 映射为 intensity
→ 输出 4D [x,y,z,intensity]
```

- **0/1/2**：X/Y/Z 坐标
- **4**：SNR 强度值 → intensity

## 故障排除

1. **CUDA 内存不足**：`--batch_size 1`
2. **模型路径错误**：`ls -la ./checkpoints/sfa3d_8d_full_300epochs/`
3. **数据集路径错误**：`ls -la ./DRadDataset/testing/velodyne/`
4. **ImageSets 缺失**：脚本会尝试直接枚举 `velodyne/` 目录，但建议提供完整 `ImageSets/`

## 日志示例

```
================================================================================
ULTRA-AGGRESSIVE NMS INFERENCE MODE
================================================================================
NMS Threshold: 0.2 (lower = more aggressive)
Peak Threshold: 0.25 (higher = fewer detections)
Inter-class NMS: Enabled
================================================================================

Processed batch 0 (1 samples) in 492.7 ms
...
Finished 10 samples in 3.54s (2.82 FPS)
Total detections: 10
Average detections per sample: 1.00
```

## 版本信息

- **创建日期**：2025-11-05（2026-09 修订：输出目录默认值、性能口径、与仓库实际文件对齐）
- **推理引擎**：PyTorch 2.0.0+cu118 / ONNX Runtime
- **推荐模型**：163 轮训练模型（已随仓库分发）

---

**注意**：
1. 推理前确保已激活 sfa3d 虚拟环境
2. 推荐使用 163 轮模型以获得最佳性能
3. 后续可视化见《启动可视化指令》
