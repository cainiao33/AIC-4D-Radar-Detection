# AGENTS.md — SFA4D 项目指引

> 本文档供 AI 编程助手阅读。假设读者对项目一无所知，所有信息均基于实际代码和文档整理，不添加推测。

---

## 1. 项目概述

**SFA4D** 是一个基于 PyTorch 的 3D 目标检测项目，专门面向 **8 维毫米波雷达点云数据**（雷达单模态，图片不进网络）。它在原始 SFA3D（Super Fast and Accurate 3D Object Detection）基础上增强——**网络骨干/损失/BEV 生成沿用 SFA3D 原版**（KFPN、无锚点检测头均为 SFA3D 原有设计，逐文件差异见 `docs/SFA3D与SFA4D源码比对报告.md`），本项目原创改动包括：

- **8D → 4D 智能映射（原创核心）**：从 8 维雷达点云 `[x, y, z, Doppler, P, Range, Azimuth, Elevation]` 中选取第 5 维信噪比强度（P）分段映射为 intensity，输出 4D 点云 `[x, y, z, intensity]`（`data_process/lidar_mapping.py`）。
- **KFPN 特征融合**：SFA3D 原有结构——在 FPN 基础上 softmax 注意力加权的多尺度特征融合（`models/fpn_resnet.py` 与原版逐字节相同，此处列出仅为说明结构）。
- **跨类别 NMS**：在后处理阶段对不同类别之间的重复检测框进行抑制。
- **无锚点检测（Anchor-Free）**：基于 CenterNet 思想，直接预测目标中心热力图、偏移、尺寸、方向角和 Z 坐标。

支持类别：Car、Cyclist、Truck（共 3 类）。

---

## 2. 技术栈与运行环境

| 组件 | 版本/说明 |
|------|-----------|
| Python | 3.8+（已验证 3.8.20） |
| PyTorch | 2.0.0+cu118（实际运行环境）；requirements.txt 中标注为 1.5.0 |
| CUDA | 11.8+（推荐），最低 10.1 |
| OpenCV | 4.2.0.34 |
| NumPy | 1.24.3（实际）/ 1.18.3（requirements.txt） |
| 其他关键库 | easydict, tqdm, tensorboard, matplotlib, scikit-learn |
| 可选 | onnxruntime（ONNX 推理）、spconv-cu118（稀疏卷积） |

**操作系统**：Windows 10/11（当前开发环境）、Linux Ubuntu 18.04+。

---

## 3. 项目目录结构

```
SFA4D/
├── sfa/                          # 核心代码目录
│   ├── config/                   # 配置文件
│   │   ├── kitti_config.py       # 数据集参数、BEV 边界、类别映射、标定矩阵
│   │   └── train_config.py       # 训练参数解析（argparse + EasyDict）
│   ├── data_process/             # 数据处理
│   │   ├── kitti_dataset.py      # PyTorch Dataset 实现
│   │   ├── kitti_dataloader.py   # DataLoader 工厂函数
│   │   ├── kitti_data_utils.py   # 点云过滤、热力图生成、标定读取
│   │   ├── kitti_bev_utils.py    # BEV 投影图生成（height/intensity/density 三通道）
│   │   ├── transformation.py     # 数据增强（旋转、缩放、水平翻转、坐标变换）
│   │   └── lidar_mapping.py      # 8D → 4D 映射实现（Scheme B）
│   ├── models/                   # 模型定义
│   │   ├── fpn_resnet.py         # ResNet-18 + FPN + KFPN 主网络（实际使用）
│   │   ├── resnet.py             # 原始 ResNet + 反卷积头（备用）
│   │   └── model_utils.py        # 模型创建、DataParallel/DistributedDataParallel 包装
│   ├── losses/                   # 损失函数
│   │   └── losses.py             # FocalLoss + L1Loss + Balanced L1Loss，多任务加权求和
│   ├── utils/                    # 工具函数
│   │   ├── evaluation_utils.py   # 解码、NMS、后处理、坐标转换（含 NumPy 版本供 ONNX 使用）
│   │   ├── evaluation_utils_current_improved.py  # 改进版评估工具
│   │   ├── evaluation_utils_improved_nms.py      # 改进版 NMS
│   │   ├── visualization_utils.py# 3D 框绘制、BEV 与 RGB 图像合并
│   │   ├── train_utils.py        # 优化器、学习率调度器、checkpoint 保存
│   │   ├── torch_utils.py        # sigmoid、tensor 归约等辅助函数
│   │   ├── misc.py               # AverageMeter、ProgressMeter、time_synchronized
│   │   ├── logger.py             # 日志记录器
│   │   └── lr_scheduler.py       # OneCyclePolicy 等自定义学习率策略
│   ├── train.py                  # 主训练脚本（支持单 GPU / 多 GPU / 分布式训练）
│   ├── test.py                   # 原始推理脚本（可视化输出）
│   ├── testing.py                # 增强推理脚本（KITTI 格式输出 + 可选指标计算）
│   ├── testing_export.py         # 导出推理结果到 KITTI 格式
│   ├── testing_export_ultra_aggressive.py  # 超激进 NMS 推理（推荐用于验证集）
│   ├── export_to_onnx.py         # PyTorch → ONNX 模型导出
│   ├── run_onnx_inference.py     # ONNX Runtime 推理（纯 NumPy 后处理）
│   ├── quantize_model_163.py     # PyTorch 官方量化（动态/静态）
│   ├── quantize_config.py        # 量化专用配置（独立于训练配置）
│   ├── batch_visualize_3d_boxes_final.py   # 批量可视化（白底 + 渐变点云）
│   ├── create_video_from_images.py         # 从可视化图片生成视频
│   ├── kitti_evaluation_163.py # KITTI 标准 mAP 评估
│   ├── simple_onnx_evaluation.py # ONNX 结果统计
│   ├── analyze_163epoch_performance.py     # 163 轮模型性能分析
│   └── ...（其他实验/调试脚本）
├── DRadDataset/                  # 8D 毫米波雷达数据集
│   ├── ImageSets/                # 数据划分索引（train.txt / val.txt / test.txt）
│   ├── training/                 # 训练数据
│   │   ├── image_2/              # 相机图像（可选）
│   │   ├── velodyne/             # 8D 雷达点云（.bin 文件，float32）
│   │   ├── calib/                # 标定参数
│   │   └── label_2/              # KITTI 格式 3D 标注
│   └── testing/                  # 测试数据（结构同 training）
├── checkpoints/                  # 训练保存的模型权重
│   └── sfa3d_8d_full_300epochs/
│       └── Model_sfa3d_8d_full_300epochs_epoch_163.pth   # 推荐使用的 163 轮模型
├── logs/                         # 训练日志（每个实验一个子目录）
│   └── <saved_fn>/
│       ├── log_train.txt
│       └── tensorboard/          # TensorBoard 事件文件
├── results/                      # 推理输出的 KITTI 格式检测结果
├── onnx_models/                  # ONNX 导出模型
├── demo/                         # 演示图片和视频
└── docs/                         # 中文技术文档（方案、报告、指南等）
```

---

## 4. 关键配置参数

### 4.1 BEV 与检测范围（`sfa/config/kitti_config.py`）

```python
boundary = {
    "minX": 0,   "maxX": 50,    # 前方 0~50m
    "minY": -25, "maxY": 25,    # 左右 ±25m
    "minZ": -2.73, "maxZ": 1.27 # 高度范围
}
BEV_WIDTH = 608   # 对应 Y 轴
BEV_HEIGHT = 608  # 对应 X 轴
DISCRETIZATION = 50 / 608  # ≈ 0.0822 m/像素
```

### 4.2 模型输出头（`configs.heads`）

```python
heads = {
    'hm_cen': 3,      # 类别热力图（3 类）
    'cen_offset': 2,  # 中心点亚像素偏移
    'direction': 2,   # 方向角（sin, cos）
    'z_coor': 1,      # Z 坐标
    'dim': 3,         # 尺寸（h, w, l）
}
```

输入 BEV 图像为 3 通道：`[intensity, height, density]`，尺寸 `608×608`。下采样率 `down_ratio = 4`，热力图尺寸 `152×152`。

### 4.3 类别映射

```python
CLASS_NAME_TO_ID = {
    'Car': 0, 'Cyclist': 1, 'Truck': 2,
    'Van': -99, 'Pedestrian': -99,  # 忽略
    'DontCare': -1                   # 忽略但生成热力图
}
```

---

## 5. 构建与运行命令

> **注意**：所有命令应在项目根目录（`SFA4D/`）下执行。进入 `sfa/` 子目录执行脚本时，脚本内部会通过 `sys.path` 自动修正模块路径。

### 5.1 环境准备

```bash
conda create -n sfa3d python=3.8
conda activate sfa3d
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 5.2 训练

#### 单 GPU 训练（快速测试）
```bash
python sfa/train.py \
    --num_epochs 3 \
    --saved_fn sfa4d_test \
    --batch_size 4 \
    --dataset-dir ./DRadDataset \
    --root-dir ./ \
    --gpu_idx 0
```

#### 完整训练（300 epoch）
```bash
python sfa/train.py \
    --num_epochs 300 \
    --saved_fn sfa3d_8d_full_300epochs \
    --batch_size 16 \
    --dataset-dir ./DRadDataset \
    --root-dir ./ \
    --gpu_idx 0 \
    --checkpoint_freq 1 \
    --print_freq 50
```

#### 多 GPU 分布式训练
```bash
python sfa/train.py \
    --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    --batch_size 64 --num_workers 8 \
    --dataset-dir ./DRadDataset
```

#### 恢复训练
```bash
python sfa/train.py \
    --resume_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_100.pth \
    ...（其他参数同完整训练）
```

**训练输出**：
- 模型权重：`checkpoints/<saved_fn>/Model_<saved_fn>_epoch_N.pth`
- 优化器状态：`checkpoints/<saved_fn>/Utils_<saved_fn>_epoch_N.pth`
- 日志：`logs/<saved_fn>/log_train.txt`
- TensorBoard：`logs/<saved_fn>/tensorboard/`

### 5.3 推理（KITTI 格式输出）

#### 超激进 NMS 推理（推荐，用于验证/测试）
```bash
python sfa/testing_export_ultra_aggressive.py \
    --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --dataset-dir ./DRadDataset \
    --saved_fn sfa4d_163_ultra_aggressive \
    --peak_thresh 0.25 \
    --nms_thresh 0.2 \
    --gpu_idx 0
```

#### 标准推理（含可视化）
```bash
python sfa/testing.py \
    --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --dataset-dir ./DRadDataset \
    --saved_fn sfa4d_163 \
    --gpu_idx 0
```

**推理输出**：
- KITTI 格式检测结果：`results/<saved_fn>/<timestamp>/kitti_predictions/000000.txt`
- 可视化图像（如启用 `--save_test_output`）：`results/<saved_fn>/<timestamp>/viz/000000.jpg`

### 5.4 可视化

```bash
python sfa/batch_visualize_3d_boxes_final.py \
    --dataset-dir ./DRadDataset \
    --result-dir ./results/sfa4d_all \
    --output-dir ./visualizations_final_all
```

### 5.5 ONNX 导出与推理

```bash
# 导出
python sfa/export_to_onnx.py \
    --model ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --output ./onnx_models/sfa3d_163_fp32.onnx

# 推理
python sfa/run_onnx_inference.py \
    --onnx_model ./onnx_models/sfa3d_163_fp32.onnx \
    --dataset-dir ./DRadDataset \
    --imagesets-dir ./DRadDataset/ImageSets
```

### 5.6 模型量化

```bash
python sfa/quantize_model_163.py \
    --model ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --method dynamic \
    --output ./quantized_models/
```

### 5.7 TensorBoard 监控

```bash
tensorboard --logdir logs/<saved_fn>/tensorboard/
# 访问 http://localhost:6006
```

---

## 6. 代码组织与模块划分

### 6.1 数据流

```
原始 8D 点云 (.bin)
    ↓
lidar_mapping.py: read_lidar_file_with_fallback()
    → 自动识别 8D/5D/4D，8D 时取 [0,1,2,4] 并将 P 映射为 intensity
    ↓
kitti_bev_utils.py: makeBEVMap()
    → 生成 3 通道 BEV 图像 [intensity, height, density], 608×608
    ↓
Dataset / DataLoader
    ↓
模型输入: (B, 3, 608, 608)
    ↓
fpn_resnet.py: PoseResNet + KFPN
    → 输出 5 个 head 的特征图 (B, C, 152, 152)
    ↓
losses.py: Compute_Loss
    → focal_loss + l1_loss + balanced_l1_loss 加权
    ↓
evaluation_utils.py: decode + post_processing
    → 热力图 NMS → Top-K → 坐标还原 → 跨类别 NMS
    ↓
KITTI 格式检测结果 / 可视化图像
```

### 6.2 核心模块职责

| 模块 | 职责 |
|------|------|
| `data_process/lidar_mapping.py` | 多维度点云读取与 8D→4D 映射，零值处理（Scheme B） |
| `data_process/kitti_bev_utils.py` | BEV 投影图生成（高度图、强度图、密度图） |
| `data_process/transformation.py` | 数据增强：Random_Rotation、Random_Scaling、水平翻转、坐标系转换 |
| `models/fpn_resnet.py` | 主干网络：ResNet-18 + 3 层 FPN 上采样 + KFPN softmax 融合 + 多 head 输出 |
| `models/model_utils.py` | 模型工厂、参数统计、DataParallel / DistributedDataParallel 包装 |
| `losses/losses.py` | 多任务损失：FocalLoss（heatmap）+ L1Loss（offset/direction）+ BalancedL1Loss（z/dim） |
| `utils/evaluation_utils.py` | 解码（decode）、NMS（_nms）、后处理（post_processing）、坐标转换；含 NumPy 版本供 ONNX 使用 |
| `utils/train_utils.py` | Adam/SGD 优化器、cosin/multi_step/one_cycle 学习率调度、checkpoint 保存 |

---

## 7. 测试与评估策略

### 7.1 训练验证
- 训练过程中每 `checkpoint_freq` 个 epoch 自动在验证集上计算 `val_loss`。
- 最佳模型按 `val_loss` 最低保存为 `Model_<saved_fn>_best.pth`。

### 7.2 推理评估
- **KITTI 格式输出**：所有推理脚本默认将结果保存为 KITTI 标准格式 `.txt` 文件，可直接用于官方评估工具。
- **指标计算**：`testing.py` 支持 `--calc-metrics` 参数，在存在 ground-truth 时计算 mAP、Precision、Recall、IoU。
- **ONNX 评估**：`run_onnx_inference.py` 输出检测统计（每类数量、平均置信度、FPS）。
- **163 轮模型专项分析**：`analyze_163epoch_performance.py` 对 163 轮模型进行深度性能分析。

### 7.3 可视化验证
- `batch_visualize_3d_boxes_final.py`：将检测结果与 BEV 点云叠加绘制，用于人工检查检测质量。
- `create_video_from_images.py`：将可视化图片合成为视频。

---

## 8. 代码风格与开发约定

### 8.1 语言与注释
- 代码注释和文档以 **中文** 为主，尤其是创新点描述（8D 映射、KFPN、跨类别 NMS 等）。
- 原始 SFA3D 遗留代码保留英文注释。

### 8.2 路径处理
- 脚本内部大量使用 `os.path.abspath()` 和相对路径拼接，确保在不同工作目录下运行时的兼容性。
- `dataset_dir` 若未指定，默认指向 `<root_dir>/dataset/kitti`；实际使用时应显式指定 `--dataset-dir ./DRadDataset`。

### 8.3 Checkpoint 保存
- 由于 Windows 环境下 PyTorch zip 序列化可能出现问题，项目统一使用 `_use_new_zipfile_serialization=False`：
  ```python
  torch.save(state_dict, path, _use_new_zipfile_serialization=False)
  ```
- 模型和优化器状态分开保存：`Model_*.pth` 和 `Utils_*.pth`。

### 8.4 数据维度约定
- 8D 雷达点云维度顺序：`[x, y, z, Doppler, P(SNR), Range, Azimuth, Elevation]`。
- 训练时实际读取索引 `[0, 1, 2, 4]`，将 `P` 映射为 `intensity`。
- BEV 图像通道顺序：`[intensity, height, density]`（对应 `RGB_Map[0,1,2]`）。

### 8.5 分布式训练
- 支持 `torch.nn.DataParallel`（单节点多 GPU）和 `torch.nn.parallel.DistributedDataParallel`（多节点）。
- 启动分布式训练时需设置 `--multiprocessing-distributed`、`--world-size`、`--rank`、`--dist-url`。

---

## 9. 安全与部署注意事项

### 9.1 模型安全
- 预训练权重通过 `torch.load(..., map_location='cpu')` 加载，避免 GPU 设备不匹配导致的错误。
- 推理脚本默认使用 `torch.no_grad()`，防止梯度泄露和内存浪费。

### 9.2 部署建议
- **GPU 服务器**：直接使用 PyTorch FP32 模型，推理速度约 110 FPS（RTX 4060 Ti）。
- **跨平台/边缘设备**：导出 ONNX 模型，使用 ONNX Runtime 推理（约 5.5 FPS，CPU）。
- **极致压缩**：使用 `quantize_model_163.py` 进行动态量化，模型体积从 ~49MB 压缩至 ~13MB（压缩率 73%）。

### 9.3 常见陷阱
- **路径错误**：确保 `--dataset-dir` 指向包含 `ImageSets/` 和 `training/`（或 `testing/`）的目录。
- **CUDA OOM**：减小 `--batch_size`（训练可降至 2，推理可保持 1）。
- **ImageSets 缺失**：若 `ImageSets/val.txt` 不存在，脚本会尝试直接从 `velodyne/` 或 `label_2/` 目录枚举文件，但建议始终提供正确的 `ImageSets`。
- **8D 数据识别失败**：`lidar_mapping.py` 通过文件大小对 8/5/4 取模判断维度，若文件损坏可能导致错误映射。

---

## 10. 快速参考：最常用命令

```bash
# 训练（单 GPU，快速测试）
python sfa/train.py --num_epochs 3 --batch_size 4 --dataset-dir ./DRadDataset --gpu_idx 0

# 训练（完整 300 epoch）
python sfa/train.py --num_epochs 300 --batch_size 16 --saved_fn sfa3d_8d_full --dataset-dir ./DRadDataset --gpu_idx 0

# 推理（推荐 163 轮模型 + 超激进 NMS）
python sfa/testing_export_ultra_aggressive.py \
    --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --dataset-dir ./DRadDataset --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0

# 可视化
python sfa/batch_visualize_3d_boxes_final.py --dataset-dir ./DRadDataset --result-dir ./results/sfa4d_all --output-dir ./visualizations

# ONNX 导出
python sfa/export_to_onnx.py --model ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --output ./onnx_models/sfa3d_163.onnx

# TensorBoard
python -m tensorboard.main --logdir logs/sfa3d_8d_full_300epochs/tensorboard/
```

---

*本文档基于 SFA4D 项目实际代码和文档整理，最后更新于 2026-07-15。*
