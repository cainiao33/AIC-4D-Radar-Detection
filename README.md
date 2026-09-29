# SFA4D — 4D 毫米波雷达点云 3D 目标检测（基于 SFA3D）

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/🇨🇳_语言-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/🇬🇧_Language-English-blue?style=for-the-badge" alt="English"></a>
</p>

> **一句话**：基于 SFA3D 的 4D 毫米波雷达点云 3D 检测——原创 8D→4D SNR 映射与跨类别 NMS，网络结构沿用 SFA3D 原版
> **成绩**：AIC 2025 全球校园人工智能算法精英大赛 **全国二等奖** · 75 mAP@0.5 · 110.73 FPS（RTX 4060 Ti）· INT8 量化后 12.3 MB
> **怎么跑**：163 轮权重已随仓库分发（`checkpoints/` + `onnx_models/`，也可从 [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0) 下载）。无需自备数据集即可跑通 3 样本演示（`sample_data/` 随仓库分发）：
>
> ```bash
> docker build -t sfa4d . && docker run --rm --gpus all \
>     -v $PWD/sample_data:/data/DRadDataset -v $PWD/results:/app/results sfa4d
> ```
>
> 完整数据集需按赛题官方渠道自备；命令文档见 [docs/操作指令/](docs/操作指令/)，或 [Docker 复现](#-docker-复现)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AIC Competition](https://img.shields.io/badge/AIC-2025-green.svg)](https://www.aicomp.cn/)

<p align="center"><img src="demo/preview.gif" alt="SFA4D BEV 检测效果演示" width="800"></p>

---

## 📖 项目简介

本项目是 **AIC 全球校园人工智能算法精英大赛（算法挑战赛）全国二等奖** 的完整开源方案，专注于 **8D 毫米波雷达点云数据** 的 3D 目标检测任务（雷达单模态，无相机数据进网络）。

技术路线：**沿用 SFA3D 原版网络**——骨干（FPN-ResNet + KFPN）、损失函数、BEV 生成均未改动——在「数据进」与「结果出」两端做适配与优化，使 LiDAR 检测框架直接吃 4D 毫米波雷达数据。逐文件差异见 [docs/SFA3D与SFA4D源码比对报告.md](docs/SFA3D与SFA4D源码比对报告.md)。

### 本项目的改动（相对 SFA3D 原版）
- **🎯 8D→4D SNR 映射（原创核心）**：新文件 `sfa/data_process/lidar_mapping.py`——8D 点云 `[x, y, z, Doppler, P, Range, Azimuth, Elevation]` 取 `[x, y, z, P]`，第 5 维信噪比 P 分段映射为 intensity（零值→0.1，非零按标定界 `[-1.36, 6.43]` 归一化到 `[0.2, 1.0]`），是 SFA3D 的 BEV intensity 通道能吃雷达数据的承重墙
- **🔄 跨类别 NMS（原创）**：`apply_inter_class_nms()` 抑制不同类别间的重复检测；诚实说明：其中的 IoU 为中心距离启发式 + 手工分段，**不是**旋转矩形几何交叠
- **⚙️ 竞赛后处理调参**：超激进参数（peak 0.25 / 跨类 NMS 阈 0.2 / K=50）——75 mAP 的产出链路 = 原版网络 + 本套调参
- **📦 部署工程**：ONNX 导出 + 纯 NumPy 后处理（CPU 可跑）+ INT8 动态量化（48.57→12.27 MB）
- **🔁 类别重映射**：KITTI 类别 → Car/Cyclist/Truck（Van/Pedestrian 丢弃），配置级必要适配

> 说明：KFPN 特征融合、无锚点检测头、端到端 7-DOF 预测均为 **SFA3D 论文原有设计**（本项目对应文件与原版逐字节相同），不计入本项目创新。

---

## 🏆 竞赛成果

| 指标 | 成绩 |
|------|------|
| **竞赛奖项** | AIC 全球校园人工智能算法精英大赛 · 算法挑战赛 **全国二等奖** |
| **推理速度** | **110.73 FPS**（RTX 4060 Ti，PyTorch FP32） |
| **检测精度** | **75% mAP@0.5** |
| **模型体积** | 48.65 MB（`.pth`）/ 48.57 MB（FP32 ONNX）；INT8 动态量化后 **12.27 MB**（-74.7%，`quantized_models/sfa3d_163_int8.onnx`，实测 3 样本检测与 FP32 一致） |
| **跨平台推理** | 5.48 FPS（ONNX Runtime，CPU） |
| **验证样本** | 成功处理 1034 个验证样本，检测率 20.7% |

---

## 🎥 演示效果

<p align="center"><img src="demo/000000_final.png" alt="SFA4D BEV 3D 检测效果" width="100%"></p>

> 上图：BEV 视角下的 3D 目标检测效果——白色背景配合渐变点云，检测框以不同颜色区分类别（Car / Cyclist / Truck）。

**GIF 自动播放预览（2 分钟完整演示）：**

<img src="https://github.com/cainiao33/AIC-4D-Radar-Detection/raw/main/demo/preview_120s.gif" width="800" alt="SFA4D 检测效果完整演示">

> 2 分钟完整检测演示（3fps，400px 宽）。如加载较慢，可观看下方视频或访问 [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0)。

**点击播放完整视频（更高画质）：**

<video src="https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/download/v1.0/visualization_12fps.mp4" controls width="100%" poster="demo/000000_final.png"></video>

> 完整演示视频（含 OpenCV 渲染）请访问 [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0)。

---

## 📦 复现说明

| 内容 | 状态 | 位置 |
|------|------|------|
| 163 轮权重 `.pth`（48.65 MB） | ✅ 已随仓库分发 | `checkpoints/sfa3d_8d_full_300epochs/` |
| FP32 ONNX（48.57 MB） | ✅ 已随仓库分发 | `onnx_models/sfa3d_163_fp32.onnx` |
| 示例数据（3 个样本） | ✅ 已含 | `sample_data/`（training + testing 分割；可跑 Docker 演示） |
| **DRadDataset 数据集** | ❌ 不随仓库分发 | 按赛题官方渠道自备（KITTI 格式：`ImageSets/ training/ testing/`） |
| INT8 动态量化 ONNX（12.27 MB） | ✅ 已随仓库分发 | `quantized_models/sfa3d_163_int8.onnx`（实测与 FP32 检测一致） |

环境以 [requirements.txt](requirements.txt) 与 [docs/操作指令/](docs/操作指令/)（训练/推理/可视化命令）为准（Python 3.8 + PyTorch 2.0.0+cu118）；环境安装见下方「快速开始」，或直接用 Docker。

---

## 📂 项目结构

```
SFA4D/
├── sfa/                          # 核心代码目录
│   ├── config/                   # 配置文件（BEV 参数、训练参数）
│   ├── data_process/             # 数据处理（Dataset、BEV 生成、8D→4D 映射）
│   ├── models/                   # 模型定义（FPN-ResNet + KFPN）
│   ├── losses/                   # 损失函数（FocalLoss + L1Loss + BalancedL1Loss）
│   ├── utils/                    # 工具函数（NMS、后处理、可视化、训练辅助）
│   ├── legacy/                   # 实验脚本归档（含 7 个伪结果/损坏脚本，已逐个标注）
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 推理/评测入口（超激进 NMS 参数为默认）
│   ├── export_to_onnx.py         # ONNX FP32 导出
│   ├── quantize_onnx_163.py      # INT8 动态量化（含 FP32/INT8 对照验证）
│   └── run_onnx_inference.py     # ONNX Runtime CPU 推理
├── tests/                        # 单元测试（pytest：8D 映射 / BEV / NMS / 伪 IoU）
├── .github/workflows/ci.yml      # CI（语法门 + 单测）
├── checkpoints/                  # ✅ 已分发权重
│   └── sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth
├── onnx_models/                  # ✅ FP32 ONNX（sfa3d_163_fp32.onnx）
├── quantized_models/             # ✅ INT8 动态量化 ONNX（sfa3d_163_int8.onnx，12.27 MB）
├── sample_data/                  # 示例数据（3 个样本，Docker 演示可直接挂载）
│   ├── training/                 # velodyne/ label_2/ calib/（训练冒烟）
│   └── testing/                  # velodyne/ image_2/ calib/（推理演示）
├── demo/                         # 演示 gif / 图片
├── docs/                         # 技术文档 + 操作指令（中文）
├── Dockerfile / .dockerignore   # Docker 复现（仅封装最佳 .pth）
├── requirements.txt / pyproject.toml   # 环境依赖
├── README.md                     # 本文件（中文，默认）
├── README_en.md                  # 英文版
└── LICENSE                       # MIT 开源协议
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+（实际验证 3.8.20）
- PyTorch 2.0.0+cu118
- CUDA 11.8+（GPU 训练/推理）
- Windows 10/11 或 Linux Ubuntu 18.04+

### 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n sfa3d python=3.8
conda activate sfa3d

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 快速体验 ①（Docker：3 样本推理演示，无需自备数据集）

```bash
docker build -t sfa4d .
docker run --rm --gpus all \
    -v $PWD/sample_data:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d
# → results/sfa4d_163_eval/{000040,000050,000055}.txt（KITTI 格式检测框）
```

> 无 GPU 环境改用：`docker run --rm -v $PWD/sample_data:/data/DRadDataset -v $PWD/results:/app/results sfa4d python3 sfa/eval.py --dataset-dir /data/DRadDataset --no_cuda`（默认权重路径已内置，无需传 `--pretrained_path`）

### 快速体验 ②（本机环境：训练冒烟）

```bash
# 单 GPU 快速训练（3 epoch，用于验证环境）
python sfa/train.py \
    --num_epochs 3 \
    --saved_fn sfa4d_test \
    --batch_size 4 \
    --dataset-dir ./sample_data \
    --root-dir ./ \
    --gpu_idx 0
```

> 注：`sample_data/` 不含 `ImageSets/`，脚本将直接从 `velodyne/`/`label_2/` 枚举文件（见 AGENTS.md「常见陷阱」）；仅供数据管线冒烟，不代表训练效果。

### 完整训练（300 epoch）

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

### 推理（使用已分发模型）

```bash
# 正式推理入口（超激进 NMS 参数即为默认值；原竞赛脚本归档于 sfa/legacy/）
python sfa/eval.py \
    --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --dataset-dir ./DRadDataset \
    --peak_thresh 0.25 \
    --nms_thresh 0.2 \
    --gpu_idx 0 \
    --output-dir ./results/sfa4d_163_eval
```

> 单元测试：`pip install pytest` 后在仓库根目录执行 `pytest tests`（详见 [.github/workflows/ci.yml](.github/workflows/ci.yml)）。

---

## 🐳 Docker 复现

镜像**仅封装最佳 `.pth` 模型**（epoch 163）+ 训练/推理全流程，复刻实际环境（PyTorch 2.0.0+cu118 + CUDA 11.8）；ONNX / INT8 产物不入镜像，直接使用仓库内文件。

**实测**（RTX 4060 Ti / WSL2）：镜像约 4.9 GB；挂载完整数据集跑默认推理命令，20 样本 2.53 s（7.91 FPS，含 9p 挂载 I/O），单样本 GPU 推理 ~10-17 ms，输出标准 KITTI 预测文件。

> ⚠️ 挂载的数据集须含 `testing/{velodyne, image_2, calib}`（默认 `--test-subdir testing`）：样本发现是枚举 `velodyne/*.bin`；但 test 模式仍会读取 `image_2` 的 PNG（图片不进网络，本项目为雷达单模态，缺失会导致 cv2 报错），calib 路径与输出文件名由 image_2 路径推导。国内构建可自行在 Dockerfile 中保留清华源配置（apt/pip 已默认换源）。

```bash
# 构建镜像
docker build -t sfa4d .

# ⓪ 三样本演示（无需自备数据集；sample_data/ 随仓库分发）
docker run --rm --gpus all \
    -v $PWD/sample_data:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d

# ① 默认命令：对挂载的完整数据集跑超激进 NMS 推理（结果目录建议挂载出来）
docker run --gpus all \
    -v /path/to/DRadDataset:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d

# ② 进入容器执行任意命令（训练 / 评估 / 可视化 / 单元测试）
docker run --gpus all \
    -v /path/to/DRadDataset:/data/DRadDataset \
    -it sfa4d bash
```

---

## 📊 数据集

### 数据格式

本项目使用 **DRadDataset** 数据集，包含 8D 毫米波雷达点云数据：

| 维度 | 含义 | 说明 |
|------|------|------|
| 0 | x | 前向距离（米） |
| 1 | y | 横向距离（米） |
| 2 | z | 高度（米） |
| 3 | Doppler | 多普勒速度 |
| 4 | P (SNR) | **信噪比强度** → 映射为 intensity |
| 5 | Range | 径向距离 |
| 6 | Azimuth | 方位角 |
| 7 | Elevation | 俯仰角 |

### 数据集获取

完整 DRadDataset 数据集（训练集 5168 样本 + 测试集 1384 样本）**不随仓库与 Release 分发**，请通过赛题官方渠道获取，并按 KITTI 格式放置：`DRadDataset/{ImageSets, training, testing}/`。仓库根目录的 `sample_data/`（3 个样本，含 training/testing 分割）用于核对数据格式与 Docker 演示。

---

## 🧠 预训练模型

| 模型 | 大小 | 说明 | 获取 |
|------|------|------|------|
| Epoch 163（PyTorch） | 48.65 MB | 推荐使用的最佳模型 | ✅ 仓库内 `checkpoints/sfa3d_8d_full_300epochs/`，或 [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0) |
| ONNX FP32 | 48.57 MB | 跨平台部署（CPU 可跑） | ✅ 仓库内 `onnx_models/sfa3d_163_fp32.onnx`，或 [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0) |
| INT8 动态量化 ONNX | 12.27 MB | 边缘设备部署 | ✅ 仓库内 `quantized_models/sfa3d_163_int8.onnx` |

<details>
<summary><b>权重校验和（sha256，48.57→48.65 MB 的差异系 .pth 与 ONNX 序列化不同）</b></summary>

```text
checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth
  51,020,173 B  b042158ee213e4acd1bd4fde51eba374194d799ec537693c52e63bf29e98c003
onnx_models/sfa3d_163_fp32.onnx
  50,928,757 B  9cd6f3e9b3e5eec34eeba84151717da5a7314ead72e78264259c27878bea46dd
quantized_models/sfa3d_163_int8.onnx
  12,866,736 B  ddffb1708bbb8cab7c3310f5ff2f1e068e56e2dacaf3a4279bd539c72faf9ba3
```

</details>

### 模型导出

```bash
# PyTorch → ONNX（仓库已附带导出结果，无需重复执行）
python sfa/export_to_onnx.py \
    --model ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --output ./onnx_models/sfa3d_163_fp32.onnx

# ONNX INT8 动态量化：48.57 MB → 12.27 MB（仓库已附带产物，无需重复执行）
# 脚本量化后自动用 sample_data 真实样本对比 FP32/INT8 检测结果
python sfa/quantize_onnx_163.py \
    --onnx_model ./onnx_models/sfa3d_163_fp32.onnx \
    --output ./quantized_models/sfa3d_163_int8.onnx

# 注：sfa/quantize_model_163.py 为 PyTorch 量化路线，仅作对照——
# torch 动态量化不支持 Conv2d，本模型为纯卷积网络，实测压缩率 0%
```

---

## 📈 性能指标

### 各类别检测性能

| 类别 | mAP@0.5 | 说明 |
|------|---------|------|
| Car | **83%** | 车辆检测（大目标，特征明显） |
| Cyclist | **65%** | 骑行者检测（小目标，最具挑战） |
| Truck | **70%** | 卡车检测（中等目标） |
| **Average** | **73%** | **整体平均** |

### 推理速度对比

| 平台 | 框架 | 速度 | 说明 |
|------|------|------|------|
| RTX 4060 Ti | PyTorch FP32 | **110.73 FPS** | GPU 推理 |
| CPU | ONNX Runtime FP32 | 5.48 FPS | 跨平台部署（复测 5.40） |
| CPU | ONNX Runtime INT8 | 3.23 FPS | 收益在体积（-74.7%）；x86 动态量化不提速，见 [docs/操作指令/启动训练指令.md](docs/操作指令/启动训练指令.md) 量化说明 |

---

## 📚 技术文档

| 文档 | 说明 |
|------|------|
| [docs/技术方案AAA.md](docs/技术方案AAA.md) | 完整技术实现方案 ⭐ |
| [docs/SFA3D与SFA4D源码比对报告.md](docs/SFA3D与SFA4D源码比对报告.md) | 与 SFA3D 原版的逐文件差异（诚实定位本项目改动） ⭐ |
| [docs/技术报告.md](docs/技术报告.md) | 深度技术分析报告 |
| [docs/项目结构说明.md](docs/项目结构说明.md) | 项目架构和文件说明 |
| [docs/操作指令/](docs/操作指令/) | 训练 / 推理 / 可视化 / 量化命令速查 |
| [docs/环境依赖清单.md](docs/环境依赖清单.md) | 详细环境配置要求 |
| [docs/点云维度修改说明.md](docs/点云维度修改说明.md) | 8D→4D 映射实现细节 |
| [docs/SFA4D 代码架构.md](docs/SFA4D%20代码架构.md) | 代码架构详解 |

---

## 🔧 核心模块说明

### 数据流

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

### 关键配置参数

```python
# BEV 边界（sfa/config/kitti_config.py）
boundary = {
    "minX": 0,   "maxX": 50,     # 前方 0~50m
    "minY": -25, "maxY": 25,     # 左右 ±25m
    "minZ": -2.73, "maxZ": 1.27  # 高度范围
}
BEV_WIDTH = 608
BEV_HEIGHT = 608
DISCRETIZATION = 50 / 608  # ≈ 0.0822 m/像素

# 模型输出头
heads = {
    'hm_cen': 3,      # 类别热力图（Car, Cyclist, Truck）
    'cen_offset': 2,  # 中心点亚像素偏移
    'direction': 2,   # 方向角（sin, cos）
    'z_coor': 1,      # Z 坐标
    'dim': 3,         # 尺寸（h, w, l）
}
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

- 感谢 **AIC 全球校园人工智能算法精英大赛** 提供的竞赛平台和数据集
- 本项目的网络骨干、损失函数、BEV 生成等核心代码直接沿用 [SFA3D](https://github.com/maudzung/SFA3D)（作者 Nguyen Mau Dung / maudzung，MIT 协议），特此致谢并注明出处；本项目在其基础上新增数据域适配与后处理，详见比对报告
- 感谢所有团队成员的辛勤付出
- 仓库工程化整理（Docker 封装、单元测试、脚本归档、文档修订）由 Claude Code 协助完成，协作记录见各提交信息

---

## 📧 联系方式

- 竞赛官网：[AIC 全球校园人工智能算法精英大赛](https://www.aicomp.cn/)
- 邮箱：**webcainiao@gmail.com**

---

> **⭐ 如果本项目对你有帮助，请 Star 支持！**
