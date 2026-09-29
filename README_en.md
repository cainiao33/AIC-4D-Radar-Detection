# SFA4D — 4D mmWave Radar Point-Cloud 3D Object Detection (based on SFA3D)

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Switch_Language-简体中文-red?style=for-the-badge" alt="简体中文"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/Switch_Language-English-blue?style=for-the-badge" alt="English"></a>
</p>

> **One-liner**: 3D detection on 4D mmWave radar point clouds built on SFA3D — original 8D→4D SNR mapping and cross-class NMS; the network architecture is unchanged from SFA3D.
> **Results**: AIC 2025 Global Campus AI Algorithm Elite Competition — **National Second Prize** · 75 mAP@0.5 · 110.73 FPS (RTX 4060 Ti) · 12.3 MB after INT8 quantization.
> **How to run**: the 163-epoch weights ship with the repo (`checkpoints/` + `onnx_models/`, also downloadable from [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0)). The 3-sample demo runs with no extra data (`sample_data/` included):
>
> ```bash
> docker build -t sfa4d . && docker run --rm --gpus all \
>     -v $PWD/sample_data:/data/DRadDataset -v $PWD/results:/app/results sfa4d
> ```
>
> The full dataset must be obtained through official competition channels; see [docs/操作指令/](docs/操作指令/) for commands (Chinese), or [Docker reproduction](#-docker-reproduction) below.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AIC Competition](https://img.shields.io/badge/AIC-2025-green.svg)](https://www.aicomp.cn/)

<p align="center"><img src="demo/preview.gif" alt="SFA4D BEV detection demo" width="800"></p>

---

## 📖 Overview

This repository open-sources the complete solution that won the **National Second Prize** of the AIC Global Campus AI Algorithm Elite Competition (Algorithm Challenge Track), focused on **3D object detection from 8D mmWave radar point clouds** (radar-only; no camera images enter the network).

Technical route: **the SFA3D network is used unchanged** — backbone (FPN-ResNet + KFPN), losses, and BEV generation are all untouched — while adaptation and optimization happen at the two ends ("data in" and "results out"), letting a LiDAR detection framework consume 4D mmWave radar data directly. For a file-by-file diff against the original, see [docs/SFA3D与SFA4D源码比对报告.md](docs/SFA3D与SFA4D源码比对报告.md) (Chinese).

### Changes in this project (vs. original SFA3D)
- **🎯 8D→4D SNR mapping (original core)**: new file `sfa/data_process/lidar_mapping.py` — from the 8D point cloud `[x, y, z, Doppler, P, Range, Azimuth, Elevation]` take `[x, y, z, P]`, mapping the 5th channel (SNR) piecewise to intensity (zeros → 0.1; non-zeros normalized from the calibrated range `[-1.36, 6.43]` to `[0.2, 1.0]`). This is the load-bearing wall that lets SFA3D's BEV intensity channel consume radar data.
- **🔄 Cross-class NMS (original)**: `apply_inter_class_nms()` suppresses duplicate detections across classes. Honest note: its IoU is a center-distance heuristic with hand-tuned segments, **not** rotated-rectangle geometric overlap.
- **⚙️ Post-competition tuning**: ultra-aggressive parameters (peak 0.25 / cross-class NMS threshold 0.2 / K=50) — the pipeline that produced 75 mAP = original network + this tuning.
- **📦 Deployment engineering**: ONNX export + pure-NumPy post-processing (CPU-runnable) + INT8 dynamic quantization (48.57 → 12.27 MB).
- **🔁 Class remapping**: KITTI classes → Car/Cyclist/Truck (Van/Pedestrian dropped); a necessary config-level adaptation.

> Note: KFPN feature fusion, the anchor-free detection head, and end-to-end 7-DOF prediction are all **original SFA3D paper designs** (the corresponding files are byte-identical to the original repo) and are not claimed as contributions of this project.

---

## 🏆 Competition Results

| Metric | Result |
|------|------|
| **Award** | AIC Global Campus AI Algorithm Elite Competition · Algorithm Challenge — **National Second Prize** |
| **Inference speed** | **110.73 FPS** (RTX 4060 Ti, PyTorch FP32) |
| **Detection accuracy** | **75% mAP@0.5** |
| **Model size** | 48.65 MB (`.pth`) / 48.57 MB (FP32 ONNX); **12.27 MB** after INT8 dynamic quantization (−74.7%, `quantized_models/sfa3d_163_int8.onnx`; detections verified identical to FP32 on 3 samples) |
| **Cross-platform inference** | 5.48 FPS (ONNX Runtime, CPU) |
| **Validation samples** | 1034 validation samples processed successfully, 20.7% detection rate |

---

## 🎥 Demo

<p align="center"><img src="demo/000000_final.png" alt="SFA4D BEV 3D detection result" width="100%"></p>

> Above: 3D object detection in BEV view — white background with gradient point cloud; boxes are color-coded by class (Car / Cyclist / Truck).

**Auto-playing GIF preview (2-minute full demo):**

<img src="https://github.com/cainiao33/AIC-4D-Radar-Detection/raw/main/demo/preview_120s.gif" width="800" alt="SFA4D full detection demo">

> 2-minute full detection demo (3 fps, 400 px wide). If it loads slowly, watch the video below or visit [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0).

**Full video, higher quality (click to play):**

<video src="https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/download/v1.0/visualization_12fps.mp4" controls width="100%" poster="demo/000000_final.png"></video>

> For the full demo video (with OpenCV rendering), visit [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0).

---

## 📦 Reproduction Status

| Item | Status | Location |
|------|------|------|
| 163-epoch weights `.pth` (48.65 MB) | ✅ shipped in-repo | `checkpoints/sfa3d_8d_full_300epochs/` |
| FP32 ONNX (48.57 MB) | ✅ shipped in-repo | `onnx_models/sfa3d_163_fp32.onnx` |
| Sample data (3 samples) | ✅ included | `sample_data/` (training + testing splits; enough for the Docker demo) |
| **DRadDataset** | ❌ not distributed | obtain via official competition channels (KITTI layout: `ImageSets/ training/ testing/`) |
| INT8 dynamic-quantized ONNX (12.27 MB) | ✅ shipped in-repo | `quantized_models/sfa3d_163_int8.onnx` (detections verified identical to FP32) |

The environment follows [requirements.txt](requirements.txt) and [docs/操作指令/](docs/操作指令/) (Python 3.8 + PyTorch 2.0.0+cu118); see "Quick Start" below, or just use Docker.

---

## 📂 Project Structure

```
SFA4D/
├── sfa/                          # core code
│   ├── config/                   # configs (BEV parameters, training parameters)
│   ├── data_process/             # data processing (Dataset, BEV generation, 8D→4D mapping)
│   ├── models/                   # model definitions (FPN-ResNet + KFPN)
│   ├── losses/                   # losses (FocalLoss + L1Loss + BalancedL1Loss)
│   ├── utils/                    # utilities (NMS, post-processing, visualization, training helpers)
│   ├── legacy/                   # archived experiment scripts (7 fake-result/broken scripts, each annotated)
│   ├── train.py                  # training entry
│   ├── eval.py                   # inference/evaluation entry (ultra-aggressive NMS params by default)
│   ├── export_to_onnx.py         # ONNX FP32 export
│   ├── quantize_onnx_163.py      # INT8 dynamic quantization (with FP32/INT8 cross-validation)
│   └── run_onnx_inference.py     # ONNX Runtime CPU inference
├── tests/                        # unit tests (pytest: 8D mapping / BEV / NMS / pseudo-IoU)
├── .github/workflows/ci.yml      # CI (syntax gate + unit tests)
├── checkpoints/                  # ✅ distributed weights
│   └── sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth
├── onnx_models/                  # ✅ FP32 ONNX (sfa3d_163_fp32.onnx)
├── quantized_models/             # ✅ INT8 dynamic-quantized ONNX (sfa3d_163_int8.onnx, 12.27 MB)
├── sample_data/                  # sample data (3 samples; directly mountable for the Docker demo)
│   ├── training/                 # velodyne/ label_2/ calib/ (training smoke test)
│   └── testing/                  # velodyne/ image_2/ calib/ (inference demo)
├── demo/                         # demo gifs / images
├── docs/                         # technical docs + command references (Chinese)
├── Dockerfile / .dockerignore    # Docker reproduction (packages only the best .pth)
├── requirements.txt / pyproject.toml   # dependencies
├── README.md                     # Chinese version (中文版，默认)
├── README_en.md                  # this file (English)
└── LICENSE                       # MIT
```

---

## 🚀 Quick Start

### Requirements

- Python 3.8+ (verified on 3.8.20)
- PyTorch 2.0.0+cu118
- CUDA 11.8+ (GPU training/inference)
- Windows 10/11 or Linux Ubuntu 18.04+

### Install dependencies

```bash
# Create a virtual environment (recommended)
conda create -n sfa3d python=3.8
conda activate sfa3d

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install the remaining dependencies
pip install -r requirements.txt
```

### Quick try ① (Docker: 3-sample inference demo, no dataset needed)

```bash
docker build -t sfa4d .
docker run --rm --gpus all \
    -v $PWD/sample_data:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d
# → results/sfa4d_163_eval/{000040,000050,000055}.txt (KITTI-format detection boxes)
```

> No GPU? Use: `docker run --rm -v $PWD/sample_data:/data/DRadDataset -v $PWD/results:/app/results sfa4d python3 sfa/eval.py --dataset-dir /data/DRadDataset --no_cuda` (the default weight path is baked in; no `--pretrained_path` needed).

### Quick try ② (local environment: training smoke test)

```bash
# Quick single-GPU training (3 epochs, to validate the environment)
python sfa/train.py \
    --num_epochs 3 \
    --saved_fn sfa4d_test \
    --batch_size 4 \
    --dataset-dir ./sample_data \
    --root-dir ./ \
    --gpu_idx 0
```

> Note: `sample_data/` has no `ImageSets/`, so the script enumerates files directly from `velodyne/`/`label_2/` (see AGENTS.md "Common pitfalls"); this only smoke-tests the data pipeline and does not represent training quality.

### Full training (300 epochs)

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

### Inference (with the distributed model)

```bash
# Official inference entry point (ultra-aggressive NMS parameters are the defaults;
# original competition scripts are archived under sfa/legacy/)
python sfa/eval.py \
    --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --dataset-dir ./DRadDataset \
    --peak_thresh 0.25 \
    --nms_thresh 0.2 \
    --gpu_idx 0 \
    --output-dir ./results/sfa4d_163_eval
```

> Unit tests: after `pip install pytest`, run `pytest tests` at the repo root (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## 🐳 Docker Reproduction

The image packages **only the best `.pth` model** (epoch 163) + the full train/infer pipeline, replicating the actual environment (PyTorch 2.0.0+cu118 + CUDA 11.8); ONNX / INT8 artifacts are not baked in — use the in-repo files directly.

**Measured** (RTX 4060 Ti / WSL2): image ≈ 4.9 GB; with the full dataset mounted, the default inference command processes 20 samples in 2.53 s (7.91 FPS, including 9p mount I/O); single-sample GPU inference ~10–17 ms; outputs standard KITTI prediction files.

> ⚠️ The mounted dataset must contain `testing/{velodyne, image_2, calib}` (default `--test-subdir testing`): sample discovery enumerates `velodyne/*.bin`, but test mode still reads the PNGs from `image_2` (images never enter the network — this project is radar-only — but a missing PNG crashes cv2), and calib paths / output filenames are derived from the `image_2` path. For builds in China, keep the Tsinghua mirror configuration in the Dockerfile (apt/pip mirrors are the default).

```bash
# Build the image
docker build -t sfa4d .

# ⓪ Three-sample demo (no dataset needed; sample_data/ ships with the repo)
docker run --rm --gpus all \
    -v $PWD/sample_data:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d

# ① Default command: ultra-aggressive NMS inference on the mounted full dataset
#    (mount the results directory out)
docker run --gpus all \
    -v /path/to/DRadDataset:/data/DRadDataset \
    -v $PWD/results:/app/results \
    sfa4d

# ② Start an interactive container (training / evaluation / visualization / unit tests)
docker run --gpus all \
    -v /path/to/DRadDataset:/data/DRadDataset \
    -it sfa4d bash
```

---

## 📊 Dataset

### Data format

This project uses the **DRadDataset**, containing 8D mmWave radar point clouds:

| Dim | Meaning | Notes |
|------|------|------|
| 0 | x | forward distance (m) |
| 1 | y | lateral distance (m) |
| 2 | z | height (m) |
| 3 | Doppler | Doppler velocity |
| 4 | P (SNR) | **signal-to-noise strength** → mapped to intensity |
| 5 | Range | radial distance |
| 6 | Azimuth | azimuth angle |
| 7 | Elevation | elevation angle |

### Obtaining the dataset

The full DRadDataset (5168 training + 1384 testing samples) is **not distributed** with the repo or the Release; obtain it through the official competition channels and lay it out in KITTI format: `DRadDataset/{ImageSets, training, testing}/`. The in-repo `sample_data/` (3 samples, with training/testing splits) is for format verification and the Docker demo.

---

## 🧠 Pretrained Models

| Model | Size | Notes | Availability |
|------|------|------|------|
| Epoch 163 (PyTorch) | 48.65 MB | recommended best model | ✅ in-repo `checkpoints/sfa3d_8d_full_300epochs/`, or [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0) |
| ONNX FP32 | 48.57 MB | cross-platform deployment (CPU-runnable) | ✅ in-repo `onnx_models/sfa3d_163_fp32.onnx`, or [Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Detection/releases/tag/v1.0) |
| INT8 dynamic-quantized ONNX | 12.27 MB | edge deployment | ✅ in-repo `quantized_models/sfa3d_163_int8.onnx` |

<details>
<summary><b>Weight checksums (sha256; the 48.57→48.65 MB difference is .pth vs ONNX serialization)</b></summary>

```text
checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth
  51,020,173 B  b042158ee213e4acd1bd4fde51eba374194d799ec537693c52e63bf29e98c003
onnx_models/sfa3d_163_fp32.onnx
  50,928,757 B  9cd6f3e9b3e5eec34eeba84151717da5a7314ead72e78264259c27878bea46dd
quantized_models/sfa3d_163_int8.onnx
  12,866,736 B  ddffb1708bbb8cab7c3310f5ff2f1e068e56e2dacaf3a4279bd539c72faf9ba3
```

</details>

### Model export

```bash
# PyTorch → ONNX (the export result already ships with the repo; no need to re-run)
python sfa/export_to_onnx.py \
    --model ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
    --output ./onnx_models/sfa3d_163_fp32.onnx

# ONNX INT8 dynamic quantization: 48.57 MB → 12.27 MB (artifact ships with the repo)
# After quantizing, the script automatically compares FP32/INT8 detections on real
# sample_data inputs
python sfa/quantize_onnx_163.py \
    --onnx_model ./onnx_models/sfa3d_163_fp32.onnx \
    --output ./quantized_models/sfa3d_163_int8.onnx

# Note: sfa/quantize_model_163.py is the PyTorch quantization route, kept only for
# comparison — torch dynamic quantization does not support Conv2d, and this model is
# a pure convolutional network (measured compression: 0%)
```

---

## 📈 Performance

### Per-class detection performance

| Class | mAP@0.5 | Notes |
|------|---------|------|
| Car | **83%** | vehicles (large objects, distinctive features) |
| Cyclist | **65%** | cyclists (small objects, most challenging) |
| Truck | **70%** | trucks (medium objects) |
| **Average** | **73%** | **overall average** |

### Inference speed comparison

| Platform | Framework | Speed | Notes |
|------|------|------|------|
| RTX 4060 Ti | PyTorch FP32 | **110.73 FPS** | GPU inference |
| CPU | ONNX Runtime FP32 | 5.48 FPS | cross-platform (re-measured 5.40) |
| CPU | ONNX Runtime INT8 | 3.23 FPS | benefit is size (−74.7%); x86 dynamic quantization does not speed up conv nets — see the quantization notes in [docs/操作指令/启动训练指令.md](docs/操作指令/启动训练指令.md) (Chinese) |

---

## 📚 Documentation (Chinese)

| Doc | Description |
|------|------|
| [docs/技术方案AAA.md](docs/技术方案AAA.md) | full technical solution ⭐ |
| [docs/SFA3D与SFA4D源码比对报告.md](docs/SFA3D与SFA4D源码比对报告.md) | file-by-file diff vs original SFA3D (honest scoping of this project's changes) ⭐ |
| [docs/技术报告.md](docs/技术报告.md) | in-depth technical analysis report |
| [docs/项目结构说明.md](docs/项目结构说明.md) | project architecture and file descriptions |
| [docs/操作指令/](docs/操作指令/) | command cheat-sheet: training / inference / visualization / quantization |
| [docs/环境依赖清单.md](docs/环境依赖清单.md) | detailed environment requirements |
| [docs/点云维度修改说明.md](docs/点云维度修改说明.md) | 8D→4D mapping implementation details |
| [docs/SFA4D 代码架构.md](docs/SFA4D%20代码架构.md) | code architecture deep-dive |

---

## 🔧 Core Modules

### Data flow

```
raw 8D point cloud (.bin)
    ↓
lidar_mapping.py: read_lidar_file_with_fallback()
    → auto-detects 8D/5D/4D; for 8D takes [0,1,2,4] and maps P to intensity
    ↓
kitti_bev_utils.py: makeBEVMap()
    → 3-channel BEV image [intensity, height, density], 608×608
    ↓
Dataset / DataLoader
    ↓
model input: (B, 3, 608, 608)
    ↓
fpn_resnet.py: PoseResNet + KFPN
    → feature maps for 5 heads (B, C, 152, 152)
    ↓
losses.py: Compute_Loss
    → weighted focal_loss + l1_loss + balanced_l1_loss
    ↓
evaluation_utils.py: decode + post_processing
    → heatmap NMS → Top-K → coordinate decoding → cross-class NMS
    ↓
KITTI-format detections / visualization images
```

### Key configuration parameters

```python
# BEV boundary (sfa/config/kitti_config.py)
boundary = {
    "minX": 0,   "maxX": 50,     # forward 0~50 m
    "minY": -25, "maxY": 25,     # lateral ±25 m
    "minZ": -2.73, "maxZ": 1.27  # height range
}
BEV_WIDTH = 608
BEV_HEIGHT = 608
DISCRETIZATION = 50 / 608  # ≈ 0.0822 m per pixel

# model output heads
heads = {
    'hm_cen': 3,      # per-class heatmap (Car, Cyclist, Truck)
    'cen_offset': 2,  # sub-pixel center offset
    'direction': 2,   # yaw (sin, cos)
    'z_coor': 1,      # Z coordinate
    'dim': 3,         # dimensions (h, w, l)
}
```

---

## 🤝 Contributing

Issues and pull requests are welcome!

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Thanks to the **AIC Global Campus AI Algorithm Elite Competition** for the platform and dataset.
- The network backbone, losses, and BEV generation come directly from [SFA3D](https://github.com/maudzung/SFA3D) (author Nguyen Mau Dung / maudzung, MIT license) — gratefully acknowledged and credited; this project adds data-domain adaptation and post-processing on top (see the diff report).
- Thanks to all team members for their hard work.
- Repository engineering (Docker packaging, unit tests, script archiving, doc revisions) was done with Claude Code assistance; see commit history.

---

## 📧 Contact

- Competition site: [AIC Global Campus AI Algorithm Elite Competition](https://www.aicomp.cn/) (Chinese)
- Email: **webcainiao@gmail.com**

---

> **⭐ If this project helps you, please star it!**
