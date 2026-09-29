# SFA3D 与 SFA4D 源码比对报告

> **目的**：逐文件比对 SFA4D 与原始 SFA3D 的源码，核实 README 所列"核心创新"的真实性，给出真实创新点与改进清单。
> **比对基准**：原版 [maudzung/SFA3D](https://github.com/maudzung/SFA3D)（2026-09 克隆）`sfa/` 目录，共 29 个 Python 文件。
> **比对方法**：逐文件 `diff -u` + MD5 校验 + 关键声明逐条溯源（模型结构、损失、BEV、后处理、数据入口）。
> **结论日期**：2026-09-29

---

## 一、总账：29 个共有文件的真实改动分布

| 类别 | 数量 | 文件 |
|---|---|---|
| **逐字节相同**（MD5 一致） | 12 | `models/fpn_resnet.py`、`losses/losses.py`、`data_process/kitti_bev_utils.py`、`utils/torch_utils.py`、`utils/misc.py`、`utils/logger.py`、`utils/demo_utils.py` + 5 个 `__init__.py` |
| **仅删作者署名/文件头注释**（功能 0 改动） | 6 | `kitti_data_utils.py`、`kitti_dataloader.py`、`transformation.py`、`model_utils.py`、`demo_front.py`、`demo_2_sides.py` |
| **有实质功能改动** | 10 | `evaluation_utils.py`（±241 行）、`kitti_dataset.py`（±40）、`train.py`（±37）、`train_config.py`（±34）、`test.py`（±31）、`kitti_config.py`（±18）、`train_utils.py`（±13）、`resnet.py`（±4）、`demo_dataset.py`（±4）、`visualization_utils.py`（±4） |

另有 1 个新增核心模块（`data_process/lidar_mapping.py`，约 105 行）和约 37 个新增顶层工程脚本（推理/评估/量化/可视化）。

**核心事实：网络结构、损失函数、BEV 生成、数据增强、学习率策略全部原封未动。** 真正的算法改动集中在两处——数据入口（1 个新文件）和后处理（1 个文件）。

> 注：`utils/lr_scheduler.py` 原与上游逐字节相同；2026-09 仓库整理时修复了上游遗留的 `types` 未导入 bug（`state_dict()` 中的 `NameError` 隐患，+2 行 `import types`），`utils/visualization_utils.py` 同批修复 1 处 `int32` 未定义（→ `np.int32`）。两处均为纯 bug 修复，无功能改动，故不再计入"逐字节相同"。

---

## 二、README 五大"核心创新"逐条核实

| README 声称 | 源码事实 | 判定 |
|---|---|---|
| 🧠 KFPN 特征融合 | `sfa/models/fpn_resnet.py` 与原版 **MD5 完全一致**。`apply_kfpn()`（softmax 加权融合，fpn_resnet.py:203-213）就是 SFA3D 原版代码——KFPN（Keypoint Feature Pyramid Network）本来就是 SFA3D 论文自己的结构 | ❌ **继承自 SFA3D，非创新** |
| ⚡ 无锚点检测（CenterNet 式） | SFA3D 本身就是 anchor-free 五头结构（hm_cen / cen_offset / direction / z_coor / dim），heads 定义原样未改 | ❌ 继承 |
| 🌐 端到端 7-DOF | 同上，SFA3D 原生能力 | ❌ 继承 |
| 🎯 8D→4D 智能映射 | 新文件 `sfa/data_process/lidar_mapping.py`，确为原创代码 | ✅ **真实创新** |
| 🔄 跨类别 NMS | `sfa/utils/evaluation_utils.py:152-216` 新增 `apply_inter_class_nms()`，原版没有 | ✅ 真实新增（但实现有水分，见 §3.2） |

---

## 三、真实的创新点（按含金量排序）

### 3.1 8D→4D SNR 映射 —— 全项目最核心的原创代码 ⭐

新文件 `sfa/data_process/lidar_mapping.py`（约 105 行），本质是**雷达→LiDAR 数据域适配**，让 SFA3D 的 BEV intensity 通道能直接吃 4D 毫米波雷达：

- 从 8 维 `[x, y, z, Doppler, P, Range, Azimuth, Elevation]` 取 `[0, 1, 2, 4]`，第 5 维 P（SNR）→ intensity；
- **Scheme B 分段映射**（lidar_mapping.py:13-45）：
  - 零值（占 98.5%）→ 固定 `0.1`；
  - 非零值按标定界 `[-1.36, 6.43]` 线性归一化到 `[0.2, 1.0]`；
- 这是分布层面的手工分位标定，常数来自统计实验，是"SFA3D 能否吃雷达数据"的承重墙。

⚠️ **伴生工程缺陷**：`read_lidar_file_with_fallback`（lidar_mapping.py:79）用"总元素数 % 8 == 0"优先试 8D——任何偶数点数的 4D 文件（4N 个 float 恰好被 8 整除）都会被误判成 8D。AGENTS.md「常见陷阱」一节已承认此问题。

### 3.2 跨类别 NMS —— 真实新增，但"改进 IoU"名不符实 ⚠️

`sfa/utils/evaluation_utils.py:152-216` 的 `apply_inter_class_nms()` 为原创：对不同类别间的重复框抑制，三判据（伪 IoU 超阈、归一化中心距 < 0.25、< 0.1）。

诚实说明：`compute_bev_box_iou()`（evaluation_utils.py:122-146）的 docstring 声称 *"more accurate IoU based on rotated rectangle overlap"*，**实际未计算旋转矩形交叠**——是中心距离启发式 + 手工分段（dist/size < 0.2 → 0.9；< 0.4 → 0.7；否则 1.2 − d）。它是调参产物，不是几何计算。

### 3.3 超激进后处理参数（竞赛成绩的真实来源）

`sfa/testing_export_ultra_aggressive.py`：`peak_thresh` 0.2 → 0.25、跨类 NMS 阈 0.5 → 0.2、K = 50。75 mAP 的产出链路 = **原版网络 + 这套后处理调参**。

### 3.4 类别重映射

`sfa/config/kitti_config.py:4-15`：KITTI 的 Ped/Car/Cyclist（Van 并入 Car）→ DRad 的 Car/Cyclist/Truck（Van/Ped → −99 丢弃）。配置级改动，属必要适配。**BEV 边界（0–50 m / ±25 m / 608×608）与原版 KITTI 前视配置完全相同**，未针对雷达量程重新设计。

---

## 四、真实但属于工程/部署级的改进（非算法创新）

1. **ONNX 部署栈**：`decode_numpy` / `post_processing_numpy` / `_topk_numpy`（evaluation_utils.py:330-502，约 200 行）——用 `scipy.maximum_filter` 替代 max_pool 做热力图 NMS，纯 NumPy 后处理支撑 CPU ONNX Runtime 推理。
2. **INT8 动态量化**：48.57 MB → 12.27 MB（−74.7%），实测 3 样本与 FP32 检测一致；保留 torch 量化对照路线（Conv2d 动态量化实测压缩率 0%，结论诚实）。
3. **训练基建**：best-model 按 val_loss 另存、Windows 兼容序列化（`_use_new_zipfile_serialization=False`）、数据集子目录可配置 / ImageSets 缺失时自动枚举兜底、非交互推理模式。
4. **约 37 个新增脚本**：评估 / 可视化 / 验证 / 163 epoch 专项分析等竞赛工程。

---

## 五、名不符实之处（对外表述需修正）

1. **"KFPN 特征融合"不能列为创新**——那是 SFA3D 论文自己的贡献，本项目代码与之逐字节相同。用于技术报告/答辩会被内行一眼戳穿。
2. **仓库定位"Radar-Camera-Fusion"无相机融合**：全链路为雷达单模态（BEV 输入只有雷达点云）。`get_image()` 仅用来定位 calib 文件路径（kitti_dataset.py:150-153），与原版 SFA3D 用法完全一样，图片不进网络。
3. `compute_bev_box_iou` 的"改进 IoU"实为中心距离启发式，docstring 与实现不符。
4. 6 个文件删除了原作者署名（Nguyen Mau Dung / CenterNet 代码链）——MIT 协议下不违法，但建议保留出处。

---

## 六、一句话总结

**SFA4D 的真实创新 = 一个数据域适配模块（8D→4D SNR 的 Scheme B 分段映射）+ 一个距离启发式跨类 NMS + 一套面向竞赛指标的后处理调参；其余是 SFA3D 原代码 + ONNX/INT8 部署工程。**

改动总量：29 个共有文件中 19 个零功能改动；真正的算法新代码约 300 行（lidar_mapping 105 行 + 跨类 NMS 约 100 行 + 超激进参数），集中在"数据进"和"结果出"两端，**网络本体一行未动**。这符合"数据适配 + 后处理"的技术路线定位，但 README 的创新表述需按 §二 对照修正。
