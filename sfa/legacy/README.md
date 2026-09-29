# sfa/legacy/ — 实验脚本归档

竞赛期间的实验、变体与调试脚本。**不在维护范围内，正确性与结果不做保证**；
正式入口见仓库根 README（`sfa/train.py` / `sfa/eval.py` / `sfa/export_to_onnx.py` 等）。

归档在 `sfa/legacy/` 而非仓库外目录，是因为这些脚本的 `sys.path` 自举逻辑
（`while not src_dir.endswith("sfa")`）要求路径中含 `sfa/` 段，移出后会失效。
运行方式不变：

```bash
python sfa/legacy/<脚本名>.py    # 与原先从 sfa/ 目录运行等价
```

> ⚠️ **路径注意**：脚本内部按 `__file__/..` 解析仓库根（当年在 `sfa/` 根时 `..` 即仓库根），
> 归档后深了一层，`..` 变成 `sfa/`——**相对路径参数（如 `--pretrained_path ./checkpoints/...`）
> 会错位到 `sfa/` 下**。规避：`--pretrained_path` 等路径参数传**绝对路径**即可正常运行
> （绝对路径在 `os.path.join` 中直接短路，其余逻辑不受影响；`sys.path` 自举仍正确）。
> 实测（2026-09）：`testing_export_ultra_aggressive.py` 以绝对路径权重运行，输出与
> `sfa/eval.py` 逐字节一致。

## 一、推理/导出变体（当年调参的中间产物，功能已被正式入口吸收）

| 脚本 | 说明 |
|------|------|
| `testing_export.py` | testing_export 家族基型（peak 0.2 / NMS 0.5） |
| `testing_export_direct.py` | 免 ImageSets 变体（NMS 0.3 硬编码） |
| `testing_export_ultra_aggressive.py` | **竞赛定稿**（peak 0.25 / 跨类 NMS 0.2），已被 `sfa/eval.py` 逐行继承 |
| `testing_export_ultra_aggressive_fs.py` | 上者的 `DRadDataset - fs` 数据集副本，仅默认路径不同 |
| `testing_validation_evaluation.py` | 验证集推理 + 内置 mAP 一体化 |
| `validation_testing_163.py` | 验证集推理 + 指标（注意其 NMS 实际用默认 0.5 而非 0.2） |
| `validation_ultra_aggressive_163.py` / `ultra_aggressive_163_val.py` | 超激进脚本的验证集适配版（两份重复） |
| `validation_inference_163.py` / `evaluate_validation.py` | 验证集推理的其他变体 |
| `run_163_working.py` | testing.py 的调参工作副本（当年"跑通"现场） |
| `run_163_quantized.py` | PyTorch 量化模型推理验证（对照实验，见 `sfa/quantize_model_163.py`） |

## 二、绕过标准管线的推理（结果与正式管线**不可比**）

以下脚本自写 BEV 归一化，与训练用 `makeBEVMap` 预处理不一致：

| 脚本 | 说明 |
|------|------|
| `direct_radar_inference_163.py` | 直接读 velodyne，自制 BEV |
| `radar_val_163_corrected.py` | "修正版"，BEV 为手写网格归一化（高度/5、强度/100、密度/50） |
| `final_163_inference.py` | 自建处理链的"最终版"验证推理 |

## 三、⚠️ 伪结果/已损坏脚本（**任何数字不得引用**）

| 脚本 | 问题 |
|------|------|
| `analyze_163epoch_performance.py` | "性能分析"数字全部硬编码，不读任何模型/数据 |
| `simple_onnx_evaluation.py` | 两个互斥的相对根 + 凭空公式的"预估 mAP" |
| `run_163_simple.py` | BEV 输入是随机噪声，输出无意义 |
| `simple_val_inference_163.py` | BEV 用随机噪声替代真实点云 |
| `simple_val_163_final.py` | 随机占位图 + `cfg.K`/`math` 两处未定义潜在崩溃 |
| `evaluate_onnx_results.py` | `from utils.evaluate import evaluate` —— 模块不存在，import 即崩 |
| `simple_class_imbalance_augmentor.py` | 纯概念演示，全部使用随机假数据 |

保留它们是为了诚实呈现竞赛过程的真实状态；正式指标以 README 与
`docs/SFA3D与SFA4D源码比对报告.md`、`sfa/kitti_evaluation_163.py` 的评测链路为准。

## 四、可视化与其他

| 脚本 | 说明 |
|------|------|
| `batch_visualize_ultra_aggressive.py` | 旧一代批量可视化（前身，未修正纵横比） |
| `visualize_with_3d_boxes.py` | 可视化变体（含 `--rotate-bev`） |
| `visualize_inference_results.py` | 可视化 + 8D 加载自检 |
| `class_imbalance_augmentor.py` | 类别不平衡增强实验（真实数据版） |
| `KITTIEVAL_163_COMPLETE.py` | subprocess 串联推理→评测的编排脚本 |
| `CORRECT_TEST_COMMANDS.txt` / `QUICK_TEST_COMMAND.txt` / `TRAINING_SET_COMMANDS.txt` | 当年的手抄命令备忘 |

现行可视化入口：`sfa/batch_visualize_3d_boxes_final.py`、`sfa/create_video_from_images.py`。
