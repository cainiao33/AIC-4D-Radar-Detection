"""
ONNX INT8 Dynamic Quantization Tool for 163-Epoch Model
将 FP32 ONNX 权重量化为 INT8 动态量化 ONNX（48.57 MB → 约 12.3 MB）

为什么走 ONNX 而不是 PyTorch 量化：
torch.quantization.quantize_dynamic 只支持 Linear/LSTM 等层，不支持 Conv2d；
本模型为纯卷积网络（FPN-ResNet + 全卷积检测头），实测 PyTorch 动态量化压缩率为 0%
（见 sfa/quantize_model_163.py，保留作对照）。INT8 部署请使用本脚本。

量化完成后自动做真实样本验证：用仓库自带的 sample_data 点云生成 BEV 输入，
FP32 与 INT8 逐样本推理，对比热力图偏差与检测数量——无需完整 DRadDataset。

用法（仓库根目录执行）：
python sfa/quantize_onnx_163.py \
    --onnx_model ./onnx_models/sfa3d_163_fp32.onnx \
    --output ./quantized_models/sfa3d_163_int8.onnx
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# sfa 包根目录加入 sys.path（与 quantize_model_163.py 同法）
SRC_DIR = os.path.dirname(os.path.realpath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

REPO_ROOT = os.path.dirname(SRC_DIR)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_bev_inputs(max_samples=3):
    """从仓库自带 sample_data 构造真实 BEV 输入（无需完整数据集）"""
    from data_process.lidar_mapping import read_lidar_file_with_fallback
    from data_process.kitti_bev_utils import makeBEVMap
    from config import kitti_config as cnf

    bins = sorted(glob.glob(os.path.join(REPO_ROOT, 'sample_data', 'training', 'velodyne', '*.bin')))
    if not bins:
        return None, []
    bevs, ids = [], []
    for b in bins[:max_samples]:
        pc = read_lidar_file_with_fallback(b)
        bevs.append(makeBEVMap(pc, cnf.boundary_back).astype(np.float32))
        ids.append(Path(b).stem)
    return np.stack(bevs), ids


def detect(outputs_batch):
    """单样本（含 batch 维）5 头输出 → NMS 后检测结果，返回各类别框数"""
    from utils.evaluation_utils import decode_numpy, post_processing_numpy

    hm_cen = sigmoid_np(outputs_batch[0])
    cen_offset = sigmoid_np(outputs_batch[1])
    dets = decode_numpy(hm_cen, cen_offset, outputs_batch[2], outputs_batch[3],
                        outputs_batch[4], K=50)
    dets = post_processing_numpy(dets, 3, 4, 0.25, inter_class_nms=True, nms_thresh=0.2)
    # 无检测的类别可能是标量 0，统一按 shape[0] 取框数
    return [int(c.shape[0]) if hasattr(c, 'shape') else 0 for c in dets[0].values()]


def main():
    parser = argparse.ArgumentParser(description='ONNX INT8 dynamic quantization for SFA4D 163-epoch model')
    parser.add_argument('--onnx_model', type=str, default='./onnx_models/sfa3d_163_fp32.onnx',
                        help='Path to FP32 ONNX model')
    parser.add_argument('--output', type=str, default='./quantized_models/sfa3d_163_int8.onnx',
                        help='Output path of INT8 ONNX model')
    args = parser.parse_args()

    print("=" * 80)
    print("ONNX INT8 Dynamic Quantization")
    print("=" * 80)

    # 量化（激活动态量化）。权重用 QUInt8：QInt8 会生成 ConvInteger(S8S8) 节点，
    # x86 CPU 版 onnxruntime 无该实现（NOT_IMPLEMENTED）；QUInt8 走 QLinearConv，x86 可跑
    quantize_dynamic(args.onnx_model, args.output, weight_type=QuantType.QUInt8)

    fp32_size = os.path.getsize(args.onnx_model) / (1024 * 1024)
    int8_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"FP32: {fp32_size:.2f} MB -> INT8: {int8_size:.2f} MB "
          f"({(1 - int8_size / fp32_size) * 100:.1f}% smaller)")

    # ---- 真实样本验证：FP32 vs INT8 逐样本对比 ----
    bevs, ids = build_bev_inputs()
    if bevs is None:
        print("未找到 sample_data，退化为随机输入验证")
        bevs = np.random.randn(1, 3, 608, 608).astype(np.float32)
        ids = ['<random>']

    outs = {}
    for tag, path in [('fp32', args.onnx_model), ('int8', args.output)]:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        outs[tag] = sess.run(None, {sess.get_inputs()[0].name: bevs})

    print(f"\n验证样本: {len(ids)} 个（sample_data 真景点云 → BEV）")
    print(f"{'sample':>8} | {'hm均值偏差':>10} | {'FP32检测(Car,Cyc,Trk)':>22} | {'INT8检测':>16}")
    for i, sid in enumerate(ids):
        hm32 = sigmoid_np(outs['fp32'][0][i])
        hm8 = sigmoid_np(outs['int8'][0][i])
        mad = float(np.abs(hm32 - hm8).mean())
        d32 = detect([outs['fp32'][k][i:i + 1] for k in range(5)])
        d8 = detect([outs['int8'][k][i:i + 1] for k in range(5)])
        print(f"{sid:>8} | {mad:10.5f} | {str(d32):>22} | {str(d8):>16}")
    print("Done.")


if __name__ == '__main__':
    main()
