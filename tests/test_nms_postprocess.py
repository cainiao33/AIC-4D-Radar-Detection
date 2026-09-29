"""跨类别 NMS 与后处理特征测试。

覆盖 sfa/utils/evaluation_utils.py 的：
- apply_inter_class_nms：仅跨类抑制；判据 = 伪 IoU > 阈值 或 归一化中心距 < 0.25
- post_processing / post_processing_numpy：10 列输入布局、score 严格 > peak_thresh、
  像素→米制缩放（×down_ratio、w/l ×608/50）、yaw=arctan2(sin, cos)、空类 (0,8) 契约
（evaluation_utils 顶层 import torch/cv2，无 torch 环境自动跳过本文件。）
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('cv2')

from utils.evaluation_utils import (  # noqa: E402
    apply_inter_class_nms,
    post_processing,
    post_processing_numpy,
)

SCALE_WL = 608.0 / 50.0  # BEV_WIDTH/bound_size_y = BEV_HEIGHT/bound_size_x = 12.16


def make_det(score, xs, ys, z, h, w, l, sin_, cos_, cls):
    """10 列输入行：[score, xs, ys, z, h, w, l, sin, cos, cls]。"""
    return [score, xs, ys, z, h, w, l, sin_, cos_, float(cls)]


def as_batch(rows):
    return np.array(rows, dtype=np.float32).reshape(1, len(rows), 10)


def car_row(score=0.9, x=10.0, y=10.0):
    """post_processing 输出格式的 8 列行（已在米制）。"""
    return np.array([score, x, y, 1.0, 1.5, 2.0, 4.0, 0.0], dtype=np.float32)


class TestApplyInterClassNms:
    def test_cross_class_same_center_suppressed(self):
        """Car(0.9) 与 Truck(0.8) 同中心：距离判据 nd=0 < 0.25 → 低分 Truck 被压。"""
        top = {0: np.array([car_row(0.9)]), 1: np.array([car_row(0.8)]), 2: np.array([]).reshape(0, 8)}
        out = apply_inter_class_nms(top, num_classes=3, iou_thresh=0.2)
        assert out[0].shape == (1, 8)
        assert len(out[1]) == 0

    def test_same_class_overlap_never_suppressed(self):
        """类内重叠不去重（类内去重由 decode 的热力图 _nms 负责）。"""
        top = {0: np.array([car_row(0.9), car_row(0.8)]), 1: np.array([]).reshape(0, 8), 2: np.array([]).reshape(0, 8)}
        out = apply_inter_class_nms(top, num_classes=3, iou_thresh=0.2)
        assert out[0].shape == (2, 8)

    def test_far_apart_both_kept(self):
        """距离 nd>1.0 → 伪 IoU=0，两跨类框都保留。"""
        top = {0: np.array([car_row(0.9, 10.0, 10.0)]),
               1: np.array([car_row(0.8, 30.0, 10.0)]),  # dist=20, avg=3 → nd≈6.7
               2: np.array([]).reshape(0, 8)}
        out = apply_inter_class_nms(top, num_classes=3, iou_thresh=0.2)
        assert out[0].shape == (1, 8)
        assert out[1].shape == (1, 8)

    def test_all_empty_passthrough(self):
        """全空输入原样返回同一对象（源码 len(all_dets)==0 分支直接 return top_preds）。"""
        empty = np.array([]).reshape(0, 8)
        top = {0: empty, 1: empty, 2: empty}
        out = apply_inter_class_nms(top, num_classes=3)
        assert out is top  # 同一对象早退，非重建

    def test_output_keys_are_int_range(self):
        """输出键恒为 int 0..num_classes-1，空类为 (0,8) ndarray。"""
        top = {0: np.array([car_row(0.9)]), 1: np.array([]).reshape(0, 8), 2: np.array([]).reshape(0, 8)}
        out = apply_inter_class_nms(top, num_classes=3)
        assert sorted(out.keys()) == [0, 1, 2]
        for c in range(3):
            assert hasattr(out[c], 'shape')
            assert out[c].shape[-1] == 8 if out[c].size else out[c].shape == (0, 8)

    def test_higher_score_wins_regardless_of_class_order(self):
        """Truck 分数更高时，被压的是 Car。"""
        top = {0: np.array([car_row(0.5)]), 1: np.array([car_row(0.95)]), 2: np.array([]).reshape(0, 8)}
        out = apply_inter_class_nms(top, num_classes=3, iou_thresh=0.2)
        assert len(out[0]) == 0
        assert out[1].shape == (1, 8)
        assert out[1][0, 0] == pytest.approx(0.95)


class TestPostProcessingGeometry:
    BOTH = pytest.mark.parametrize('func', [post_processing, post_processing_numpy],
                                   ids=['torch-path', 'numpy-path'])

    @BOTH
    def test_score_threshold_is_strictly_greater(self, func):
        """score == peak_thresh 的框被丢弃（严格 >）。"""
        dets = as_batch([
            make_det(0.25, 1, 1, 1, 1, 1, 1, 0.0, 1.0, 0),
            make_det(0.26, 2, 2, 1, 1, 1, 1, 0.0, 1.0, 0),
        ])
        out = func(dets, 3, 4, peak_thresh=0.25, inter_class_nms=False)[0]
        assert out[0].shape == (1, 8)
        assert out[0][0, 0] == pytest.approx(0.26, abs=1e-6)

    @BOTH
    def test_pixel_to_metric_scaling(self, func):
        """xs/ys × down_ratio；w → w/50×608；l → l/50×608；yaw=arctan2(sin,cos)。"""
        dets = as_batch([make_det(0.9, 2.5, 3.0, -1.0, 1.5, 1.0, 2.0, 1.0, 0.0, 0)])
        row = func(dets, 3, 4, peak_thresh=0.2, inter_class_nms=False)[0][0][0]
        assert row[0] == pytest.approx(0.9, abs=1e-6)          # score
        assert row[1] == pytest.approx(2.5 * 4)                # xs × down_ratio
        assert row[2] == pytest.approx(3.0 * 4)                # ys × down_ratio
        assert row[3] == pytest.approx(-1.0)                   # z 原样
        assert row[4] == pytest.approx(1.5)                    # h 原样
        assert row[5] == pytest.approx(1.0 * SCALE_WL, rel=1e-5)  # w
        assert row[6] == pytest.approx(2.0 * SCALE_WL, rel=1e-5)  # l
        assert row[7] == pytest.approx(np.pi / 2, abs=1e-5)    # yaw = arctan2(1,0)

    @BOTH
    def test_class_split_and_empty_contract(self, func):
        """只给 cls=1 的框：cls0/cls2 为 (0,8)，len==0；从不返回标量。"""
        dets = as_batch([make_det(0.9, 1, 1, 1, 1, 1, 1, 0.0, 1.0, 1)])
        out = func(dets, 3, 4, peak_thresh=0.2, inter_class_nms=False)[0]
        assert out[1].shape == (1, 8)
        for c in (0, 2):
            assert hasattr(out[c], 'shape')
            assert len(out[c]) == 0

    @BOTH
    def test_inter_class_nms_integration(self, func):
        """跨类同中心 → 低分类被压（与 apply_inter_class_nms 行为一致）。"""
        dets = as_batch([
            make_det(0.9, 2.5, 2.5, 1, 1.5, 1, 2, 0.0, 1.0, 0),
            make_det(0.8, 2.5, 2.5, 1, 1.5, 1, 2, 0.0, 1.0, 2),
        ])
        out = func(dets, 3, 4, peak_thresh=0.25, inter_class_nms=True, nms_thresh=0.2)[0]
        assert out[0].shape == (1, 8)
        assert len(out[2]) == 0

    def test_post_processing_accepts_torch_tensor(self):
        """torch 路径：输入 CUDA/cpu 张量经 .cpu().numpy() 守卫转换。"""
        dets = torch.tensor([[[0.9, 2.5, 3.0, -1.0, 1.5, 1.0, 2.0, 1.0, 0.0, 0.0]]],
                            dtype=torch.float32)
        out = post_processing(dets, 3, 4, peak_thresh=0.2, inter_class_nms=False)[0]
        assert out[0].shape == (1, 8)
        assert out[0][0, 1] == pytest.approx(10.0)

    def test_numpy_variant_rejects_nothing_special(self):
        """post_processing_numpy 与 post_processing 在相同 numpy 输入上结果一致。"""
        rng = np.random.RandomState(42)
        rows = [make_det(s, x, y, z, h, w, l, sn, cs, c) for s, x, y, z, h, w, l, sn, cs, c
                in zip(rng.uniform(0.3, 0.9, 6), rng.uniform(0, 100, 6), rng.uniform(0, 100, 6),
                       rng.uniform(-2, 1, 6), rng.uniform(1, 2, 6), rng.uniform(1, 3, 6),
                       rng.uniform(2, 5, 6), rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6),
                       rng.randint(0, 3, 6))]
        dets = as_batch(rows)
        a = post_processing(dets.copy(), 3, 4, 0.25, inter_class_nms=True, nms_thresh=0.2)[0]
        b = post_processing_numpy(dets.copy(), 3, 4, 0.25, inter_class_nms=True, nms_thresh=0.2)[0]
        for c in range(3):
            assert a[c].shape == b[c].shape
            if len(a[c]):
                assert np.allclose(a[c], b[c], atol=1e-6)
