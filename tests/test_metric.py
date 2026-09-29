"""伪 IoU（compute_bev_box_iou）分段特征测试。

实现在 sfa/utils/evaluation_utils.py:125 —— 中心距离启发式，非旋转矩形几何交叠：
nd = dist / avg_size（avg_size = (w1+l1+w2+l2)/4，w=box[5], l=box[6]，其余列为死代码）：
  nd > 1.0 → 0.0；nd < 0.2 → 0.9；nd < 0.4 → 0.7；否则 max(0, 1.2 - nd)
（分段边界因严格不等号落入下一档：nd==0.2 → 0.7，nd==0.4 → 0.8，nd==1.0 → 0.2。）
"""

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('cv2')

from utils.evaluation_utils import compute_bev_box_iou  # noqa: E402


def box(x, y, w=2.0, l=2.0, score=0.9):
    """[score, x, y, z, h, w, l, yaw] —— 仅 x/y/w/l 参与计算。"""
    return np.array([score, x, y, 1.0, 1.5, w, l, 0.3], dtype=np.float32)


@pytest.mark.parametrize('offset, expected', [
    (0.0, 0.9),    # nd=0.00 < 0.2 → 0.9（同中心）
    (0.3, 0.9),    # nd=0.15 < 0.2 → 0.9
    (0.6, 0.7),    # nd=0.30 ∈ [0.2,0.4) → 0.7
    (1.2, 0.6),    # nd=0.60 ∈ [0.4,1.0] → 1.2-0.6=0.6
    (1.8, 0.3),    # nd=0.90 → 1.2-0.9=0.3
    (2.2, 0.0),    # nd=1.10 > 1.0 → 0.0（远距判不重叠）
])
def test_segments(offset, expected):
    """w=l=2 → avg_size=2，x 偏移即 dist，nd = offset/2。"""
    iou = compute_bev_box_iou(box(0.0, 0.0), box(offset, 0.0))
    assert iou == pytest.approx(expected, abs=1e-6)


def test_boundary_nd_exactly_1_gives_0_2():
    """nd==1.0 不满足 > 1.0，落入 else → 1.2-1.0 = 0.2（严格不等号边界）。"""
    iou = compute_bev_box_iou(box(0.0, 0.0), box(2.0, 0.0))  # sqrt(2²)=2 精确
    assert iou == pytest.approx(0.2, abs=1e-6)


def test_zero_avg_size_returns_zero():
    a = box(0.0, 0.0, w=0.0, l=0.0)
    b = box(0.0, 0.0, w=0.0, l=0.0)
    assert compute_bev_box_iou(a, b) == 0.0


def test_y_axis_distance_counts():
    iou_y = compute_bev_box_iou(box(0.0, 0.0), box(0.0, 1.2))
    iou_x = compute_bev_box_iou(box(0.0, 0.0), box(1.2, 0.0))
    assert iou_y == pytest.approx(iou_x)
    assert iou_y == pytest.approx(0.6, abs=1e-6)


def test_yaw_z_h_score_are_dead_code():
    """特征行为：yaw/z/h/score 列不参与计算。"""
    base = compute_bev_box_iou(box(0.0, 0.0), box(1.2, 0.0))
    varied = compute_bev_box_iou(
        np.array([0.1, 0.0, 0.0, 9.9, 5.0, 2.0, 2.0, -1.7], dtype=np.float32),
        np.array([0.8, 1.2, 0.0, -3.0, 0.2, 2.0, 2.0, 2.5], dtype=np.float32),
    )
    assert varied == pytest.approx(base, abs=1e-6)


def test_output_range():
    """可达输出全集 {0.0, 0.9, 0.7} ∪ [0.2, 0.8]，范围 [0.0, 0.9]。"""
    for off in np.linspace(0.0, 3.0, 31):
        v = float(compute_bev_box_iou(box(0.0, 0.0), box(float(off), 0.0)))
        assert 0.0 <= v <= 0.9
