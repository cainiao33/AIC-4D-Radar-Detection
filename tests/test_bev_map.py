"""BEV 生成特征测试：固化 makeBEVMap / get_filtered_lidar 的行为。

makeBEVMap 输入约定：Nx4 [x, y, z_shifted, intensity]，z 已由 get_filtered_lidar
平移到 [0, 4]；输出 (3, 608, 608) float64，通道 [intensity, height, density]。
像素索引：row = floor(x / DISCRETIZATION)（x 米，minX=0 无偏移）；
col = trunc(floor(y / DISCRETIZATION) + 304.5)。
"""

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')  # noqa: F401  kitti_bev_utils 顶层 import cv2

import config.kitti_config as cnf  # noqa: E402
from data_process.kitti_bev_utils import makeBEVMap  # noqa: E402
from data_process.kitti_data_utils import get_filtered_lidar  # noqa: E402

DISC = cnf.DISCRETIZATION  # 50/608 ≈ 0.0822368


def test_shape_and_dtype():
    out = makeBEVMap(np.zeros((0, 4)), cnf.boundary)
    assert out.shape == (3, 608, 608)
    assert out.dtype == np.float64


def test_empty_point_cloud_all_zero():
    out = makeBEVMap(np.zeros((0, 4)), cnf.boundary)
    assert np.all(out == 0.0)


def test_single_point_channels():
    """点 (x=1.0, y=0.0, z=2.0, intensity=0.5)：
    row=floor(1.0/DISC)=12, col=304；height=2/4=0.5；density=log(2)/log(64)=1/6。"""
    pc = np.array([[1.0, 0.0, 2.0, 0.5]])
    out = makeBEVMap(pc, cnf.boundary)
    row, col = 12, 304
    assert out[0, row, col] == pytest.approx(0.5)                    # intensity 原值
    assert out[1, row, col] == pytest.approx(2.0 / 4.0)              # height / (maxZ-minZ)
    assert out[2, row, col] == pytest.approx(np.log(2) / np.log(64))  # density, 1 点 → 1/6
    # 其余像素为零
    mask = np.ones_like(out[0], dtype=bool)
    mask[row, col] = False
    assert np.all(out[0][mask] == 0.0)


def test_cell_keeps_top_z_point():
    """同一 BEV 格取 z 最大的点（height 与 intensity 均取该点），density 按点数。"""
    pc = np.array([
        [1.0, 0.0, 1.0, 0.3],
        [1.0, 0.0, 3.0, 0.9],
    ])
    out = makeBEVMap(pc, cnf.boundary)
    row, col = 12, 304
    assert out[1, row, col] == pytest.approx(3.0 / 4.0)   # 高度取 z=3.0
    assert out[0, row, col] == pytest.approx(0.9)         # intensity 取同一（顶部）点
    assert out[2, row, col] == pytest.approx(np.log(3) / np.log(64))  # 2 点 → log(3)/log(64)


def test_y_minus25_maps_to_col0():
    """y=-25（南边界）→ col = trunc(floor(-304.0 + 304.5)) = 0。"""
    pc = np.array([[1.0, -25.0, 2.0, 0.7]])
    out = makeBEVMap(pc, cnf.boundary)
    assert out[0, 12, 0] == pytest.approx(0.7)


def test_x_at_max_boundary_is_cropped():
    """特征行为：x=50 落在 row 608，被 [:608,:608] 裁掉（边界点消失）。"""
    pc = np.array([[50.0, 0.0, 2.0, 0.9]])
    out = makeBEVMap(pc, cnf.boundary)
    assert np.all(out == 0.0)


class TestGetFilteredLidar:
    def test_closed_interval_crop_and_z_shift(self):
        """点过滤为闭区间 [minX,maxX]×[minY,maxY]×[minZ,maxZ]，随后 z -= minZ。"""
        pts = np.array([
            [0.0, -25.0, -2.73, 0.5],   # 三轴下边界 → 保留
            [50.0, 25.0, 1.27, 0.5],    # 三轴上边界 → 保留
            [50.1, 0.0, 0.0, 0.5],      # x 越界 → 剔除
            [0.0, 25.1, 0.0, 0.5],      # y 越界 → 剔除
            [0.0, 0.0, -2.74, 0.5],     # z 越界 → 剔除
            [1.0, 0.0, 0.0, 0.5],       # 内部点 → 保留
        ])
        out = get_filtered_lidar(pts.copy(), cnf.boundary)
        assert out.shape == (3, 4)
        assert np.allclose(out[:, 2], [0.0, 4.0, 2.73])  # z 平移到 [0,4]

    def test_no_labels_returns_array_only(self):
        out = get_filtered_lidar(np.array([[1.0, 0.0, 0.0, 0.5]]), cnf.boundary)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 4)
