"""8D→4D SNR 映射（Scheme B）特征测试：固化 sfa/data_process/lidar_mapping.py 的行为。

仅依赖 numpy。公式：零值 P → 0.1；非零 P ∈ [-1.36, 6.43] 线性映射到 [0.2, 1.0]，
超出界外 clip。
"""

import numpy as np
import pytest

from data_process.lidar_mapping import (
    SCHEME_B_NONZERO_MAX,
    SCHEME_B_NONZERO_MIN,
    SCHEME_B_NONZERO_RANGE,
    SCHEME_B_ZERO_MAPPING,
    map_P_to_intensity_scheme_B,
    process_8d_lidar_data_scheme_B,
    read_lidar_file_with_fallback,
)


class TestSchemeBConstants:
    def test_constants_match_documentation(self):
        """Scheme B 标定界：零值→0.1，非零 [-1.36, 6.43] → [0.2, 1.0]。"""
        assert SCHEME_B_ZERO_MAPPING == 0.1
        assert (SCHEME_B_NONZERO_MIN, SCHEME_B_NONZERO_MAX) == (-1.36, 6.43)
        assert SCHEME_B_NONZERO_RANGE == (0.2, 1.0)


class TestMapPToIntensitySchemeB:
    def test_zero_maps_to_0_1(self):
        out = map_P_to_intensity_scheme_B(np.array([0.0, 0.0, 0.0]))
        assert np.all(out == pytest.approx(0.1))

    def test_all_zero_early_return(self):
        """全零输入走提前返回分支，结果仍全 0.1。"""
        out = map_P_to_intensity_scheme_B(np.zeros(5))
        assert out.shape == (5,)
        assert np.all(out == pytest.approx(0.1))

    def test_calibration_anchors(self):
        """P=-1.36 → 0.2；P=6.43 → 1.0（标定界端点）。"""
        out = map_P_to_intensity_scheme_B(np.array([-1.36, 6.43]))
        assert out[0] == pytest.approx(0.2, abs=1e-6)
        assert out[1] == pytest.approx(1.0, abs=1e-6)

    def test_out_of_range_clipped(self):
        out = map_P_to_intensity_scheme_B(np.array([100.0, -5.0]))
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.2)

    def test_midpoint(self):
        """P=2.535（区间中点）→ (0.2+1.0)/2 = 0.6。"""
        out = map_P_to_intensity_scheme_B(np.array([2.535]))
        assert out[0] == pytest.approx(0.6, abs=1e-6)

    def test_mixed_zero_and_nonzero(self):
        out = map_P_to_intensity_scheme_B(np.array([0.0, 6.43, 0.0]))
        assert out[0] == pytest.approx(0.1)
        assert out[1] == pytest.approx(1.0, abs=1e-6)
        assert out[2] == pytest.approx(0.1)

    def test_output_dtype_and_shape(self):
        p = np.random.RandomState(0).randn(10).astype(np.float32)
        out = map_P_to_intensity_scheme_B(p)
        assert out.shape == p.shape
        assert out.dtype == np.float32

    def test_nan_passthrough(self):
        """特征行为：NaN 走非零分支，clip 不改变 NaN，输出 NaN（无防护）。"""
        out = map_P_to_intensity_scheme_B(np.array([np.nan]))
        assert np.isnan(out[0])


class TestProcess8D:
    def test_columns_and_mapping(self):
        """输出 Nx4 = [x, y, z 原样] + [P 列经 Scheme B 映射]；Doppler/Range/Az/Elev 丢弃。"""
        pts = np.array([
            [1.0, 2.0, 3.0, 9.9, 0.0, 7.7, 8.8, 9.9],
            [4.0, 5.0, 6.0, -1.0, 6.43, -2.0, -3.0, -4.0],
        ], dtype=np.float32)
        out = process_8d_lidar_data_scheme_B(pts)
        assert out.shape == (2, 4)
        assert out.dtype == np.float32
        assert out[0, 0] == 1.0 and out[0, 1] == 2.0 and out[0, 2] == 3.0
        assert out[0, 3] == pytest.approx(0.1)
        assert out[1, 3] == pytest.approx(1.0, abs=1e-6)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Nx8"):
            process_8d_lidar_data_scheme_B(np.zeros((3, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="Nx8"):
            process_8d_lidar_data_scheme_B(np.zeros(8, dtype=np.float32))

    def test_empty_input(self):
        out = process_8d_lidar_data_scheme_B(np.zeros((0, 8), dtype=np.float32))
        assert out.shape == (0, 4)


class TestReadLidarFileWithFallback:
    def _write_bin(self, tmp_path, data, name='frame.bin'):
        f = tmp_path / name
        np.asarray(data, dtype=np.float32).tofile(str(f))
        return str(f)

    def test_8d_applies_scheme_b(self, tmp_path):
        # 1 个 8D 点：P=0 → intensity=0.1
        f = self._write_bin(tmp_path, [1, 2, 3, 4, 0, 6, 7, 8])
        out = read_lidar_file_with_fallback(f)
        assert out.shape == (1, 4)
        assert out[0, 3] == pytest.approx(0.1)

    def test_5d_selects_columns_0124(self, tmp_path):
        # 5 个 float（不整除 8）→ 5D 路径，取列 [0,1,2,4]，无映射
        f = self._write_bin(tmp_path, [10, 11, 12, 13, 99])
        out = read_lidar_file_with_fallback(f)
        assert out.shape == (1, 4)
        assert list(out[0]) == [10, 11, 12, 99]

    def test_4d_passthrough(self, tmp_path):
        # 4 个 float → 原样
        f = self._write_bin(tmp_path, [1, 2, 3, 4])
        out = read_lidar_file_with_fallback(f)
        assert out.shape == (1, 4)
        assert list(out[0]) == [1, 2, 3, 4]

    def test_ambiguous_length_prefers_8d(self, tmp_path):
        """40 个 float 同时整除 8/5/4：契约是按 5 个 8D 点处理（8 优先）。"""
        data = list(range(40))
        f = self._write_bin(tmp_path, data)
        out = read_lidar_file_with_fallback(f)
        assert out.shape == (5, 4)
        # 第 1 个 8D 点 = [0,1,2,3,4,5,6,7]，P=4 → 非零映射，非原值 4
        assert out[0, 0] == 0.0 and out[0, 1] == 1.0 and out[0, 2] == 2.0
        assert out[0, 3] != 4.0

    def test_incompatible_length_raises(self, tmp_path):
        f = self._write_bin(tmp_path, [1, 2, 3, 4, 5, 6, 7])
        with pytest.raises(ValueError, match="Unexpected LiDAR data length"):
            read_lidar_file_with_fallback(f)

    def test_empty_file_returns_empty_4d(self, tmp_path):
        f = self._write_bin(tmp_path, [])
        out = read_lidar_file_with_fallback(f)
        assert out.shape == (0, 4)
