"""
单元测试 — data_analyzer 模块
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_analyzer import (
    ac_value,
    batch_ac_analysis,
    distribution_analysis,
    frequency_analysis,
    hot_cold_analysis,
    missing_analysis,
)


# ── 测试数据生成器 ────────────────────────────────────────

def make_sample_df(n: int = 200) -> pd.DataFrame:
    """生成合规的随机测试数据。"""
    import random

    rows = []
    for i in range(n):
        reds = sorted(random.sample(range(1, 34), 6))
        blue = random.randint(1, 16)
        rows.append({
            "issue": str(2000000 + i),
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 3),
            "r1": reds[0], "r2": reds[1], "r3": reds[2],
            "r4": reds[3], "r5": reds[4], "r6": reds[5],
            "blue": blue,
        })
    df = pd.DataFrame(rows)
    df["date"] = df["date"].dt.date
    return df


SAMPLE_DF = make_sample_df(200)


# ── AC 值测试 ─────────────────────────────────────────────

class TestACValue:
    def test_min_ac(self):
        """连续6个数的 AC 值最小（应为0）。"""
        assert ac_value([1, 2, 3, 4, 5, 6]) == 0

    def test_max_ac(self):
        """分散号码的 AC 值较大。"""
        v = ac_value([1, 5, 12, 20, 28, 33])
        assert v >= 4

    def test_range(self):
        """AC 值范围在 0-9 之间。"""
        import random
        for _ in range(100):
            nums = sorted(random.sample(range(1, 34), 6))
            v = ac_value(nums)
            assert 0 <= v <= 9, f"AC 值 {v} 超范围，号码={nums}"

    def test_batch(self):
        """batch_ac_analysis 应返回与数据行数相同长度的 Series。"""
        result = batch_ac_analysis(SAMPLE_DF)
        assert len(result) == len(SAMPLE_DF)
        assert (result >= 0).all()
        assert (result <= 9).all()


# ── 频率分析测试 ──────────────────────────────────────────

class TestFrequencyAnalysis:
    def test_returns_all_keys(self):
        result = frequency_analysis(SAMPLE_DF)
        for key in ("red_freq", "red_rate", "blue_freq", "blue_rate"):
            assert key in result

    def test_red_freq_index(self):
        result = frequency_analysis(SAMPLE_DF)
        assert list(result["red_freq"].index) == list(range(1, 34))

    def test_blue_freq_index(self):
        result = frequency_analysis(SAMPLE_DF)
        assert list(result["blue_freq"].index) == list(range(1, 17))

    def test_rate_sums_to_one(self):
        result = frequency_analysis(SAMPLE_DF)
        assert abs(result["red_rate"].sum() - 1.0) < 1e-6
        assert abs(result["blue_rate"].sum() - 1.0) < 1e-6

    def test_total_periods(self):
        result = frequency_analysis(SAMPLE_DF)
        assert result["total_periods"] == len(SAMPLE_DF)


# ── 遗漏分析测试 ──────────────────────────────────────────

class TestMissingAnalysis:
    def test_current_missing_non_negative(self):
        result = missing_analysis(SAMPLE_DF)
        assert (result["red_current_missing"] >= 0).all()
        assert (result["blue_current_missing"] >= 0).all()

    def test_max_missing_ge_current(self):
        result = missing_analysis(SAMPLE_DF)
        assert (result["red_max_missing"] >= result["red_current_missing"]).all()
        assert (result["blue_max_missing"] >= result["blue_current_missing"]).all()

    def test_all_nums_covered(self):
        result = missing_analysis(SAMPLE_DF)
        assert len(result["red_current_missing"]) == 33
        assert len(result["blue_current_missing"]) == 16


# ── 热冷分析测试 ──────────────────────────────────────────

class TestHotColdAnalysis:
    def test_zscore_index(self):
        result = hot_cold_analysis(SAMPLE_DF, window=30)
        assert len(result["red_zscore"]) == 33
        assert len(result["blue_zscore"]) == 16

    def test_hot_cold_lists(self):
        result = hot_cold_analysis(SAMPLE_DF, window=30)
        for n in result["red_hot"]:
            assert 1 <= n <= 33
        for n in result["red_cold"]:
            assert 1 <= n <= 33

    def test_window_parameter(self):
        for w in [10, 20, 50]:
            result = hot_cold_analysis(SAMPLE_DF, window=w)
            assert result["window"] == w


# ── 分布分析测试 ──────────────────────────────────────────

class TestDistributionAnalysis:
    def test_sum_stats(self):
        result = distribution_analysis(SAMPLE_DF)
        stats = result["sum_stats"]
        # 理论最小和值 = 1+2+3+4+5+6=21, 最大=28+29+30+31+32+33=183
        assert 21 <= stats["min"] <= 183
        assert 21 <= stats["max"] <= 183
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_odd_even(self):
        result = distribution_analysis(SAMPLE_DF)
        # 每期奇数个数为 0-6
        dist = result["odd_even_dist"]
        assert all(0 <= k <= 6 for k in dist.index)

    def test_consecutive_non_negative(self):
        result = distribution_analysis(SAMPLE_DF)
        dist = result["consecutive_dist"]
        assert all(k >= 0 for k in dist.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
