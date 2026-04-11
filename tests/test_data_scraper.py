"""
单元测试 — data_scraper 模块（离线验证，不发起真实网络请求）
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_scraper import _validate_row, _to_dataframe, DataManager


class TestValidateRow:
    def test_valid_row(self):
        row = {
            "issue": "2024001",
            "date": "2024-01-02",
            "red_balls": [3, 11, 18, 22, 28, 31],
            "blue_ball": 7,
        }
        assert _validate_row(row) is True

    def test_wrong_red_count(self):
        row = {
            "issue": "2024001",
            "date": "2024-01-02",
            "red_balls": [3, 11, 18, 22, 28],   # 只有5个
            "blue_ball": 7,
        }
        assert _validate_row(row) is False

    def test_red_out_of_range(self):
        row = {
            "issue": "2024001",
            "date": "2024-01-02",
            "red_balls": [0, 11, 18, 22, 28, 31],  # 0 超范围
            "blue_ball": 7,
        }
        assert _validate_row(row) is False

    def test_duplicate_red(self):
        row = {
            "issue": "2024001",
            "date": "2024-01-02",
            "red_balls": [11, 11, 18, 22, 28, 31],  # 重复
            "blue_ball": 7,
        }
        assert _validate_row(row) is False

    def test_blue_out_of_range(self):
        row = {
            "issue": "2024001",
            "date": "2024-01-02",
            "red_balls": [3, 11, 18, 22, 28, 31],
            "blue_ball": 17,  # 超范围
        }
        assert _validate_row(row) is False

    def test_empty_issue(self):
        row = {
            "issue": "",
            "date": "2024-01-02",
            "red_balls": [3, 11, 18, 22, 28, 31],
            "blue_ball": 7,
        }
        assert _validate_row(row) is False


class TestToDataFrame:
    def test_basic_conversion(self):
        records = [
            {
                "issue": "2024001",
                "date": "2024-01-02",
                "red_balls": [3, 11, 18, 22, 28, 31],
                "blue_ball": 7,
            }
        ]
        df = _to_dataframe(records)
        assert len(df) == 1
        assert list(df.columns) == ["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]
        assert df.iloc[0]["r1"] == 3  # 已排序
        assert df.iloc[0]["blue"] == 7

    def test_red_balls_sorted(self):
        """红球应该升序排列。"""
        records = [
            {
                "issue": "2024001",
                "date": "2024-01-02",
                "red_balls": [31, 3, 18, 11, 28, 22],  # 乱序
                "blue_ball": 5,
            }
        ]
        df = _to_dataframe(records)
        vals = [df.iloc[0][f"r{i}"] for i in range(1, 7)]
        assert vals == sorted(vals)


class TestDataManagerQualityCheck:
    def _make_df(self) -> pd.DataFrame:
        import random
        rows = []
        for i in range(50):
            reds = sorted(random.sample(range(1, 34), 6))
            rows.append({
                "issue": str(2000 + i),
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 3),
                "r1": reds[0], "r2": reds[1], "r3": reds[2],
                "r4": reds[3], "r5": reds[4], "r6": reds[5],
                "blue": random.randint(1, 16),
            })
        df = pd.DataFrame(rows)
        df["date"] = df["date"].dt.date
        return df

    def test_valid_data_passes(self):
        mgr = DataManager()
        df = self._make_df()
        report = mgr.quality_check(df)
        assert report["status"] in ("OK", "WARNING")
        assert report["total_periods"] == 50

    def test_empty_df_returns_error(self):
        mgr = DataManager()
        report = mgr.quality_check(pd.DataFrame())
        assert report["status"] == "ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
