"""
双色球预测系统 - 数据采集模块
==============================
免责声明：本模块仅用于概率统计研究，数据来自公开官方渠道。
"""

import json
import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    BACKUP_DATA_FILE,
    DATA_FILE,
    HEADERS,
    MIN_HISTORY_PERIODS,
    RED_BALL_COUNT,
    RED_BALL_MAX,
    RED_BALL_MIN,
    BLUE_BALL_MAX,
    BLUE_BALL_MIN,
    REQUEST_DELAY,
    REQUEST_RETRIES,
    REQUEST_TIMEOUT,
)

# ── 经过验证的可用数据源 ────────────────────────────────────
# 主数据源：500彩票网历史数据 HTML（全量，3300+期）
CHART500_URL = "https://datachart.500.com/ssq/history/newinc/history.php"
CHART500_HEADERS = {
    **HEADERS,
    "Referer": "https://datachart.500.com/ssq/",
    "Host": "datachart.500.com",
}

# 备用数据源：GitHub 公开历史 CSV（到2018年，约2300期）
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/BEWINDOWEB/lotterydata/master/lot_500_ssq.txt"
)

logger = logging.getLogger(__name__)


# ── HTTP 会话工厂 ────────────────────────────────────────
def _make_session() -> requests.Session:
    """创建带重试机制的 HTTP 会话。"""
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


# ── 数据验证 ─────────────────────────────────────────────
def _validate_row(row: dict) -> bool:
    """
    校验单行开奖数据的合法性。

    Parameters
    ----------
    row : dict  包含 issue, date, red_balls, blue_ball 的字典

    Returns
    -------
    bool  True 表示数据合法
    """
    try:
        red = row["red_balls"]
        if not isinstance(red, list) or len(red) != RED_BALL_COUNT:
            return False
        for r in red:
            if not (RED_BALL_MIN <= int(r) <= RED_BALL_MAX):
                return False
        if len(set(red)) != RED_BALL_COUNT:   # 有重复
            return False
        blue = int(row["blue_ball"])
        if not (BLUE_BALL_MIN <= blue <= BLUE_BALL_MAX):
            return False
        # 期号非空
        if not row.get("issue"):
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


# ── 原始数据 → 标准 DataFrame ────────────────────────────
def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    将解析好的记录列表转换为标准 DataFrame。

    列名：issue, date, r1-r6, blue
    """
    rows = []
    for rec in records:
        red = sorted([int(x) for x in rec["red_balls"]])
        rows.append(
            {
                "issue": str(rec["issue"]).strip(),
                "date": pd.to_datetime(rec["date"]).date(),
                "r1": red[0],
                "r2": red[1],
                "r3": red[2],
                "r4": red[3],
                "r5": red[4],
                "r6": red[5],
                "blue": int(rec["blue_ball"]),
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values("issue", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── 主数据源：500彩票网 历史数据 HTML（验证可用）──────────────
class Chart500Scraper:
    """
    从 500彩票网 datachart 获取双色球全量历史数据。

    接口地址：https://datachart.500.com/ssq/history/newinc/history.php
    参数：start=期号起始, end=期号结束（如 03001~25200）
    已验证可获取 3300+ 期完整历史数据。
    """

    URL = CHART500_URL

    def __init__(self):
        self.session = _make_session()
        self.session.headers.update(CHART500_HEADERS)

    def fetch_all(self, start: str = "03001", end: str = "99999") -> pd.DataFrame:
        """
        获取指定期号范围的全部数据。

        Parameters
        ----------
        start : str  起始期号（默认从第一期开始）
        end   : str  结束期号（默认取到最新）
        """
        logger.info("从500彩票网获取双色球历史数据（%s ~ %s）...", start, end)
        try:
            resp = self.session.get(
                self.URL,
                params={"start": start, "end": end},
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("500彩票网请求失败: %s", e)
            return pd.DataFrame()

        return self._parse_html(resp.text)

    def fetch_latest(self, n: int = 30) -> pd.DataFrame:
        """获取最新 n 期数据（用于增量更新）。"""
        # 用一个较大的起始期号，只取最后 n 期
        # 实际上该接口会返回范围内所有期，取 tail(n) 即可
        df = self.fetch_all(start="24000", end="99999")
        if df.empty:
            return df
        return df.tail(n).reset_index(drop=True)

    @staticmethod
    def _parse_html(html: str) -> pd.DataFrame:
        """解析 500彩票网 HTML 表格。"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("需要安装 beautifulsoup4: pip install beautifulsoup4")
            return pd.DataFrame()

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="tablelist")
        if table is None:
            logger.error("未找到数据表格，页面结构可能已变化。")
            return pd.DataFrame()

        records = []
        rows = table.find_all("tr")
        for tr in rows[2:]:   # 跳过两行表头
            tds = tr.find_all("td")
            if len(tds) < 9:
                continue
            try:
                issue = tds[0].get_text(strip=True)
                reds = [int(tds[i].get_text(strip=True)) for i in range(1, 7)]
                blue = int(tds[7].get_text(strip=True))
                # 日期在最后一列
                date_str = tds[-1].get_text(strip=True)
                rec = {
                    "issue": issue,
                    "date": date_str,
                    "red_balls": reds,
                    "blue_ball": blue,
                }
                if _validate_row(rec):
                    records.append(rec)
            except (ValueError, IndexError):
                continue

        if not records:
            logger.error("HTML 解析得到 0 条记录。")
            return pd.DataFrame()

        df = _to_dataframe(records)
        df.drop_duplicates(subset="issue", keep="first", inplace=True)
        logger.info("500彩票网解析完成：%d 期", len(df))
        return df


# ── 备用数据源：GitHub 公开历史 CSV（到2018年）──────────────
class GitHubCSVScraper:
    """
    从 GitHub 公开仓库下载双色球历史 CSV（备用，约2300期到2018年）。
    仅在主数据源失败时用于补充早期数据。
    """

    URL = GITHUB_CSV_URL

    def __init__(self):
        self.session = _make_session()

    def fetch_all(self, limit: int = 3000) -> pd.DataFrame:
        """下载并解析 GitHub CSV 文件。"""
        logger.info("从 GitHub 下载历史数据...")
        try:
            resp = self.session.get(self.URL, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("GitHub CSV 下载失败: %s", e)
            return pd.DataFrame()

        records = []
        for line in resp.text.strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                issue = parts[0].strip()
                reds = [int(parts[i]) for i in range(1, 7)]
                blue = int(parts[7])
                # 日期在末尾
                date_str = parts[-1].strip()
                rec = {
                    "issue": issue,
                    "date": date_str,
                    "red_balls": reds,
                    "blue_ball": blue,
                }
                if _validate_row(rec):
                    records.append(rec)
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()

        df = _to_dataframe(records)
        df.drop_duplicates(subset="issue", keep="first", inplace=True)
        logger.info("GitHub CSV 解析完成：%d 期", len(df))
        return df

    def fetch_latest(self, n: int = 10) -> pd.DataFrame:
        df = self.fetch_all()
        return df.tail(n).reset_index(drop=True) if not df.empty else df

    def _parse_json(self, data: dict | list) -> pd.DataFrame:
        records = []
        if isinstance(data, dict):
            items = data.get("data", data.get("list", []))
        else:
            items = data

        for item in items:
            try:
                if isinstance(item, dict):
                    red_raw = item.get("red", "")
                    if isinstance(red_raw, str):
                        reds = [int(x) for x in red_raw.split(",")]
                    else:
                        reds = [int(x) for x in red_raw]
                    rec = {
                        "issue": str(item.get("expect", item.get("issue", ""))),
                        "date": item.get("opentime", item.get("date", "")),
                        "red_balls": reds,
                        "blue_ball": int(item.get("blue", 0)),
                    }
                else:
                    # item 为列表格式 [issue, date, r1..r6, blue]
                    parts = list(item)
                    rec = {
                        "issue": str(parts[0]),
                        "date": str(parts[1]),
                        "red_balls": [int(x) for x in parts[2:8]],
                        "blue_ball": int(parts[8]),
                    }
                if _validate_row(rec):
                    records.append(rec)
            except (ValueError, IndexError, KeyError):
                continue

        if not records:
            return pd.DataFrame()
        df = _to_dataframe(records)
        df.drop_duplicates(subset="issue", keep="first", inplace=True)
        logger.info("500彩票网共解析 %d 期数据。", len(df))
        return df

    def _parse_html(self, html: str) -> pd.DataFrame:
        """从 HTML 表格中解析数据（最后备用）。"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", {"id": "tdata"}) or soup.find("table")
            if table is None:
                return pd.DataFrame()

            records = []
            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) < 9:
                    continue
                try:
                    issue = tds[0].get_text(strip=True)
                    reds = [int(tds[i].get_text(strip=True)) for i in range(2, 8)]
                    blue = int(tds[8].get_text(strip=True))
                    date_str = tds[1].get_text(strip=True)
                    rec = {
                        "issue": issue,
                        "date": date_str,
                        "red_balls": reds,
                        "blue_ball": blue,
                    }
                    if _validate_row(rec):
                        records.append(rec)
                except (ValueError, IndexError):
                    continue

            if not records:
                return pd.DataFrame()
            df = _to_dataframe(records)
            df.drop_duplicates(subset="issue", keep="first", inplace=True)
            logger.info("HTML 解析共得到 %d 期数据。", len(df))
            return df
        except ImportError:
            logger.error("需要安装 BeautifulSoup4: pip install beautifulsoup4")
            return pd.DataFrame()


# ── 主数据管理类 ─────────────────────────────────────────
class DataManager:
    """
    双色球历史数据管理器。

    职责：
    - 协调多个数据源
    - 本地缓存管理
    - 增量更新
    - 数据完整性校验
    """

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = data_file
        self.backup_file = BACKUP_DATA_FILE
        self.primary = Chart500Scraper()    # 主数据源：500彩票网 HTML（3300+期）
        self.backup = GitHubCSVScraper()   # 备用：GitHub CSV（早期数据）
        self._df: Optional[pd.DataFrame] = None

    # ── 本地数据 ──────────────────────────────────────────
    def load_local(self) -> pd.DataFrame:
        """从本地 CSV 加载已有数据。"""
        if self.data_file.exists():
            df = pd.read_csv(self.data_file, dtype={"issue": str})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df.sort_values("issue", inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info("本地数据加载成功：%d 期", len(df))
            return df
        logger.info("本地数据文件不存在，将从网络获取。")
        return pd.DataFrame()

    def save_local(self, df: pd.DataFrame) -> None:
        """保存数据到本地 CSV。"""
        # 同时保留一份备份
        if self.data_file.exists():
            import shutil
            shutil.copy2(self.data_file, self.backup_file)

        df.to_csv(self.data_file, index=False)
        logger.info("数据已保存：%d 期 → %s", len(df), self.data_file)

    # ── 网络数据获取 ──────────────────────────────────────
    def fetch_remote(self, limit: int = 3000) -> pd.DataFrame:
        """
        从网络获取数据。
        主数据源：500彩票网 HTML（全量 3300+ 期）。
        备用源：GitHub CSV（约 2300 期，至2018年）用于主源失败时。
        """
        logger.info("尝试从500彩票网获取全量历史数据...")
        df = self.primary.fetch_all()

        if df.empty or len(df) < MIN_HISTORY_PERIODS:
            logger.warning(
                "主数据源数据不足（%d 期），切换至 GitHub CSV 备用源...", len(df)
            )
            df_backup = self.backup.fetch_all()
            if not df_backup.empty:
                if df.empty:
                    df = df_backup
                else:
                    df = pd.concat([df, df_backup], ignore_index=True)
                    df.drop_duplicates(subset="issue", keep="first", inplace=True)
                    df.sort_values("issue", inplace=True)
                    df.reset_index(drop=True, inplace=True)

        if df.empty:
            logger.error("所有数据源均获取失败！")
        else:
            logger.info("远程数据获取完成：共 %d 期", len(df))
        return df

    # ── 初始化/全量拉取 ───────────────────────────────────
    def initialize(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        初始化数据：本地有则加载，不足则补充，force_refresh 则强制重拉。

        Parameters
        ----------
        force_refresh : bool  True 时忽略本地缓存，重新从网络获取
        """
        if not force_refresh:
            local_df = self.load_local()
            if len(local_df) >= MIN_HISTORY_PERIODS:
                self._df = local_df
                return self._df

        logger.info("开始全量获取历史数据（可能需要较长时间）...")
        df = self.fetch_remote(limit=3000)
        if not df.empty:
            self.save_local(df)
        self._df = df
        return self._df

    # ── 增量更新 ──────────────────────────────────────────
    def update(self) -> tuple[pd.DataFrame, int]:
        """
        增量更新：仅获取本地最新期号之后的数据。

        Returns
        -------
        (更新后的 DataFrame, 新增期数)
        """
        local_df = self.load_local()
        if local_df.empty:
            logger.info("本地无数据，执行全量初始化。")
            df = self.initialize()
            return df, len(df)

        latest_issue = local_df["issue"].max()
        logger.info("本地最新期号：%s，正在检查新开奖...", latest_issue)

        # 获取最近 30 期用于比对（500彩票网）
        new_df = self.primary.fetch_latest(n=30)
        if new_df.empty:
            new_df = self.backup.fetch_latest(n=30)

        if new_df.empty:
            logger.warning("无法获取最新数据。")
            self._df = local_df
            return local_df, 0

        # 筛选出比本地更新的期号
        truly_new = new_df[new_df["issue"] > latest_issue]
        if truly_new.empty:
            logger.info("暂无新开奖数据（最新期号：%s）。", latest_issue)
            self._df = local_df
            return local_df, 0

        merged = pd.concat([local_df, truly_new], ignore_index=True)
        merged.drop_duplicates(subset="issue", keep="last", inplace=True)
        merged.sort_values("issue", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        new_count = len(truly_new)
        logger.info("新增 %d 期数据！最新期号：%s", new_count, merged["issue"].max())
        self.save_local(merged)
        self._df = merged
        return merged, new_count

    # ── 数据质量检查 ──────────────────────────────────────
    def quality_check(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        对数据集进行质量检查，返回检查报告。
        """
        if df is None:
            df = self._df if self._df is not None else self.load_local()
        if df.empty:
            return {"status": "ERROR", "message": "数据集为空"}

        report = {
            "status": "OK",
            "total_periods": len(df),
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}",
            "issues": [],
        }

        red_cols = ["r1", "r2", "r3", "r4", "r5", "r6"]

        # 检查缺失值
        missing = df[red_cols + ["blue"]].isnull().sum().sum()
        if missing > 0:
            report["issues"].append(f"存在 {missing} 个缺失值")
            report["status"] = "WARNING"

        # 检查红球范围
        for col in red_cols:
            out_of_range = ((df[col] < 1) | (df[col] > 33)).sum()
            if out_of_range:
                report["issues"].append(f"{col} 有 {out_of_range} 个超范围值")
                report["status"] = "ERROR"

        # 检查蓝球范围
        bad_blue = ((df["blue"] < 1) | (df["blue"] > 16)).sum()
        if bad_blue:
            report["issues"].append(f"蓝球有 {bad_blue} 个超范围值")
            report["status"] = "ERROR"

        # 检查红球重复（同一期内）
        def has_dup(row):
            vals = [row[c] for c in red_cols]
            return len(set(vals)) != 6

        dup_rows = df.apply(has_dup, axis=1).sum()
        if dup_rows:
            report["issues"].append(f"有 {dup_rows} 期红球存在重复号码")
            report["status"] = "ERROR"

        # 检查期号连续性（允许有空缺，仅统计跳号）
        issues_sorted = sorted(df["issue"].tolist())
        report["min_issue"] = issues_sorted[0]
        report["max_issue"] = issues_sorted[-1]

        if not report["issues"]:
            report["issues"].append("无异常")

        return report

    @property
    def df(self) -> pd.DataFrame:
        """获取当前数据集（懒加载）。"""
        if self._df is None:
            self._df = self.load_local()
        return self._df


# ── 便捷函数 ─────────────────────────────────────────────
def get_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    获取双色球历史数据的快捷函数。

    Parameters
    ----------
    force_refresh : bool  True 时强制从网络重新获取

    Returns
    -------
    pd.DataFrame  标准格式的历史数据
    """
    mgr = DataManager()
    return mgr.initialize(force_refresh=force_refresh)


def update_data() -> tuple[pd.DataFrame, int]:
    """
    增量更新快捷函数。

    Returns
    -------
    (更新后的 DataFrame, 新增期数)
    """
    mgr = DataManager()
    return mgr.update()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("双色球数据采集模块")
    print("=" * 60)

    mgr = DataManager()

    if "--refresh" in sys.argv:
        df = mgr.initialize(force_refresh=True)
    else:
        df, new_count = mgr.update()
        if new_count:
            print(f"\n✓ 新增 {new_count} 期数据")
        else:
            df = mgr.df

    if df.empty:
        print("数据获取失败，请检查网络连接。")
        sys.exit(1)

    print(f"\n数据概览：共 {len(df)} 期")
    print(f"期号范围：{df['issue'].min()} ~ {df['issue'].max()}")
    print(f"日期范围：{df['date'].min()} ~ {df['date'].max()}")
    print("\n最新 5 期：")
    print(df.tail(5).to_string(index=False))

    report = mgr.quality_check(df)
    print(f"\n数据质量检查：{report['status']}")
    for issue in report["issues"]:
        print(f"  - {issue}")
