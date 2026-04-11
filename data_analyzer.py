"""
双色球预测系统 - 数据分析模块
==============================
实现频率、遗漏、热冷、分布、相关性、周期性等统计分析。
免责声明：本模块仅用于概率统计研究和学术学习。
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RED_COLS = ["r1", "r2", "r3", "r4", "r5", "r6"]


# ════════════════════════════════════════════════════════
# 基础辅助
# ════════════════════════════════════════════════════════

def _red_series(df: pd.DataFrame) -> pd.Series:
    """将 6 列红球展平为单列 Series。"""
    return df[RED_COLS].values.flatten()


def _all_red_numbers() -> np.ndarray:
    return np.arange(1, 34)


def _all_blue_numbers() -> np.ndarray:
    return np.arange(1, 17)


# ════════════════════════════════════════════════════════
# 1. 频率分析
# ════════════════════════════════════════════════════════

def frequency_analysis(df: pd.DataFrame) -> dict:
    """
    计算红球和蓝球的历史出现频率。

    Parameters
    ----------
    df : DataFrame  标准历史数据

    Returns
    -------
    dict
        red_freq  : Series[int]  红球出现次数 (index=号码)
        red_rate  : Series[float] 红球出现频率
        blue_freq : Series[int]  蓝球出现次数
        blue_rate : Series[float] 蓝球出现频率
        expected_red_freq : float  理论期望频率（均匀分布）
    """
    n = len(df)
    red_series = _red_series(df)
    red_freq = pd.Series(red_series, dtype=int).value_counts().reindex(
        _all_red_numbers(), fill_value=0
    ).sort_index()

    blue_freq = df["blue"].value_counts().reindex(
        _all_blue_numbers(), fill_value=0
    ).sort_index()

    # 理论期望：红球 6/33，蓝球 1/16
    expected_red = n * 6 / 33
    expected_blue = n / 16

    # 卡方检验（频率是否均匀）
    chi2_red, p_red = stats.chisquare(red_freq.values)
    chi2_blue, p_blue = stats.chisquare(blue_freq.values)

    return {
        "red_freq": red_freq,
        "red_rate": red_freq / (n * 6),
        "blue_freq": blue_freq,
        "blue_rate": blue_freq / n,
        "expected_red_freq": expected_red,
        "expected_blue_freq": expected_blue,
        "chi2_red": chi2_red,
        "p_value_red": p_red,         # < 0.05 表示分布不均匀（有统计显著性）
        "chi2_blue": chi2_blue,
        "p_value_blue": p_blue,
        "total_periods": n,
    }


# ════════════════════════════════════════════════════════
# 2. 遗漏分析
# ════════════════════════════════════════════════════════

def missing_analysis(df: pd.DataFrame) -> dict:
    """
    计算每个号码的当前遗漏、历史最大遗漏和平均遗漏。

    遗漏：上次出现后至今未出现的期数。

    Returns
    -------
    dict with keys:
        red_current_missing  : Series[int]  红球当前遗漏期数
        red_max_missing      : Series[int]  红球历史最大遗漏
        red_avg_missing      : Series[float] 红球平均遗漏间距
        blue_current_missing : Series[int]
        blue_max_missing     : Series[int]
        blue_avg_missing     : Series[float]
    """
    n = len(df)

    def _compute_missing(balls: pd.DataFrame, nums: np.ndarray, cols: list[str]):
        current_miss = {}
        max_miss = {}
        avg_miss = {}

        for num in nums:
            # 每期该号码是否出现（布尔数组）
            if len(cols) > 1:
                appeared = balls[cols].isin([num]).any(axis=1).values
            else:
                appeared = (balls[cols[0]] == num).values

            # 当前遗漏：从最后一次出现到末尾
            last_idx = np.where(appeared)[0]
            if len(last_idx) == 0:
                current_miss[num] = n           # 从未出现
                max_miss[num] = n
                avg_miss[num] = float(n)
            else:
                current_miss[num] = n - last_idx[-1] - 1

                # 计算所有间隔
                gaps = []
                prev = -1
                for idx in last_idx:
                    if prev >= 0:
                        gaps.append(idx - prev - 1)
                    prev = idx
                # 末尾到最后一次出现的间隔
                gaps.append(n - last_idx[-1] - 1)

                max_miss[num] = max(gaps) if gaps else 0
                avg_miss[num] = float(np.mean(gaps)) if gaps else 0.0

        return (
            pd.Series(current_miss, name="current").sort_index(),
            pd.Series(max_miss, name="max").sort_index(),
            pd.Series(avg_miss, name="avg").sort_index(),
        )

    red_cur, red_max, red_avg = _compute_missing(df, _all_red_numbers(), RED_COLS)
    blue_cur, blue_max, blue_avg = _compute_missing(df, _all_blue_numbers(), ["blue"])

    return {
        "red_current_missing": red_cur,
        "red_max_missing": red_max,
        "red_avg_missing": red_avg,
        "blue_current_missing": blue_cur,
        "blue_max_missing": blue_max,
        "blue_avg_missing": blue_avg,
    }


# ════════════════════════════════════════════════════════
# 3. 热冷分析（Z-score）
# ════════════════════════════════════════════════════════

def hot_cold_analysis(df: pd.DataFrame, window: int = 30) -> dict:
    """
    基于最近 window 期的出现频率计算 Z-score，评估号码热度。

    Z-score > 1.0  → 热号
    Z-score < -1.0 → 冷号
    其余           → 温号

    Returns
    -------
    dict with:
        red_zscore  : Series[float]  红球 Z-score
        blue_zscore : Series[float]  蓝球 Z-score
        red_hot     : list[int]  热号
        red_cold    : list[int]  冷号
        blue_hot    : list[int]
        blue_cold   : list[int]
    """
    recent = df.tail(window)

    red_freq_recent = pd.Series(_red_series(recent), dtype=int).value_counts().reindex(
        _all_red_numbers(), fill_value=0
    ).sort_index()

    blue_freq_recent = recent["blue"].value_counts().reindex(
        _all_blue_numbers(), fill_value=0
    ).sort_index()

    def _zscore(freq: pd.Series) -> pd.Series:
        mu = freq.mean()
        sigma = freq.std()
        if sigma == 0:
            return pd.Series(0.0, index=freq.index)
        return (freq - mu) / sigma

    red_z = _zscore(red_freq_recent)
    blue_z = _zscore(blue_freq_recent)

    return {
        "red_zscore": red_z,
        "blue_zscore": blue_z,
        "red_freq_recent": red_freq_recent,
        "blue_freq_recent": blue_freq_recent,
        "red_hot": red_z[red_z > 1.0].index.tolist(),
        "red_cold": red_z[red_z < -1.0].index.tolist(),
        "blue_hot": blue_z[blue_z > 1.0].index.tolist(),
        "blue_cold": blue_z[blue_z < -1.0].index.tolist(),
        "window": window,
    }


# ════════════════════════════════════════════════════════
# 4. 分布分析
# ════════════════════════════════════════════════════════

def distribution_analysis(df: pd.DataFrame) -> dict:
    """
    分析每期红球的奇偶比、区间分布、和值、跨度、质数分布等。

    Returns
    -------
    dict with distribution statistics
    """
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    results = {}

    red = df[RED_COLS].values   # shape (n, 6)

    # ── 奇偶比 ───────────────────────────────────────────
    is_odd = red % 2 == 1
    odd_count = is_odd.sum(axis=1)           # 每期奇数个数
    results["odd_even_dist"] = pd.Series(odd_count).value_counts().sort_index()
    results["avg_odd_count"] = float(odd_count.mean())

    # ── 和值 ─────────────────────────────────────────────
    sum_vals = red.sum(axis=1)
    results["sum_stats"] = {
        "mean": float(sum_vals.mean()),
        "std": float(sum_vals.std()),
        "min": int(sum_vals.min()),
        "max": int(sum_vals.max()),
        "percentile_25": float(np.percentile(sum_vals, 25)),
        "percentile_75": float(np.percentile(sum_vals, 75)),
    }
    results["sum_dist"] = pd.Series(sum_vals).value_counts().sort_index()

    # ── 跨度（最大 - 最小）────────────────────────────────
    span = red.max(axis=1) - red.min(axis=1)
    results["span_stats"] = {
        "mean": float(span.mean()),
        "std": float(span.std()),
    }
    results["span_dist"] = pd.Series(span).value_counts().sort_index()

    # ── 区间分布（三等分：1-11, 12-22, 23-33）──────────────
    z1 = ((red >= 1) & (red <= 11)).sum(axis=1)
    z2 = ((red >= 12) & (red <= 22)).sum(axis=1)
    z3 = ((red >= 23) & (red <= 33)).sum(axis=1)
    zone_df = pd.DataFrame({"z1": z1, "z2": z2, "z3": z3})
    results["zone_dist"] = zone_df.value_counts().reset_index(name="count")
    results["zone_avg"] = {"zone1": float(z1.mean()), "zone2": float(z2.mean()), "zone3": float(z3.mean())}

    # ── 质数分布 ──────────────────────────────────────────
    is_prime = np.vectorize(lambda x: x in PRIMES)(red)
    prime_count = is_prime.sum(axis=1)
    results["prime_dist"] = pd.Series(prime_count).value_counts().sort_index()
    results["avg_prime_count"] = float(prime_count.mean())

    # ── 连号（相邻号码相差1）─────────────────────────────
    def count_consecutive(row):
        s = sorted(row)
        count = sum(1 for i in range(len(s) - 1) if s[i + 1] - s[i] == 1)
        return count

    consec = np.apply_along_axis(count_consecutive, 1, red)
    results["consecutive_dist"] = pd.Series(consec).value_counts().sort_index()
    results["avg_consecutive"] = float(consec.mean())

    # ── 重号（与上期相同的号码数）────────────────────────
    repeat_counts = []
    for i in range(1, len(df)):
        prev = set(df[RED_COLS].iloc[i - 1])
        curr = set(df[RED_COLS].iloc[i])
        repeat_counts.append(len(prev & curr))
    results["repeat_dist"] = pd.Series(repeat_counts).value_counts().sort_index()
    results["avg_repeat"] = float(np.mean(repeat_counts))

    return results


# ════════════════════════════════════════════════════════
# 5. AC 值分析
# ════════════════════════════════════════════════════════

def ac_value(numbers: list[int]) -> int:
    """
    计算号码组合的算术复杂度（AC值）。

    AC = 不同差值的个数 - (n-1)
    AC 越大表示号码分布越分散，通常认为 AC ≥ 4 为较优组合。

    Parameters
    ----------
    numbers : list[int]  6个红球号码

    Returns
    -------
    int  AC 值（0 ~ 9）
    """
    nums = sorted(numbers)
    diffs = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            diffs.add(nums[j] - nums[i])
    return len(diffs) - (len(nums) - 1)


def batch_ac_analysis(df: pd.DataFrame) -> pd.Series:
    """计算历史每期的 AC 值分布。"""
    return df[RED_COLS].apply(lambda row: ac_value(row.tolist()), axis=1)


# ════════════════════════════════════════════════════════
# 6. 周期性分析（傅里叶变换）
# ════════════════════════════════════════════════════════

def periodicity_analysis(df: pd.DataFrame, max_period: int = 80) -> dict:
    """
    使用离散傅里叶变换检测红球和蓝球出现的周期规律。

    Returns
    -------
    dict with dominant periods and power spectrum
    """
    from numpy.fft import rfft, rfftfreq

    results = {}

    # 对每个红球构建出现序列（0/1），然后做 FFT
    red_periods = {}
    n = len(df)
    freqs = rfftfreq(n)
    periods = 1.0 / (freqs + 1e-10)

    for num in _all_red_numbers():
        signal = df[RED_COLS].isin([num]).any(axis=1).astype(float).values
        power = np.abs(rfft(signal)) ** 2
        # 找到 2~max_period 范围内功率最大的周期
        mask = (periods >= 2) & (periods <= max_period)
        if mask.sum() == 0:
            continue
        peak_idx = np.argmax(power[mask])
        dominant_period = periods[mask][peak_idx]
        red_periods[num] = round(float(dominant_period), 1)

    # 蓝球
    blue_periods = {}
    for num in _all_blue_numbers():
        signal = (df["blue"] == num).astype(float).values
        power = np.abs(rfft(signal)) ** 2
        mask = (periods >= 2) & (periods <= max_period)
        if mask.sum() == 0:
            continue
        peak_idx = np.argmax(power[mask])
        dominant_period = periods[mask][peak_idx]
        blue_periods[num] = round(float(dominant_period), 1)

    results["red_dominant_periods"] = red_periods
    results["blue_dominant_periods"] = blue_periods

    # 整体和值序列的周期分析
    sum_signal = df[RED_COLS].sum(axis=1).values.astype(float)
    sum_signal -= sum_signal.mean()
    power = np.abs(rfft(sum_signal)) ** 2
    mask = (periods >= 2) & (periods <= max_period)
    if mask.sum() > 0:
        top5_idx = np.argsort(power[mask])[-5:][::-1]
        results["sum_top_periods"] = [round(float(periods[mask][i]), 1) for i in top5_idx]

    return results


# ════════════════════════════════════════════════════════
# 7. 相关性分析（共现矩阵）
# ════════════════════════════════════════════════════════

def cooccurrence_analysis(df: pd.DataFrame) -> dict:
    """
    计算红球两两共现频率矩阵，以及红蓝球联合分布。

    Returns
    -------
    dict with:
        red_cooccurrence : DataFrame (33x33) 红球共现次数矩阵
        red_blue_joint   : DataFrame (33x16) 红蓝球联合频次
    """
    n = len(df)
    red_nums = _all_red_numbers()
    blue_nums = _all_blue_numbers()

    # 红球共现矩阵
    co_matrix = pd.DataFrame(0, index=red_nums, columns=red_nums)
    for _, row in df[RED_COLS].iterrows():
        nums = row.tolist()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                co_matrix.loc[nums[i], nums[j]] += 1
                co_matrix.loc[nums[j], nums[i]] += 1

    # 归一化为共现率
    co_rate = co_matrix / n

    # 红蓝联合分布
    rb_joint = pd.DataFrame(0, index=red_nums, columns=blue_nums)
    for _, row in df[RED_COLS + ["blue"]].iterrows():
        blue = int(row["blue"])
        for col in RED_COLS:
            rb_joint.loc[int(row[col]), blue] += 1

    return {
        "red_cooccurrence": co_matrix,
        "red_cooccurrence_rate": co_rate,
        "red_blue_joint": rb_joint,
    }


# ════════════════════════════════════════════════════════
# 8. 特征工程（为机器学习准备特征）
# ════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    为机器学习模型构建特征矩阵。

    特征包括：
    - 近 N 期红球/蓝球出现频次
    - 当前遗漏期数
    - 最近期的和值、奇偶数、区间分布
    - 上期红蓝球号码（滞后特征）

    Parameters
    ----------
    df      : 历史数据
    windows : 滑动窗口列表

    Returns
    -------
    pd.DataFrame  特征矩阵（行=期次，列=特征）
    """
    if windows is None:
        windows = [10, 20, 30, 50]

    features_list = []
    n = len(df)

    # 至少需要 max(windows) 期历史才能构建特征
    start_idx = max(windows)

    for i in range(start_idx, n):
        feat = {"issue": df.iloc[i]["issue"], "date": df.iloc[i]["date"]}

        # 历史窗口内特征
        for w in windows:
            window_df = df.iloc[i - w : i]
            red_series = _red_series(window_df)

            # 红球频次特征
            red_freq = pd.Series(red_series, dtype=int).value_counts()
            for num in _all_red_numbers():
                feat[f"red_{num:02d}_freq_w{w}"] = red_freq.get(num, 0)

            # 蓝球频次
            blue_freq = window_df["blue"].value_counts()
            for num in _all_blue_numbers():
                feat[f"blue_{num:02d}_freq_w{w}"] = blue_freq.get(num, 0)

            # 和值统计
            sums = window_df[RED_COLS].sum(axis=1)
            feat[f"sum_mean_w{w}"] = float(sums.mean())
            feat[f"sum_std_w{w}"] = float(sums.std())

            # 奇数平均数
            is_odd = window_df[RED_COLS].values % 2 == 1
            feat[f"odd_mean_w{w}"] = float(is_odd.sum(axis=1).mean())

        # 当前遗漏特征（截至第 i 期）
        for num in _all_red_numbers():
            appeared = df[RED_COLS].iloc[:i].isin([num]).any(axis=1)
            last_idx_arr = np.where(appeared.values)[0]
            feat[f"red_{num:02d}_missing"] = (
                i - last_idx_arr[-1] - 1 if len(last_idx_arr) > 0 else i
            )

        for num in _all_blue_numbers():
            appeared = (df["blue"].iloc[:i] == num)
            last_idx_arr = np.where(appeared.values)[0]
            feat[f"blue_{num:02d}_missing"] = (
                i - last_idx_arr[-1] - 1 if len(last_idx_arr) > 0 else i
            )

        # 上期特征（滞后1期）
        prev = df.iloc[i - 1]
        for k, col in enumerate(RED_COLS, 1):
            feat[f"prev_r{k}"] = int(prev[col])
        feat["prev_blue"] = int(prev["blue"])
        feat[f"prev_sum"] = int(prev[RED_COLS].sum())
        feat["prev_span"] = int(prev[RED_COLS].max() - prev[RED_COLS].min())

        features_list.append(feat)

    features_df = pd.DataFrame(features_list)
    logger.info("特征矩阵构建完成：%d 行 × %d 列", len(features_df), len(features_df.columns))
    return features_df


# ════════════════════════════════════════════════════════
# 9. 马尔可夫链状态转移矩阵
# ════════════════════════════════════════════════════════

def build_markov_transition(df: pd.DataFrame) -> dict:
    """
    构建蓝球的一阶马尔可夫链转移矩阵。
    （红球因维度高，建议分区间或用组合状态）

    Returns
    -------
    dict with:
        blue_transition : DataFrame (16x16) 转移概率矩阵
        blue_stationary : Series[float]  稳态分布
    """
    blues = df["blue"].values
    n_states = 16
    trans = np.zeros((n_states, n_states), dtype=float)

    for i in range(len(blues) - 1):
        from_s = int(blues[i]) - 1
        to_s = int(blues[i + 1]) - 1
        trans[from_s, to_s] += 1

    # 归一化为概率
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    trans_prob = trans / row_sums

    # 计算稳态分布（左特征向量）
    eigenvalues, eigenvectors = np.linalg.eig(trans_prob.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.abs(eigenvectors[:, stationary_idx].real)
    stationary /= stationary.sum()

    nums = list(range(1, 17))
    return {
        "blue_transition": pd.DataFrame(trans_prob, index=nums, columns=nums),
        "blue_stationary": pd.Series(stationary, index=nums),
    }


# ════════════════════════════════════════════════════════
# 10. 可视化
# ════════════════════════════════════════════════════════

def visualize_frequency(freq_result: dict, save_path: str = None):
    """绘制红球和蓝球频率分布柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 尝试使用中文字体
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 红球频率
        ax = axes[0]
        red_freq = freq_result["red_freq"]
        expected = freq_result["expected_red_freq"]
        colors = ["#E74C3C" if v >= expected else "#F0A8A8" for v in red_freq.values]
        ax.bar(red_freq.index, red_freq.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(expected, color="#2C3E50", linestyle="--", linewidth=1.5, label=f"期望频次 {expected:.1f}")
        ax.set_title("红球历史出现频率", fontsize=14, fontweight="bold")
        ax.set_xlabel("号码")
        ax.set_ylabel("出现次数")
        ax.set_xticks(red_freq.index)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # 蓝球频率
        ax = axes[1]
        blue_freq = freq_result["blue_freq"]
        expected_b = freq_result["expected_blue_freq"]
        colors_b = ["#2980B9" if v >= expected_b else "#AED6F1" for v in blue_freq.values]
        ax.bar(blue_freq.index, blue_freq.values, color=colors_b, edgecolor="white", linewidth=0.5)
        ax.axhline(expected_b, color="#2C3E50", linestyle="--", linewidth=1.5, label=f"期望频次 {expected_b:.1f}")
        ax.set_title("蓝球历史出现频率", fontsize=14, fontweight="bold")
        ax.set_xlabel("号码")
        ax.set_ylabel("出现次数")
        ax.set_xticks(blue_freq.index)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("频率图已保存：%s", save_path)
        else:
            plt.show()
        plt.close()
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化。")


def visualize_heatmap(co_result: dict, save_path: str = None):
    """绘制红球共现热力图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams["font.sans-serif"] = ["PingFang SC", "SimHei", "Arial Unicode MS"]

        fig, ax = plt.subplots(figsize=(14, 12))
        matrix = co_result["red_cooccurrence_rate"]
        sns.heatmap(
            matrix, ax=ax,
            cmap="YlOrRd", linewidths=0.1,
            xticklabels=matrix.columns,
            yticklabels=matrix.index,
            annot=False,
        )
        ax.set_title("红球共现率热力图", fontsize=14, fontweight="bold")
        ax.set_xlabel("号码")
        ax.set_ylabel("号码")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("热力图已保存：%s", save_path)
        else:
            plt.show()
        plt.close()
    except ImportError:
        logger.warning("matplotlib/seaborn 未安装，跳过可视化。")


# ════════════════════════════════════════════════════════
# 综合分析入口
# ════════════════════════════════════════════════════════

class DataAnalyzer:
    """
    双色球数据分析器，整合所有统计分析功能。
    """

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("数据集为空，无法进行分析。")
        self.df = df
        self._cache: dict = {}
        logger.info("DataAnalyzer 初始化，共 %d 期数据。", len(df))

    def _cached(self, key: str, func, *args, **kwargs):
        if key not in self._cache:
            self._cache[key] = func(*args, **kwargs)
        return self._cache[key]

    def frequency(self) -> dict:
        return self._cached("frequency", frequency_analysis, self.df)

    def missing(self) -> dict:
        return self._cached("missing", missing_analysis, self.df)

    def hot_cold(self, window: int = 30) -> dict:
        return self._cached(f"hot_cold_{window}", hot_cold_analysis, self.df, window)

    def distribution(self) -> dict:
        return self._cached("distribution", distribution_analysis, self.df)

    def ac_values(self) -> pd.Series:
        return self._cached("ac_values", batch_ac_analysis, self.df)

    def periodicity(self) -> dict:
        return self._cached("periodicity", periodicity_analysis, self.df)

    def cooccurrence(self) -> dict:
        return self._cached("cooccurrence", cooccurrence_analysis, self.df)

    def markov(self) -> dict:
        return self._cached("markov", build_markov_transition, self.df)

    def features(self, windows: list[int] = None) -> pd.DataFrame:
        key = f"features_{windows}"
        return self._cached(key, build_features, self.df, windows)

    def full_report(self) -> str:
        """生成文本格式的综合分析报告。"""
        lines = []
        lines.append("=" * 60)
        lines.append("双色球统计分析报告")
        lines.append(f"数据期数：{len(self.df)}  |  "
                     f"期号范围：{self.df['issue'].min()} ~ {self.df['issue'].max()}")
        lines.append("=" * 60)

        # 频率
        freq = self.frequency()
        top5_red = freq["red_freq"].nlargest(5)
        low5_red = freq["red_freq"].nsmallest(5)
        lines.append("\n【频率分析】")
        lines.append(f"  红球出现最多：{top5_red.to_dict()}")
        lines.append(f"  红球出现最少：{low5_red.to_dict()}")
        lines.append(f"  蓝球出现最多：{freq['blue_freq'].idxmax()} "
                     f"({freq['blue_freq'].max()}次)")
        lines.append(f"  红球卡方检验 p={freq['p_value_red']:.4f} "
                     f"({'分布不均匀' if freq['p_value_red'] < 0.05 else '分布较均匀'})")

        # 遗漏
        miss = self.missing()
        lines.append("\n【遗漏分析】")
        red_miss_top5 = miss["red_current_missing"].nlargest(5)
        lines.append(f"  当前遗漏最长红球：{red_miss_top5.to_dict()}")
        lines.append(f"  当前遗漏最长蓝球：{miss['blue_current_missing'].idxmax()} "
                     f"({miss['blue_current_missing'].max()}期)")

        # 热冷
        hc = self.hot_cold()
        lines.append("\n【热冷分析（近30期）】")
        lines.append(f"  热号红球：{hc['red_hot']}")
        lines.append(f"  冷号红球：{hc['red_cold']}")
        lines.append(f"  热号蓝球：{hc['blue_hot']}")
        lines.append(f"  冷号蓝球：{hc['blue_cold']}")

        # 分布
        dist = self.distribution()
        lines.append("\n【分布分析】")
        lines.append(f"  平均和值：{dist['sum_stats']['mean']:.1f} "
                     f"(±{dist['sum_stats']['std']:.1f})")
        lines.append(f"  平均跨度：{dist['span_stats']['mean']:.1f}")
        lines.append(f"  平均奇数个数：{dist['avg_odd_count']:.2f}")
        lines.append(f"  平均质数个数：{dist['avg_prime_count']:.2f}")
        lines.append(f"  平均连号对数：{dist['avg_consecutive']:.2f}")

        # AC值
        ac = self.ac_values()
        lines.append(f"\n  AC值分布：均值={ac.mean():.2f}, "
                     f"≥4的比例={( ac >= 4).mean() * 100:.1f}%")

        lines.append("\n" + "=" * 60)
        lines.append("⚠ 注：统计规律不能预测随机事件，彩票购买需理性。")
        lines.append("=" * 60)

        return "\n".join(lines)
