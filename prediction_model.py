"""
双色球预测系统 - 预测模型模块
==============================
集成统计模型 + 机器学习模型 + 集成策略。
免责声明：本模块仅用于概率统计研究，彩票具有高度随机性。
"""

import itertools
import logging
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RED_COLS = ["r1", "r2", "r3", "r4", "r5", "r6"]
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _red_score(numbers: list[int], freq: pd.Series, missing: pd.Series,
               zscore: pd.Series) -> float:
    """
    对一组红球号码打综合分数。

    综合考虑：出现频率、当前遗漏、热度 Z-score。
    """
    freq_score = sum(freq.get(n, 0) for n in numbers)
    miss_score = sum(min(missing.get(n, 0), 30) for n in numbers)  # 遗漏越大分越高（但有上限）
    hot_score = sum(zscore.get(n, 0) for n in numbers)
    return freq_score * 0.4 + miss_score * 0.3 + hot_score * 0.3


def _describe_combination(numbers: list[int], blue: int) -> dict:
    """生成号码组合的特征描述。"""
    nums = sorted(numbers)
    odd = sum(1 for n in nums if n % 2 == 1)
    primes = sum(1 for n in nums if n in PRIMES)
    z1 = sum(1 for n in nums if 1 <= n <= 11)
    z2 = sum(1 for n in nums if 12 <= n <= 22)
    z3 = sum(1 for n in nums if 23 <= n <= 33)
    total = sum(nums)
    span = nums[-1] - nums[0]
    consecutive = sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)

    # AC 值
    diffs = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            diffs.add(nums[j] - nums[i])
    ac = len(diffs) - (len(nums) - 1)

    return {
        "numbers": nums,
        "blue": blue,
        "奇偶比": f"{odd}:{6-odd}",
        "区间分布": f"{z1}-{z2}-{z3}",
        "和值": total,
        "跨度": span,
        "连号对数": consecutive,
        "质数个数": primes,
        "AC值": ac,
    }


# ════════════════════════════════════════════════════════
# 1. 贝叶斯概率模型
# ════════════════════════════════════════════════════════

class BayesianModel:
    """
    基于贝叶斯概率的号码推荐。

    使用 Dirichlet-Multinomial 共轭先验更新频率。
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float  Dirichlet 先验参数（平滑系数）
        """
        self.alpha = alpha
        self.red_posterior: Optional[pd.Series] = None
        self.blue_posterior: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "BayesianModel":
        """根据历史数据更新后验分布。"""
        n = len(df)

        # 红球频次
        red_all = df[RED_COLS].values.flatten()
        red_counts = pd.Series(red_all, dtype=int).value_counts().reindex(
            range(1, 34), fill_value=0
        )
        # 后验 = (观测次数 + alpha) / (总次数 + alpha * K)
        K_red = 33
        self.red_posterior = (red_counts + self.alpha) / (red_counts.sum() + self.alpha * K_red)

        # 蓝球
        K_blue = 16
        blue_counts = df["blue"].value_counts().reindex(range(1, 17), fill_value=0)
        self.blue_posterior = (blue_counts + self.alpha) / (blue_counts.sum() + self.alpha * K_blue)

        logger.debug("BayesianModel 训练完成，%d 期数据。", n)
        return self

    def predict_red(self, n_balls: int = 6, n_groups: int = 5) -> list[list[int]]:
        """根据后验概率抽样 n_groups 组红球。"""
        if self.red_posterior is None:
            raise RuntimeError("请先调用 fit()")
        groups = []
        probs = self.red_posterior.values / self.red_posterior.values.sum()
        nums = self.red_posterior.index.tolist()
        for _ in range(n_groups * 10):  # 多采样，过滤不合格组合
            selected = np.random.choice(nums, size=n_balls, replace=False, p=probs)
            groups.append(sorted(selected.tolist()))
            if len(groups) >= n_groups:
                break
        return groups[:n_groups]

    def predict_blue(self) -> int:
        """返回概率最高的蓝球。"""
        if self.blue_posterior is None:
            raise RuntimeError("请先调用 fit()")
        return int(self.blue_posterior.idxmax())

    def predict_blue_top(self, n: int = 3) -> list[int]:
        """返回概率最高的 n 个蓝球。"""
        if self.blue_posterior is None:
            raise RuntimeError("请先调用 fit()")
        return self.blue_posterior.nlargest(n).index.tolist()


# ════════════════════════════════════════════════════════
# 2. 马尔可夫链预测
# ════════════════════════════════════════════════════════

class MarkovModel:
    """
    基于马尔可夫链的蓝球预测和红球区间预测。
    """

    def __init__(self):
        self.blue_transition: Optional[pd.DataFrame] = None
        self.last_blue: Optional[int] = None

    def fit(self, df: pd.DataFrame) -> "MarkovModel":
        """构建转移矩阵。"""
        blues = df["blue"].values
        n_states = 16
        trans = np.zeros((n_states, n_states), dtype=float)
        for i in range(len(blues) - 1):
            trans[int(blues[i]) - 1, int(blues[i + 1]) - 1] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans / row_sums
        nums = list(range(1, 17))
        self.blue_transition = pd.DataFrame(trans_prob, index=nums, columns=nums)
        self.last_blue = int(blues[-1])
        logger.debug("MarkovModel 训练完成。")
        return self

    def predict_blue(self) -> int:
        """预测下一期最可能的蓝球。"""
        if self.blue_transition is None or self.last_blue is None:
            raise RuntimeError("请先调用 fit()")
        probs = self.blue_transition.loc[self.last_blue]
        return int(probs.idxmax())

    def predict_blue_proba(self) -> pd.Series:
        """返回蓝球转移概率分布。"""
        if self.blue_transition is None or self.last_blue is None:
            raise RuntimeError("请先调用 fit()")
        return self.blue_transition.loc[self.last_blue]


# ════════════════════════════════════════════════════════
# 3. ARIMA 时间序列预测（和值预测）
# ════════════════════════════════════════════════════════

class ARIMAModel:
    """
    使用 ARIMA 预测红球和值的区间。
    （和值本身是时序信号，用于约束候选号码范围）
    """

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order = order
        self.fitted = None
        self.last_sum: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "ARIMAModel":
        """训练 ARIMA 模型。"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            sum_series = df[RED_COLS].sum(axis=1).astype(float)
            self.last_sum = float(sum_series.iloc[-1])
            model = ARIMA(sum_series.values, order=self.order)
            self.fitted = model.fit()
            logger.debug("ARIMA 模型训练完成，order=%s", self.order)
        except Exception as e:
            logger.warning("ARIMA 训练失败: %s，将使用均值替代。", e)
            self.fitted = None
            self.last_sum = float(df[RED_COLS].sum(axis=1).mean())
        return self

    def predict_sum(self, steps: int = 1) -> float:
        """预测未来 steps 期的红球和值。"""
        if self.fitted is None:
            return self.last_sum or 100.0
        try:
            forecast = self.fitted.forecast(steps=steps)
            return float(forecast[steps - 1])
        except Exception:
            return self.last_sum or 100.0

    def predict_sum_range(self, steps: int = 1, sigma: float = 15.0) -> tuple[float, float]:
        """返回和值预测的置信区间。"""
        pred = self.predict_sum(steps)
        return max(21.0, pred - sigma), min(183.0, pred + sigma)


# ════════════════════════════════════════════════════════
# 4. 随机森林模型
# ════════════════════════════════════════════════════════

class RandomForestModel:
    """
    使用随机森林预测每个号码的出现概率。

    为每个红球（1-33）和蓝球（1-16）训练独立的二分类器。
    """

    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators
        self.red_models: dict = {}
        self.blue_models: dict = {}
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame) -> "RandomForestModel":
        """
        Parameters
        ----------
        features_df : 特征矩阵（由 data_analyzer.build_features 生成）
        target_df   : 对应的目标值（历史数据中的真实开奖号码）
        """
        # 提取数值特征列
        self.feature_cols = [
            c for c in features_df.columns
            if c not in ("issue", "date") and features_df[c].dtype in [np.float64, np.int64]
        ]
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # 训练红球分类器
        for num in range(1, 34):
            y = target_df[RED_COLS].isin([num]).any(axis=1).astype(int).values
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_scaled, y)
            self.red_models[num] = clf

        # 训练蓝球分类器
        for num in range(1, 17):
            y = (target_df["blue"] == num).astype(int).values
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_scaled, y)
            self.blue_models[num] = clf

        logger.info("RandomForest 训练完成：%d 红球分类器 + %d 蓝球分类器",
                    len(self.red_models), len(self.blue_models))
        return self

    def predict_proba(self, latest_features: pd.DataFrame) -> dict:
        """
        返回各号码的出现概率估计。

        Parameters
        ----------
        latest_features : 最新一行特征

        Returns
        -------
        dict with red_proba (Series) and blue_proba (Series)
        """
        X = latest_features[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        red_proba = {}
        for num, clf in self.red_models.items():
            p = clf.predict_proba(X_scaled)[0]
            red_proba[num] = p[1] if len(p) > 1 else 0.0

        blue_proba = {}
        for num, clf in self.blue_models.items():
            p = clf.predict_proba(X_scaled)[0]
            blue_proba[num] = p[1] if len(p) > 1 else 0.0

        return {
            "red_proba": pd.Series(red_proba).sort_index(),
            "blue_proba": pd.Series(blue_proba).sort_index(),
        }


# ════════════════════════════════════════════════════════
# 5. XGBoost / LightGBM 模型
# ════════════════════════════════════════════════════════

class GBDTModel:
    """
    使用 XGBoost 或 LightGBM 预测号码概率。
    自动检测可用库，优先使用 LightGBM。
    """

    def __init__(self, lib: str = "auto"):
        """
        Parameters
        ----------
        lib : "xgboost" | "lightgbm" | "auto"
        """
        self.lib = lib
        self._resolve_lib()
        self.red_models: dict = {}
        self.blue_models: dict = {}
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def _resolve_lib(self):
        if self.lib == "auto":
            try:
                import lightgbm  # noqa
                self.lib = "lightgbm"
            except ImportError:
                try:
                    import xgboost  # noqa
                    self.lib = "xgboost"
                except ImportError:
                    self.lib = "sklearn_gbdt"
        logger.debug("GBDTModel 使用库：%s", self.lib)

    def _make_clf(self):
        if self.lib == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            )
        elif self.lib == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, eval_metric="logloss",
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )

    def fit(self, features_df: pd.DataFrame, target_df: pd.DataFrame) -> "GBDTModel":
        self.feature_cols = [
            c for c in features_df.columns
            if c not in ("issue", "date") and features_df[c].dtype in [np.float64, np.int64]
        ]
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        for num in range(1, 34):
            y = target_df[RED_COLS].isin([num]).any(axis=1).astype(int).values
            clf = self._make_clf()
            clf.fit(X_scaled, y)
            self.red_models[num] = clf

        for num in range(1, 17):
            y = (target_df["blue"] == num).astype(int).values
            clf = self._make_clf()
            clf.fit(X_scaled, y)
            self.blue_models[num] = clf

        logger.info("GBDTModel(%s) 训练完成。", self.lib)
        return self

    def predict_proba(self, latest_features: pd.DataFrame) -> dict:
        X = latest_features[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        red_proba = {}
        for num, clf in self.red_models.items():
            p = clf.predict_proba(X_scaled)[0]
            red_proba[num] = p[1] if len(p) > 1 else 0.0

        blue_proba = {}
        for num, clf in self.blue_models.items():
            p = clf.predict_proba(X_scaled)[0]
            blue_proba[num] = p[1] if len(p) > 1 else 0.0

        return {
            "red_proba": pd.Series(red_proba).sort_index(),
            "blue_proba": pd.Series(blue_proba).sort_index(),
        }


# ════════════════════════════════════════════════════════
# 6. 集成预测器
# ════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    将多个模型的预测结果按权重融合，生成最终推荐号码。

    集成策略：
    - 动态加权平均各模型的号码概率
    - 按概率排序后用贪心+多样性采样生成多组号码
    - 结合遗漏/热冷信号做二次调整
    """

    def __init__(self, weights: dict = None):
        """
        Parameters
        ----------
        weights : dict  各模型权重，如 {"bayesian": 0.3, "rf": 0.4, ...}
        """
        self.weights = weights or {
            "bayesian": 0.25,
            "markov": 0.10,
            "arima": 0.05,
            "rf": 0.25,
            "gbdt": 0.35,
        }
        self.bayesian: Optional[BayesianModel] = None
        self.markov: Optional[MarkovModel] = None
        self.arima: Optional[ARIMAModel] = None
        self.rf: Optional[RandomForestModel] = None
        self.gbdt: Optional[GBDTModel] = None

        # 分析器提供的统计信号
        self._freq: Optional[pd.Series] = None
        self._missing: Optional[pd.Series] = None
        self._zscore: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, features_df: pd.DataFrame = None) -> "EnsemblePredictor":
        """
        训练所有子模型。

        Parameters
        ----------
        df          : 历史数据
        features_df : 特征矩阵（可选，若为 None 则自动构建）
        """
        logger.info("开始训练集成模型...")

        # 统计信号
        from data_analyzer import frequency_analysis, missing_analysis, hot_cold_analysis
        freq_result = frequency_analysis(df)
        miss_result = missing_analysis(df)
        hc_result = hot_cold_analysis(df)

        self._freq = freq_result["red_freq"] / freq_result["red_freq"].sum()
        self._missing = miss_result["red_current_missing"]
        self._zscore = hc_result["red_zscore"]

        # 贝叶斯
        self.bayesian = BayesianModel().fit(df)

        # 马尔可夫
        self.markov = MarkovModel().fit(df)

        # ARIMA
        self.arima = ARIMAModel().fit(df)

        # ML 模型（需要特征矩阵）
        if features_df is not None and len(features_df) > 50:
            # 对齐 target
            target_df = df.iloc[len(df) - len(features_df):].reset_index(drop=True)

            try:
                self.rf = RandomForestModel().fit(features_df, target_df)
            except Exception as e:
                logger.warning("RandomForest 训练失败: %s", e)

            try:
                self.gbdt = GBDTModel().fit(features_df, target_df)
            except Exception as e:
                logger.warning("GBDT 训练失败: %s", e)

        logger.info("集成模型训练完成。")
        return self

    def _aggregate_red_proba(self, latest_features: pd.DataFrame = None) -> pd.Series:
        """汇总各模型对红球的概率估计。"""
        all_probas = []

        # 贝叶斯
        if self.bayesian and self.bayesian.red_posterior is not None:
            all_probas.append(
                (self.weights.get("bayesian", 0.25), self.bayesian.red_posterior)
            )

        # 频率归一化作为基础信号
        if self._freq is not None:
            all_probas.append((0.1, self._freq))

        # RF
        if self.rf and latest_features is not None:
            try:
                rf_pred = self.rf.predict_proba(latest_features)
                rp = rf_pred["red_proba"]
                rp = rp / (rp.sum() + 1e-10)
                all_probas.append((self.weights.get("rf", 0.25), rp))
            except Exception as e:
                logger.debug("RF 预测失败: %s", e)

        # GBDT
        if self.gbdt and latest_features is not None:
            try:
                gb_pred = self.gbdt.predict_proba(latest_features)
                rp = gb_pred["red_proba"]
                rp = rp / (rp.sum() + 1e-10)
                all_probas.append((self.weights.get("gbdt", 0.35), rp))
            except Exception as e:
                logger.debug("GBDT 预测失败: %s", e)

        if not all_probas:
            # 回退：均匀分布
            return pd.Series(1.0 / 33, index=range(1, 34))

        # 加权平均
        total_w = sum(w for w, _ in all_probas)
        agg = pd.Series(0.0, index=range(1, 34))
        for w, p in all_probas:
            aligned = p.reindex(range(1, 34), fill_value=0.0)
            agg += (w / total_w) * aligned

        # 遗漏调整：遗漏越长，适当提升概率（但不超过3倍）
        if self._missing is not None:
            miss_norm = self._missing.reindex(range(1, 34), fill_value=0)
            miss_boost = 1.0 + np.clip(miss_norm / miss_norm.max(), 0, 1) * 0.5
            agg *= miss_boost
            agg /= agg.sum()

        return agg

    def _aggregate_blue_proba(self, latest_features: pd.DataFrame = None) -> pd.Series:
        """汇总各模型对蓝球的概率估计。"""
        all_probas = []

        if self.bayesian and self.bayesian.blue_posterior is not None:
            all_probas.append((0.3, self.bayesian.blue_posterior))

        if self.markov:
            try:
                mp = self.markov.predict_blue_proba()
                mp = mp / (mp.sum() + 1e-10)
                all_probas.append((0.3, mp))
            except Exception:
                pass

        if self.rf and latest_features is not None:
            try:
                rf_pred = self.rf.predict_proba(latest_features)
                bp = rf_pred["blue_proba"]
                bp = bp / (bp.sum() + 1e-10)
                all_probas.append((0.2, bp))
            except Exception:
                pass

        if self.gbdt and latest_features is not None:
            try:
                gb_pred = self.gbdt.predict_proba(latest_features)
                bp = gb_pred["blue_proba"]
                bp = bp / (bp.sum() + 1e-10)
                all_probas.append((0.2, bp))
            except Exception:
                pass

        if not all_probas:
            return pd.Series(1.0 / 16, index=range(1, 17))

        total_w = sum(w for w, _ in all_probas)
        agg = pd.Series(0.0, index=range(1, 17))
        for w, p in all_probas:
            aligned = p.reindex(range(1, 17), fill_value=0.0)
            agg += (w / total_w) * aligned

        return agg / agg.sum()

    def predict(
        self,
        n_single: int = 10,
        complex_specs: list[tuple] = None,
        latest_features: pd.DataFrame = None,
    ) -> dict:
        """
        生成预测结果。

        Parameters
        ----------
        n_single       : 单式预测注数
        complex_specs  : 复式规格列表，如 [(7,2),(8,3)]
        latest_features: 最新特征行

        Returns
        -------
        dict with:
            single      : list of dict  单式预测
            complex     : list of dict  复式预测
            red_proba   : Series        红球概率
            blue_proba  : Series        蓝球概率
        """
        red_proba = self._aggregate_red_proba(latest_features)
        blue_proba = self._aggregate_blue_proba(latest_features)

        # ── 单式预测 ──────────────────────────────────────
        singles = []
        nums = red_proba.index.tolist()
        probs = red_proba.values / red_proba.values.sum()
        blue_nums = blue_proba.index.tolist()
        blue_probs = blue_proba.values / blue_proba.values.sum()

        # 预生成候选组合
        attempts = 0
        seen = set()
        while len(singles) < n_single and attempts < 5000:
            attempts += 1
            red = sorted(np.random.choice(nums, size=6, replace=False, p=probs).tolist())
            blue = int(np.random.choice(blue_nums, p=blue_probs))
            key = tuple(red + [blue])
            if key in seen:
                continue
            seen.add(key)

            # 基础质量过滤（软约束）
            total = sum(red)
            if not (60 <= total <= 160):
                continue
            span = red[-1] - red[0]
            if span < 10:
                continue

            confidence = (
                sum(red_proba.get(n, 0) for n in red) / 6 * 0.7
                + blue_proba.get(blue, 0) * 0.3
            )

            info = _describe_combination(red, blue)
            info["置信度"] = round(float(confidence) * 100, 2)
            info["选号依据"] = self._reason(red, blue, red_proba, blue_proba)
            singles.append(info)

        # 按置信度降序排列
        singles.sort(key=lambda x: x["置信度"], reverse=True)

        # ── 复式预测 ──────────────────────────────────────
        complexes = []
        if complex_specs is None:
            complex_specs = [(7, 2), (8, 2), (8, 3)]

        top_red = red_proba.nlargest(12).index.tolist()
        top_blue = blue_proba.nlargest(4).index.tolist()

        for n_red, n_blue in complex_specs:
            # 从概率最高的号码中选取
            c_red = sorted(top_red[:n_red])
            c_blue = sorted(top_blue[:n_blue])
            # 单注数
            from math import comb
            ticket_count = comb(n_red, 6) * n_blue
            complexes.append({
                "规格": f"{n_red}红+{n_blue}蓝",
                "红球": c_red,
                "蓝球": c_blue,
                "涵盖单注数": ticket_count,
                "参考金额(元)": ticket_count * 2,
            })

        return {
            "single": singles,
            "complex": complexes,
            "red_proba": red_proba,
            "blue_proba": blue_proba,
            "arima_sum_range": (
                self.arima.predict_sum_range() if self.arima else (60, 160)
            ),
        }

    def _reason(self, red: list[int], blue: int,
                red_proba: pd.Series, blue_proba: pd.Series) -> str:
        """生成号码选择理由说明。"""
        reasons = []
        top5_red = red_proba.nlargest(5).index.tolist()
        hot = [n for n in red if n in top5_red]
        if hot:
            reasons.append(f"包含高概率红球 {hot}")

        if self._missing is not None:
            long_miss = [n for n in red if self._missing.get(n, 0) >= 15]
            if long_miss:
                reasons.append(f"含遗漏回补号 {long_miss}")

        if self._zscore is not None:
            hot_z = [n for n in red if self._zscore.get(n, 0) > 0.5]
            if hot_z:
                reasons.append(f"近期热号 {hot_z}")

        if blue_proba.get(blue, 0) >= blue_proba.mean():
            reasons.append(f"蓝球 {blue} 处于高概率区间")

        return "；".join(reasons) if reasons else "综合统计推荐"


# ════════════════════════════════════════════════════════
# 7. 回测评估
# ════════════════════════════════════════════════════════

class Backtester:
    """
    对历史数据进行滚动回测，评估预测模型的统计性能。
    """

    def __init__(self, train_ratio: float = 0.8):
        self.train_ratio = train_ratio

    def evaluate(self, df: pd.DataFrame, n_test: int = 50) -> dict:
        """
        滚动回测：每期用前 N 期训练，预测第 N+1 期。

        Parameters
        ----------
        df     : 历史数据
        n_test : 回测期数

        Returns
        -------
        dict with evaluation metrics
        """
        if len(df) < 100 + n_test:
            n_test = max(10, len(df) - 100)

        results = []
        logger.info("开始回测，共 %d 期...", n_test)

        for i in range(n_test):
            test_idx = len(df) - n_test + i
            train_df = df.iloc[:test_idx]
            true_row = df.iloc[test_idx]
            true_reds = set(int(true_row[c]) for c in RED_COLS)
            true_blue = int(true_row["blue"])

            # 简化版快速预测（仅用贝叶斯）
            bay = BayesianModel().fit(train_df)
            pred_groups = bay.predict_red(n_groups=1)
            pred_red = set(pred_groups[0]) if pred_groups else set()
            pred_blue = bay.predict_blue()

            hit_red = len(pred_red & true_reds)
            hit_blue = (pred_blue == true_blue)

            # 计算奖级
            prize = self._get_prize(hit_red, hit_blue)
            results.append({
                "test_idx": test_idx,
                "hit_red": hit_red,
                "hit_blue": int(hit_blue),
                "prize": prize,
                "prize_money": self._get_prize_money(prize),
            })

        # 汇总指标
        res_df = pd.DataFrame(results)
        total_cost = n_test * 2  # 每注2元
        total_prize = res_df["prize_money"].sum()
        roi = (total_prize - total_cost) / total_cost * 100 if total_cost > 0 else 0

        return {
            "total_tests": n_test,
            "avg_red_hits": float(res_df["hit_red"].mean()),
            "blue_hit_rate": float(res_df["hit_blue"].mean()),
            "prize_distribution": res_df["prize"].value_counts().to_dict(),
            "total_cost": total_cost,
            "total_prize": total_prize,
            "roi": round(roi, 2),
            "details": res_df,
        }

    def compare_random(self, df: pd.DataFrame, n_test: int = 50) -> dict:
        """与随机选号对比。"""
        random_results = []
        for _ in range(n_test):
            rand_red = set(random.sample(range(1, 34), 6))
            rand_blue = random.randint(1, 16)
            # 从测试集随机选一期作为真实结果
            true_row = df.iloc[random.randint(len(df) // 2, len(df) - 1)]
            true_reds = set(int(true_row[c]) for c in RED_COLS)
            true_blue = int(true_row["blue"])
            hit_red = len(rand_red & true_reds)
            hit_blue = (rand_blue == true_blue)
            prize = self._get_prize(hit_red, hit_blue)
            random_results.append({
                "hit_red": hit_red,
                "hit_blue": int(hit_blue),
                "prize_money": self._get_prize_money(prize),
            })

        rr = pd.DataFrame(random_results)
        return {
            "avg_red_hits": float(rr["hit_red"].mean()),
            "blue_hit_rate": float(rr["hit_blue"].mean()),
            "roi": round(
                (rr["prize_money"].sum() - n_test * 2) / (n_test * 2) * 100, 2
            ),
        }

    @staticmethod
    def _get_prize(hit_red: int, hit_blue: bool) -> str:
        if hit_red == 6 and hit_blue:
            return "一等奖"
        if hit_red == 6:
            return "二等奖"
        if hit_red == 5 and hit_blue:
            return "三等奖"
        if hit_red == 5 or (hit_red == 4 and hit_blue):
            return "四等奖"
        if hit_red == 4 or (hit_red == 3 and hit_blue):
            return "五等奖"
        if hit_blue and hit_red < 3:
            return "六等奖"
        return "未中奖"

    @staticmethod
    def _get_prize_money(prize: str) -> int:
        mapping = {
            "一等奖": 5_000_000,
            "二等奖": 200_000,
            "三等奖": 3_000,
            "四等奖": 200,
            "五等奖": 10,
            "六等奖": 5,
            "未中奖": 0,
        }
        return mapping.get(prize, 0)


# ════════════════════════════════════════════════════════
# 便捷函数
# ════════════════════════════════════════════════════════

def format_prediction_output(prediction: dict) -> str:
    """将预测结果格式化为人类可读的文本。"""
    lines = []
    lines.append("=" * 65)
    lines.append("双色球预测结果（统计研究，仅供参考，切勿沉迷）")
    lines.append("=" * 65)

    lines.append("\n▶ 单式推荐（按置信度排序）")
    lines.append("-" * 65)
    for i, pred in enumerate(prediction["single"], 1):
        red_str = " ".join(f"{n:02d}" for n in pred["numbers"])
        blue_str = f"{pred['blue']:02d}"
        lines.append(
            f"  第{i:02d}注: 红球[{red_str}] + 蓝球[{blue_str}]"
            f"  置信度:{pred['置信度']:.1f}%"
        )
        lines.append(
            f"        奇偶:{pred['奇偶比']}  区间:{pred['区间分布']}"
            f"  和:{pred['和值']}  跨:{pred['跨度']}"
            f"  AC:{pred['AC值']}"
        )
        lines.append(f"        理由：{pred['选号依据']}")

    lines.append("\n▶ 复式推荐")
    lines.append("-" * 65)
    for cp in prediction["complex"]:
        red_str = " ".join(f"{n:02d}" for n in cp["红球"])
        blue_str = " ".join(f"{n:02d}" for n in cp["蓝球"])
        lines.append(
            f"  {cp['规格']}  [{red_str}] | [{blue_str}]"
            f"  涵盖{cp['涵盖单注数']}注 / ¥{cp['参考金额(元)']}元"
        )

    sum_range = prediction.get("arima_sum_range", (60, 160))
    lines.append(f"\n▶ 和值预测区间（ARIMA）: {sum_range[0]:.0f} ~ {sum_range[1]:.0f}")

    lines.append("\n" + "=" * 65)
    lines.append("⚠ 彩票开奖完全随机，上述结果基于统计模型，不构成投资建议。")
    lines.append("⚠ 请理性购彩，量力而行。未成年人禁止购彩。")
    lines.append("=" * 65)

    return "\n".join(lines)
