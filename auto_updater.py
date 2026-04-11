"""
双色球预测系统 - 自动更新模块
==============================
实现定时数据更新、模型重训练、预测生成和验证功能。
免责声明：本模块仅用于概率统计研究和学术学习。
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    DATA_DIR,
    DRAW_DAYS,
    DRAW_HOUR,
    DRAW_MINUTE,
    MODEL_DIR,
    PREDICTION_LOG,
    REPORT_DIR,
    RETRAIN_THRESHOLD,
    UPDATE_HOUR,
    UPDATE_MINUTE,
)

logger = logging.getLogger(__name__)

# 模型缓存文件
MODEL_CACHE = MODEL_DIR / "ensemble_model.pkl"
WEIGHTS_FILE = MODEL_DIR / "model_weights.json"
LAST_UPDATE_FILE = DATA_DIR / ".last_update"


# ════════════════════════════════════════════════════════
# 状态管理
# ════════════════════════════════════════════════════════

def _read_last_update() -> Optional[str]:
    """读取上次更新记录。"""
    if LAST_UPDATE_FILE.exists():
        return LAST_UPDATE_FILE.read_text().strip()
    return None


def _write_last_update(issue: str):
    """记录最新期号。"""
    LAST_UPDATE_FILE.write_text(str(issue))


def _save_model(model) -> None:
    """序列化保存集成模型。"""
    try:
        with open(MODEL_CACHE, "wb") as f:
            pickle.dump(model, f, protocol=4)
        logger.info("模型已保存：%s", MODEL_CACHE)
    except Exception as e:
        logger.error("模型保存失败：%s", e)


def _load_model():
    """从磁盘加载集成模型。"""
    if not MODEL_CACHE.exists():
        return None
    try:
        with open(MODEL_CACHE, "rb") as f:
            model = pickle.load(f)
        logger.info("模型加载成功：%s", MODEL_CACHE)
        return model
    except Exception as e:
        logger.warning("模型加载失败（%s），将重新训练。", e)
        return None


def _save_weights(weights: dict) -> None:
    """保存动态权重。"""
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)


def _load_weights() -> Optional[dict]:
    """加载动态权重。"""
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════
# 预测日志
# ════════════════════════════════════════════════════════

def save_prediction(prediction: dict, target_issue: str) -> None:
    """
    将预测结果保存到历史日志（用于事后验证）。

    Parameters
    ----------
    prediction   : EnsemblePredictor.predict() 的返回值
    target_issue : 预测的目标期号
    """
    rows = []
    for i, pred in enumerate(prediction.get("single", []), 1):
        rows.append({
            "predict_time": datetime.now().isoformat(),
            "target_issue": target_issue,
            "group": i,
            "r1": pred["numbers"][0],
            "r2": pred["numbers"][1],
            "r3": pred["numbers"][2],
            "r4": pred["numbers"][3],
            "r5": pred["numbers"][4],
            "r6": pred["numbers"][5],
            "blue": pred["blue"],
            "confidence": pred.get("置信度", 0),
        })

    new_df = pd.DataFrame(rows)
    if PREDICTION_LOG.exists():
        old_df = pd.read_csv(PREDICTION_LOG)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(PREDICTION_LOG, index=False)
    logger.info("预测结果已记录：目标期号 %s，共 %d 注", target_issue, len(rows))


def verify_prediction(actual_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    将历史预测与真实开奖结果对比验证。

    Parameters
    ----------
    actual_df : 真实开奖数据（含已开奖期号）

    Returns
    -------
    DataFrame  验证结果（命中情况）
    """
    if not PREDICTION_LOG.exists():
        logger.info("暂无预测历史日志。")
        return None

    pred_df = pd.read_csv(PREDICTION_LOG, dtype={"target_issue": str})
    actual_df = actual_df.copy()
    actual_df["issue"] = actual_df["issue"].astype(str)

    RED_COLS = ["r1", "r2", "r3", "r4", "r5", "r6"]
    results = []

    for _, pred_row in pred_df.iterrows():
        issue = str(pred_row["target_issue"])
        match = actual_df[actual_df["issue"] == issue]
        if match.empty:
            continue  # 该期尚未开奖

        actual = match.iloc[0]
        actual_reds = set(int(actual[c]) for c in RED_COLS)
        actual_blue = int(actual["blue"])

        pred_reds = set(int(pred_row[c]) for c in RED_COLS)
        pred_blue = int(pred_row["blue"])

        hit_red = len(pred_reds & actual_reds)
        hit_blue = int(pred_blue == actual_blue)

        prize = _get_prize_level(hit_red, hit_blue)
        results.append({
            "target_issue": issue,
            "group": pred_row.get("group", 0),
            "hit_red": hit_red,
            "hit_blue": hit_blue,
            "prize": prize,
            "confidence": pred_row.get("confidence", 0),
        })

    if not results:
        logger.info("暂无可验证的历史预测（对应期号尚未开奖）。")
        return None

    result_df = pd.DataFrame(results)
    logger.info(
        "验证完成：%d 条记录，平均红球命中 %.2f 个，蓝球命中率 %.1f%%",
        len(result_df),
        result_df["hit_red"].mean(),
        result_df["hit_blue"].mean() * 100,
    )
    return result_df


def _get_prize_level(hit_red: int, hit_blue: bool) -> str:
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
    if hit_blue:
        return "六等奖"
    return "未中奖"


# ════════════════════════════════════════════════════════
# 核心更新流程
# ════════════════════════════════════════════════════════

class AutoUpdater:
    """
    自动更新管理器：
    - 检查新开奖数据
    - 触发模型重训练
    - 生成下期预测
    - 验证历史预测
    """

    def __init__(self):
        self._new_data_count = 0

    def is_draw_day(self) -> bool:
        """判断今天是否为开奖日（周二、四、日）。"""
        return datetime.now().weekday() in DRAW_DAYS

    def should_update_now(self) -> bool:
        """判断当前时间是否应执行更新（开奖日21:30后）。"""
        now = datetime.now()
        if not self.is_draw_day():
            return False
        return now.hour > UPDATE_HOUR or (
            now.hour == UPDATE_HOUR and now.minute >= UPDATE_MINUTE
        )

    def run_update_cycle(self, force_retrain: bool = False) -> dict:
        """
        执行完整更新周期：
        1. 拉取最新开奖数据
        2. 验证历史预测
        3. 如需要，重新训练模型
        4. 生成新一期预测
        5. 生成报告

        Returns
        -------
        dict  本次更新的结果摘要
        """
        from data_scraper import DataManager
        from data_analyzer import DataAnalyzer, build_features
        from prediction_model import EnsemblePredictor, format_prediction_output

        summary = {
            "update_time": datetime.now().isoformat(),
            "new_periods": 0,
            "retrained": False,
            "prediction_generated": False,
            "error": None,
        }

        try:
            # ── Step 1: 数据更新 ─────────────────────────
            logger.info("Step 1/4: 更新数据...")
            mgr = DataManager()
            df, new_count = mgr.update()
            summary["new_periods"] = new_count
            self._new_data_count += new_count

            if df.empty:
                summary["error"] = "数据获取失败"
                return summary

            logger.info("当前共 %d 期数据，本次新增 %d 期。", len(df), new_count)

            # ── Step 2: 验证历史预测 ──────────────────────
            logger.info("Step 2/4: 验证历史预测...")
            verify_df = verify_prediction(df)
            if verify_df is not None and not verify_df.empty:
                summary["verification"] = {
                    "records": len(verify_df),
                    "avg_red_hits": round(float(verify_df["hit_red"].mean()), 2),
                    "blue_hit_rate": round(float(verify_df["hit_blue"].mean()), 4),
                    "prize_dist": verify_df["prize"].value_counts().to_dict(),
                }
                self._update_weights_from_verify(verify_df)

            # ── Step 3: 重训练判断 ────────────────────────
            need_retrain = (
                force_retrain
                or not MODEL_CACHE.exists()
                or self._new_data_count >= RETRAIN_THRESHOLD
            )

            if need_retrain:
                logger.info("Step 3/4: 重新训练模型...")
                model = self._train(df)
                if model:
                    _save_model(model)
                    self._new_data_count = 0
                    summary["retrained"] = True
                else:
                    logger.warning("模型训练失败，尝试加载旧模型。")
                    model = _load_model()
            else:
                logger.info("Step 3/4: 加载已有模型（新增 %d 期，未达重训练阈值 %d）。",
                            new_count, RETRAIN_THRESHOLD)
                model = _load_model()
                if model is None:
                    model = self._train(df)
                    if model:
                        _save_model(model)
                    summary["retrained"] = True

            # ── Step 4: 生成预测 ──────────────────────────
            logger.info("Step 4/4: 生成下期预测...")
            if model:
                # 计算最新特征
                try:
                    feat_df = build_features(df, windows=[10, 20, 30])
                    latest_feat = feat_df.tail(1).reset_index(drop=True)
                except Exception as e:
                    logger.warning("特征构建失败：%s，不使用 ML 特征。", e)
                    latest_feat = None

                from config import NUM_SINGLE_PREDICTIONS, COMPLEX_SPECS
                pred = model.predict(
                    n_single=NUM_SINGLE_PREDICTIONS,
                    complex_specs=COMPLEX_SPECS,
                    latest_features=latest_feat,
                )

                # 推算下期期号
                next_issue = self._next_issue(df)
                save_prediction(pred, next_issue)
                summary["prediction_generated"] = True
                summary["next_issue"] = next_issue

                # 保存文本报告
                report_text = format_prediction_output(pred)
                report_file = REPORT_DIR / f"prediction_{next_issue}.txt"
                report_file.write_text(report_text, encoding="utf-8")
                logger.info("预测报告已保存：%s", report_file)
                summary["report_file"] = str(report_file)

        except Exception as e:
            logger.exception("更新周期异常：%s", e)
            summary["error"] = str(e)

        return summary

    def _train(self, df: pd.DataFrame):
        """内部训练流程，返回训练好的 EnsemblePredictor。"""
        try:
            from data_analyzer import build_features
            from prediction_model import EnsemblePredictor

            logger.info("正在构建特征矩阵...")
            feat_df = build_features(df, windows=[10, 20, 30])

            weights = _load_weights()
            model = EnsemblePredictor(weights=weights)
            model.fit(df, features_df=feat_df)
            return model
        except Exception as e:
            logger.error("模型训练异常：%s", e)
            return None

    def _update_weights_from_verify(self, verify_df: pd.DataFrame) -> None:
        """
        根据验证结果动态调整模型权重（简单启发式）。
        （此处为示意，真实场景可用在线学习算法）
        """
        weights = _load_weights()
        if weights is None:
            return
        # 若蓝球命中率高于期望值 (1/16=6.25%)，适当提升马尔可夫权重
        blue_rate = verify_df["hit_blue"].mean()
        if blue_rate > 0.10:
            weights["markov"] = min(0.20, weights.get("markov", 0.10) * 1.1)
        elif blue_rate < 0.04:
            weights["markov"] = max(0.05, weights.get("markov", 0.10) * 0.9)
        _save_weights(weights)
        logger.debug("权重已更新：%s", weights)

    @staticmethod
    def _next_issue(df: pd.DataFrame) -> str:
        """推算下一期期号（通常为当前最大期号 + 1）。"""
        max_issue = df["issue"].max()
        try:
            return str(int(max_issue) + 1)
        except ValueError:
            return max_issue + "_next"


# ════════════════════════════════════════════════════════
# 定时调度（使用 schedule 库）
# ════════════════════════════════════════════════════════

def start_scheduler(run_now: bool = False) -> None:
    """
    启动定时调度器，在每个开奖日的 21:30 执行更新。

    Parameters
    ----------
    run_now : bool  True 时立即执行一次，然后再定时
    """
    try:
        import schedule
    except ImportError:
        logger.error("请安装 schedule 库：pip install schedule")
        return

    updater = AutoUpdater()

    def job():
        logger.info("=" * 50)
        logger.info("定时任务触发，开始执行更新周期...")
        summary = updater.run_update_cycle()
        logger.info("更新完成：%s", summary)

    # 每天 21:30 执行
    update_time = f"{UPDATE_HOUR:02d}:{UPDATE_MINUTE:02d}"
    schedule.every().day.at(update_time).do(job)
    logger.info("定时调度已启动，每天 %s 检查更新。", update_time)

    if run_now:
        logger.info("立即执行一次更新...")
        job()

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("调度器已停止。")


# ════════════════════════════════════════════════════════
# 命令行入口
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("双色球自动更新模块")
    print("用法：python auto_updater.py [--now] [--force-retrain] [--daemon]")

    force_retrain = "--force-retrain" in sys.argv
    run_now = "--now" in sys.argv
    daemon = "--daemon" in sys.argv

    if daemon:
        start_scheduler(run_now=run_now)
    else:
        updater = AutoUpdater()
        summary = updater.run_update_cycle(force_retrain=force_retrain)
        print("\n更新结果：")
        for k, v in summary.items():
            print(f"  {k}: {v}")
