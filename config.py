"""
双色球预测系统 - 配置文件
==============================
免责声明：本系统仅用于概率统计研究和学术学习目的。
彩票开奖具有高度随机性，历史数据无法预测未来结果。
请理性对待，切勿沉迷。
"""

import os
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"

# 确保目录存在
for d in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 数据文件路径 ─────────────────────────────────────────
DATA_FILE = DATA_DIR / "ssq_history.csv"
BACKUP_DATA_FILE = DATA_DIR / "ssq_history_backup.csv"
FEATURES_FILE = DATA_DIR / "ssq_features.csv"
PREDICTION_LOG = DATA_DIR / "prediction_history.csv"

# ── 数据源配置 ───────────────────────────────────────────
# 主数据源：中国福彩网官方 API
CWL_API_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
)
CWL_API_PARAMS = {
    "name": "ssq",
    "issueCount": "",
    "issueStart": "",
    "issueEnd": "",
    "dayStart": "",
    "dayEnd": "",
    "pageNum": 1,
    "pageSize": 30,
    "week": "",
    "systemType": "PC",
}

# 备用数据源（500彩票网 JSON 接口）
BACKUP_API_URL = "https://datachart.500.com/ssq/history/newinit.php"
BACKUP_API_PARAMS = {"limit": 500, "type": "history"}

# 第三方历史数据（用于首次初始化，下载500期以上）
THIRD_PARTY_HISTORY_URL = (
    "https://datachart.500.com/ssq/history/newinit.php?limit=1000"
)

# HTTP 请求配置
REQUEST_TIMEOUT = 30  # 秒
REQUEST_RETRIES = 3
REQUEST_DELAY = 1.5   # 两次请求之间的延迟（秒）
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.cwl.gov.cn/",
}

# ── 双色球规则常量 ───────────────────────────────────────
RED_BALL_MIN = 1
RED_BALL_MAX = 33
RED_BALL_COUNT = 6          # 每期选 6 个红球
BLUE_BALL_MIN = 1
BLUE_BALL_MAX = 16
BLUE_BALL_COUNT = 1         # 每期选 1 个蓝球

# 红球区间划分（用于区间分析）
RED_ZONES = {
    "zone1": (1, 11),    # 第一区
    "zone2": (12, 22),   # 第二区
    "zone3": (23, 33),   # 第三区
}

# 奖级定义（红球命中数, 蓝球命中）
PRIZE_LEVELS = {
    "一等奖": (6, True),
    "二等奖": (6, False),
    "三等奖": (5, True),
    "四等奖": (5, False),
    "五等奖": (4, True),
    "六等奖": (3, True),
    "小奖":   (4, False),   # 4红0蓝（无奖）— 仅统计用
}

# 各奖级固定奖金（元）
PRIZE_MONEY = {
    "一等奖": 5_000_000,   # 浮动，此处用保底值
    "二等奖": 200_000,     # 浮动，此处用参考值
    "三等奖": 3_000,
    "四等奖": 200,
    "五等奖": 10,
    "六等奖": 5,
}

# 单注投注成本
TICKET_PRICE = 2  # 元

# ── 分析参数 ─────────────────────────────────────────────
# 热冷分析窗口（期数）
HOT_WINDOW = 30
COLD_WINDOW = 50

# 遗漏分析参数
MAX_MISSING_THRESHOLD = 50   # 超过此遗漏期数认为"超长遗漏"

# 最小历史数据期数要求
MIN_HISTORY_PERIODS = 100

# 特征工程：滑动窗口大小列表
FEATURE_WINDOWS = [10, 20, 30, 50, 100]

# 傅里叶变换周期检测最大周期（期）
MAX_PERIOD = 100

# ── 机器学习模型参数 ─────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2              # 训练/测试集比例
CROSS_VAL_FOLDS = 5

# RandomForest 参数
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# XGBoost 参数
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM 参数
LGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ── 预测输出配置 ─────────────────────────────────────────
NUM_SINGLE_PREDICTIONS = 10   # 单式预测注数
NUM_COMPLEX_PREDICTIONS = 3   # 复式预测组数

# 复式规格：(红球数, 蓝球数)
COMPLEX_SPECS = [
    (7, 2),
    (8, 2),
    (8, 3),
]

# 集成模型权重初始值（后续根据回测动态调整）
ENSEMBLE_WEIGHTS = {
    "bayesian": 0.20,
    "markov": 0.10,
    "arima": 0.10,
    "random_forest": 0.20,
    "xgboost": 0.20,
    "lightgbm": 0.20,
}

# ── 自动更新配置 ─────────────────────────────────────────
# 双色球开奖时间：周二、四、日 20:30
DRAW_DAYS = [1, 3, 6]   # 0=周一, 1=周二, ...
DRAW_HOUR = 20
DRAW_MINUTE = 30

# 自动更新检查时间（每天晚上 21:30）
UPDATE_HOUR = 21
UPDATE_MINUTE = 30

# 模型重训练触发条件（新数据累计达到 N 期）
RETRAIN_THRESHOLD = 5

# ── 日志配置 ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOG_DIR / "ssq_system.log"
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5

# ── 可视化配置 ───────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_SIZE_SQUARE = (12, 10)
COLOR_RED = "#E74C3C"
COLOR_BLUE = "#2980B9"
COLOR_GREEN = "#27AE60"
COLOR_GOLD = "#F39C12"

# ── 免责声明 ─────────────────────────────────────────────
DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════╗
║                        ⚠ 重要免责声明 ⚠                          ║
╠══════════════════════════════════════════════════════════════════╣
║  本系统仅供概率统计研究和学术学习使用，不构成任何投资建议。          ║
║  双色球开奖由随机机械摇号产生，具有高度随机性。                      ║
║  历史数据无法预测未来开奖结果，任何"必中"宣传均为虚假信息。          ║
║  请理性购彩，量力而行，切勿沉迷，切勿借贷购彩。                      ║
║  未成年人禁止购买彩票。                                            ║
╚══════════════════════════════════════════════════════════════════╝
"""
