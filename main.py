"""
双色球预测系统 - 主程序
==============================
整合所有模块，提供命令行接口。

用法示例：
    python main.py --help
    python main.py init                    # 初始化并获取历史数据
    python main.py predict                 # 生成本期预测
    python main.py update                  # 增量更新数据并预测
    python main.py analyze                 # 运行统计分析
    python main.py backtest --periods 100  # 回测评估
    python main.py report                  # 生成综合报告
    python main.py daemon                  # 启动定时守护进程

免责声明：本系统仅用于概率统计研究和学术学习，不构成投资建议。
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from config import (
    DISCLAIMER,
    LOG_BACKUP_COUNT,
    LOG_DATE_FORMAT,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    NUM_SINGLE_PREDICTIONS,
    COMPLEX_SPECS,
    REPORT_DIR,
)


# ════════════════════════════════════════════════════════
# 日志配置
# ════════════════════════════════════════════════════════

def setup_logging(level: str = LOG_LEVEL, verbose: bool = False) -> None:
    """配置日志系统（同时输出到控制台和文件）。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # 控制台处理器
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else getattr(logging, level))
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # 文件处理器（滚动日志）
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"警告：无法创建日志文件 ({e})，仅输出到控制台。")


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
# 子命令实现
# ════════════════════════════════════════════════════════

def cmd_init(args) -> int:
    """初始化：从网络获取历史数据，完成后启动状态服务。"""
    from data_scraper import DataManager

    print("\n初始化数据（首次运行可能需要几分钟）...")
    mgr = DataManager()
    df = mgr.initialize(force_refresh=args.refresh)

    if df.empty:
        print("错误：数据初始化失败，请检查网络连接。")
        return 1

    print(f"✓ 数据初始化成功：共 {len(df)} 期")
    print(f"  期号范围：{df['issue'].min()} ~ {df['issue'].max()}")
    print(f"  日期范围：{df['date'].min()} ~ {df['date'].max()}")

    report = mgr.quality_check(df)
    print(f"  数据质量：{report['status']}")
    for issue in report["issues"]:
        print(f"    - {issue}")

    # 启动内置 HTTP 控制面板，保持进程存活
    port = int(os.environ.get("PORT", getattr(args, "port", 3000)))
    _start_dashboard(port=port)
    return 0


# ════════════════════════════════════════════════════════
# Web 控制面板（多路由 HTTP 服务器）
# ════════════════════════════════════════════════════════

def _start_dashboard(port: int = 3000) -> None:
    """
    启动功能完整的 Web 控制面板，支持：
    - 实时刷新开奖数据
    - 购彩结果录入与追踪
    - 命中统计与模型反馈优化
    - 动态预测展示
    """
    import glob
    import json as _json
    import threading
    import urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import numpy as np
    import pandas as pd

    from config import DATA_DIR, REPORT_DIR
    from data_scraper import DataManager

    RED_COLS = ["r1", "r2", "r3", "r4", "r5", "r6"]
    PURCHASES_FILE = DATA_DIR / "purchases.json"
    TOKENS_FILE = DATA_DIR / "paid_tokens.json"
    ADMIN_KEY = os.environ.get("ADMIN_KEY", "ssq-admin-2024")

    # ── Token 管理 ───────────────────────────────────────
    def load_tokens() -> dict:
        if TOKENS_FILE.exists():
            try:
                with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                    return _json.load(f)
            except Exception:
                pass
        return {}

    def save_tokens(tokens: dict):
        TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKENS_FILE, "w", encoding="utf-8") as f:
            _json.dump(tokens, f, ensure_ascii=False, indent=2)

    def generate_token(plan: str = "monthly") -> str:
        import uuid, time
        now = time.time()
        expire = {
            "trial": now + 3600,          # 1小时试用
            "once": now + 86400,           # 24小时单次
            "monthly": now + 86400 * 30,   # 30天
            "yearly": now + 86400 * 365,   # 年度
            "lifetime": now + 86400 * 36500,
        }.get(plan, now + 86400 * 30)
        tok = str(uuid.uuid4()).replace("-", "")[:20].upper()
        tokens = load_tokens()
        tokens[tok] = {"plan": plan, "expire": expire, "created": now, "uses": 0}
        save_tokens(tokens)
        return tok

    def verify_token(tok: str) -> dict:
        import time
        tokens = load_tokens()
        if tok not in tokens:
            return {"valid": False, "msg": "无效Token"}
        t = tokens[tok]
        if time.time() > t["expire"]:
            return {"valid": False, "msg": "Token已过期"}
        t["uses"] = t.get("uses", 0) + 1
        save_tokens(tokens)
        return {"valid": True, "plan": t["plan"],
                "expire": t["expire"], "uses": t["uses"]}

    # ── 共享状态（线程安全用锁）─────────────────────────────
    state_lock = threading.Lock()
    state = {"refreshing": False, "refresh_msg": "", "last_refresh": "启动时加载"}

    # ── 购彩记录持久化 ────────────────────────────────────
    def load_purchases() -> list:
        if PURCHASES_FILE.exists():
            with open(PURCHASES_FILE, encoding="utf-8") as f:
                return _json.load(f)
        return []

    def save_purchases(records: list) -> None:
        with open(PURCHASES_FILE, "w", encoding="utf-8") as f:
            _json.dump(records, f, ensure_ascii=False, indent=2)

    # ── 奖级判断 ─────────────────────────────────────────
    def prize_level(hit_red: int, hit_blue: bool) -> tuple[str, int]:
        """返回 (奖级名称, 奖金)"""
        table = [
            (6, True,  "一等奖", 5_000_000),
            (6, False, "二等奖", 200_000),
            (5, True,  "三等奖", 3_000),
            (5, False, "四等奖", 200),
            (4, True,  "四等奖", 200),
            (4, False, "五等奖", 10),
            (3, True,  "五等奖", 10),
            (0, True,  "六等奖", 5),
            (1, True,  "六等奖", 5),
            (2, True,  "六等奖", 5),
        ]
        for r, b, name, money in table:
            if hit_red == r and hit_blue == b:
                return name, money
        return "未中奖", 0

    # ── 模型权重自适应优化 ─────────────────────────────────
    def optimize_weights_from_feedback(purchases: list) -> None:
        """
        根据购彩反馈记录调整集成模型权重并重新保存。
        策略：统计各模型推荐号码的平均命中率，按比例重分配权重。
        """
        from auto_updater import _load_model, _save_model
        from config import ENSEMBLE_WEIGHTS

        verified = [p for p in purchases if p.get("actual_red") and p.get("hit_red") is not None]
        if len(verified) < 3:
            return  # 样本太少，不调整

        # 计算平均命中红球数和蓝球命中率作为基准
        avg_hit_red = sum(p["hit_red"] for p in verified) / len(verified)
        avg_hit_blue = sum(1 for p in verified if p.get("hit_blue")) / len(verified)

        model = _load_model()
        if model is None:
            return

        weights = dict(model.weights) if hasattr(model, "weights") and model.weights else dict(ENSEMBLE_WEIGHTS)

        # 蓝球命中率超过随机期望(6.25%)时加强马尔可夫权重
        if avg_hit_blue > 0.10:
            weights["markov"] = min(0.20, weights.get("markov", 0.10) * 1.15)
        elif avg_hit_blue < 0.04:
            weights["markov"] = max(0.05, weights.get("markov", 0.10) * 0.85)

        # 红球命中率超过随机期望(≈1.09)时加强贝叶斯权重
        if avg_hit_red > 1.5:
            weights["bayesian"] = min(0.35, weights.get("bayesian", 0.20) * 1.1)
        elif avg_hit_red < 0.8:
            weights["bayesian"] = max(0.10, weights.get("bayesian", 0.20) * 0.9)

        # 重新归一化
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}

        model.weights = weights
        _save_model(model)
        logger.info("模型权重已根据购彩反馈更新：%s", weights)

    # ── 数据快照（供渲染用）───────────────────────────────
    def build_snapshot() -> dict:
        mgr = DataManager()
        df = mgr.load_local()
        if df.empty:
            return {}

        red_flat = df[RED_COLS].values.flatten()
        red_freq = {n: int((red_flat == n).sum()) for n in range(1, 34)}
        blue_freq = {n: int((df["blue"] == n).sum()) for n in range(1, 17)}
        top10_red = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        top5_blue = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        latest10 = df.tail(10)[["issue", "date"] + RED_COLS + ["blue"]].to_dict(orient="records")

        pred_content = ""
        pred_files = sorted(glob.glob(str(REPORT_DIR / "prediction_*.txt")), reverse=True)
        if pred_files:
            with open(pred_files[0], encoding="utf-8") as f:
                pred_content = f.read()

        return {
            "total": len(df),
            "min_issue": df["issue"].min(),
            "max_issue": df["issue"].max(),
            "min_date": str(df["date"].min()),
            "max_date": str(df["date"].max()),
            "top10_red": top10_red,
            "top5_blue": top5_blue,
            "latest10": latest10,
            "pred_content": pred_content,
        }

    # ── HTML 渲染 ─────────────────────────────────────────
    def render_html() -> str:
        snap = build_snapshot()
        purchases = load_purchases()

        def ball_html(nums, cls):
            return "".join(f'<span class="{cls}">{int(n):02d}</span>' for n in nums)

        latest_rows = ""
        for r in reversed(snap.get("latest10", [])):
            reds = ball_html([r[c] for c in RED_COLS], "red")
            blue = f'<span class="blue">{int(r["blue"]):02d}</span>'
            latest_rows += f"<tr><td>{r['issue']}</td><td>{r['date']}</td><td>{reds}</td><td>{blue}</td></tr>"

        top_red_html = "".join(
            f'<span class="red">{n:02d}</span><small class="freq-cnt">×{cnt}</small> '
            for n, cnt in snap.get("top10_red", [])
        )
        top_blue_html = "".join(
            f'<span class="blue">{n:02d}</span><small class="freq-cnt">×{cnt}</small> '
            for n, cnt in snap.get("top5_blue", [])
        )

        pred_html = ""
        pc = snap.get("pred_content", "")
        if pc:
            pred_html = f'<section id="pred-section"><h2>🎯 最新预测</h2><div class="card"><pre id="pred-text">{pc}</pre></div></section>'

        # 购彩历史行
        purchase_rows = ""
        total_cost = 0
        total_prize = 0
        for p in reversed(purchases[-30:]):  # 最近30条
            ts = p.get("time", "")[:16]
            issue = p.get("issue", "-")
            ptype = p.get("type", "单式")
            type_badge = (
                '<span class="badge badge-complex">复式</span>'
                if ptype == "复式" else
                '<span class="badge badge-single">单式</span>'
            )

            # 红球展示：复式显示全部选择的球，并加括号提示
            my_reds_html = ball_html(p.get("my_red", []), "red")
            if ptype == "复式":
                n_red = len(p.get("my_red", []))
                my_reds_html = f'<span style="font-size:10px;color:#888">{n_red}红</span> ' + my_reds_html

            # 蓝球展示：复式可能多个
            my_blues = p.get("my_blue", [])
            if isinstance(my_blues, list):
                my_blue_html = "".join(f'<span class="blue">{int(b):02d}</span>' for b in my_blues)
                if ptype == "复式":
                    my_blue_html = f'<span style="font-size:10px;color:#888">{len(my_blues)}蓝</span> ' + my_blue_html
            else:
                my_blue_html = f'<span class="blue">{int(my_blues):02d}</span>' if my_blues else "-"

            tickets = p.get("tickets", 1)
            cost = tickets * 2
            total_cost += cost

            if p.get("actual_red"):
                a_reds = ball_html(p["actual_red"], "red-dim")
                a_blue_val = p.get("actual_blue")
                a_blue = f'<span class="blue-dim">{int(a_blue_val):02d}</span>' if a_blue_val else "-"
                prize_name = p.get("prize", "未中奖")
                prize_money = p.get("prize_money", 0)
                total_prize += prize_money
                prize_cls = "prize-win" if prize_money > 0 else "prize-none"

                # 命中摘要
                if ptype == "复式":
                    hit_summary = p.get("hit_summary", f'最高{p.get("hit_red",0)}红')
                else:
                    hr = p.get("hit_red", 0)
                    hb = "✓" if p.get("hit_blue") else "✗"
                    hit_summary = f'红{hr}个 蓝{hb}'

                result_td = (
                    f'<td>{a_reds} {a_blue}</td>'
                    f'<td>{hit_summary}</td>'
                    f'<td class="{prize_cls}">{prize_name}<br><small>¥{prize_money:,}</small></td>'
                    f'<td><button class="btn-sm btn-del" onclick="deletePurchase(\'{p["id"]}\')">删除</button></td>'
                )
            else:
                result_td = (
                    f'<td colspan="2"><button class="btn-sm btn-fill" '
                    f'onclick="fillResult(\'{p["id"]}\',\'{issue}\')">录入开奖结果</button></td>'
                    f'<td>-</td>'
                    f'<td><button class="btn-sm btn-del" onclick="deletePurchase(\'{p["id"]}\')">删除</button></td>'
                )

            purchase_rows += (
                f'<tr><td>{ts}<br>{type_badge}</td><td>{issue}</td>'
                f'<td>{my_reds_html}</td><td>{my_blue_html}</td>'
                f'<td>{tickets}注<br>¥{cost}</td>'
                f'{result_td}</tr>'
            )

        roi = (total_prize - total_cost) / total_cost * 100 if total_cost > 0 else 0
        roi_cls = "prize-win" if roi > 0 else "prize-none"

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>双色球预测系统</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'PingFang SC','Helvetica Neue',Arial,sans-serif;background:#f0f2f5;color:#333;min-height:100vh}}
header{{background:linear-gradient(135deg,#c0392b,#922b21);color:#fff;padding:18px 28px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 8px rgba(0,0,0,.2)}}
header h1{{font-size:20px;font-weight:700;letter-spacing:1px}}
.header-right{{display:flex;gap:10px;align-items:center}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px}}
h2{{color:#2c3e50;font-size:16px;margin:24px 0 10px;display:flex;align-items:center;gap:6px}}
.card{{background:#fff;border-radius:10px;padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:700px){{.grid2{{grid-template-columns:1fr}}}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}}
@media(max-width:700px){{.stat-grid{{grid-template-columns:repeat(2,1fr)}}}}
.stat-box{{background:#fff;border-radius:10px;padding:14px 18px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.stat-box .val{{font-size:26px;font-weight:700;color:#c0392b}}
.stat-box .lbl{{font-size:12px;color:#888;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{background:#34495e;color:#fff;padding:8px 10px;text-align:center;font-weight:500}}
td{{border-bottom:1px solid #eee;padding:7px 10px;text-align:center;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#fafafa}}
.red,.blue,.red-dim,.blue-dim{{border-radius:50%;display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;margin:1px;font-size:12px;font-weight:600}}
.red{{background:#e74c3c;color:#fff}}
.blue{{background:#2980b9;color:#fff}}
.red-dim{{background:#f1948a;color:#fff}}
.blue-dim{{background:#7fb3d3;color:#fff}}
.freq-cnt{{font-size:10px;color:#888;margin-right:6px}}
pre{{background:#1a1a2e;color:#e0e0e0;padding:16px;border-radius:8px;font-size:12.5px;white-space:pre-wrap;line-height:1.6;overflow-x:auto}}
.disclaimer{{background:#fff8e1;border-left:4px solid #f39c12;padding:10px 16px;border-radius:0 6px 6px 0;font-size:13px;color:#7d6608;margin-bottom:18px}}
/* 按钮 */
.btn{{display:inline-flex;align-items:center;gap:6px;padding:9px 20px;border:none;border-radius:7px;cursor:pointer;font-size:14px;font-weight:600;transition:all .2s}}
.btn-primary{{background:#2980b9;color:#fff}} .btn-primary:hover{{background:#1f6391}}
.btn-success{{background:#27ae60;color:#fff}} .btn-success:hover{{background:#1e8449}}
.btn-warning{{background:#e67e22;color:#fff}} .btn-warning:hover{{background:#ca6f1e}}
.btn-sm{{padding:4px 10px;font-size:12px;border-radius:5px;border:none;cursor:pointer;font-weight:500}}
.btn-fill{{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}}
.btn-fill:hover{{background:#c8e6c9}}
.btn-del{{background:#fce4ec;color:#c62828;border:1px solid #ef9a9a}}
.btn-del:hover{{background:#ffcdd2}}
.btn:disabled{{opacity:.5;cursor:not-allowed}}
/* 模式标签 */
.mode-tabs{{display:flex;gap:0;border:1.5px solid #2980b9;border-radius:8px;overflow:hidden;width:fit-content}}
.mode-tab{{padding:7px 24px;border:none;background:#fff;color:#2980b9;font-size:14px;font-weight:600;cursor:pointer;transition:all .15s}}
.mode-tab.active{{background:#2980b9;color:#fff}}
/* 徽章 */
.badge-single{{background:#e3f2fd;color:#1565c0}}
.badge-complex{{background:#fce4ec;color:#b71c1c}}
/* 表单 */
.form-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;align-items:flex-end}}
.form-group{{display:flex;flex-direction:column;gap:4px}}
.form-group label{{font-size:12px;color:#666;font-weight:500}}
.form-group input{{border:1.5px solid #ddd;border-radius:6px;padding:7px 10px;font-size:13px;width:100%}}
.form-group input:focus{{outline:none;border-color:#2980b9}}
.ball-input{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.ball-input input{{width:48px;text-align:center;padding:6px 4px}}
/* 标签 */
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}}
.prize-win{{color:#27ae60;font-weight:600}}
.prize-none{{color:#999}}
/* 刷新状态 */
.refresh-status{{font-size:13px;color:#888;padding:4px 0}}
.spin{{display:inline-block;animation:spin 1s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
/* toast */
#toast{{position:fixed;bottom:30px;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:10px 22px;border-radius:20px;font-size:14px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:999}}
#toast.show{{opacity:1}}
/* 模态框 */
.modal-bg{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100;align-items:center;justify-content:center}}
.modal-bg.open{{display:flex}}
.modal{{background:#fff;border-radius:12px;padding:24px;width:min(520px,95vw);max-height:90vh;overflow-y:auto}}
.modal h3{{margin-bottom:16px;color:#2c3e50}}
.roi-row{{display:flex;gap:20px;margin-top:8px;font-size:13px}}
.roi-row span{{color:#666}}
.roi-row strong{{font-size:16px}}
</style>
</head>
<body>
<header>
  <h1>🎱 双色球预测系统</h1>
  <div class="header-right">
    <span class="refresh-status" id="refresh-status">上次刷新：{snap.get("max_date","—")}</span>
    <a href="/recommend" style="padding:7px 14px;background:rgba(255,255,255,0.2);color:#fff;border-radius:6px;text-decoration:none;font-size:13px">🎯 智能推荐</a>
    <a href="/trend" style="padding:7px 14px;background:rgba(255,255,255,0.2);color:#fff;border-radius:6px;text-decoration:none;font-size:13px">📊 走势图</a>
    <a href="/hot-cold" style="padding:7px 14px;background:rgba(255,255,255,0.2);color:#fff;border-radius:6px;text-decoration:none;font-size:13px">🔥 冷热分析</a>
    <a href="/stats-analysis" style="padding:7px 14px;background:rgba(255,255,255,0.2);color:#fff;border-radius:6px;text-decoration:none;font-size:13px">📈 统计分析</a>
    <a href="/cooccur" style="padding:7px 14px;background:rgba(255,255,255,0.2);color:#fff;border-radius:6px;text-decoration:none;font-size:13px">🔗 共现热图</a>
    <button class="btn btn-primary" id="btn-refresh" onclick="doRefresh()">
      <span id="refresh-icon">⟳</span> 刷新数据
    </button>
  </div>
</header>

<main>
<div class="disclaimer">
  ⚠️ <strong>免责声明：</strong>本系统仅供概率统计研究和学术学习，不构成任何投资建议。
  彩票开奖具有高度随机性，请理性购彩，量力而行，切勿沉迷。
</div>

<!-- 统计卡片 -->
<div class="stat-grid">
  <div class="stat-box"><div class="val" id="stat-total">{snap.get("total","—")}</div><div class="lbl">历史期数</div></div>
  <div class="stat-box"><div class="val" id="stat-latest">{snap.get("max_issue","—")}</div><div class="lbl">最新期号</div></div>
  <div class="stat-box"><div class="val">{len(purchases)}</div><div class="lbl">购彩记录</div></div>
  <div class="stat-box"><div class="val {roi_cls}" id="stat-roi">{roi:+.1f}%</div><div class="lbl">投资回报率</div></div>
</div>

<div class="grid2">
  <!-- 左：最近10期 -->
  <div>
    <h2>📅 最近开奖</h2>
    <div class="card" style="padding:0;overflow:hidden">
      <table>
        <thead><tr><th>期号</th><th>日期</th><th>红球</th><th>蓝球</th></tr></thead>
        <tbody id="latest-tbody">{latest_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- 右：频率 -->
  <div>
    <h2>🔥 红球频率 Top10</h2>
    <div class="card"><span id="top-red-balls">{top_red_html}</span></div>
    <h2>🔵 蓝球频率 Top5</h2>
    <div class="card"><span id="top-blue-balls">{top_blue_html}</span></div>
  </div>
</div>

<section id="pred-section"{' style="display:none"' if not pred_html else ''}>
  <h2>🎯 最新预测</h2>
  <div class="card" style="position:relative">
    <pre id="pred-text" style="white-space:pre-wrap;font-size:12px;line-height:1.9;color:#2c3e50;font-family:'PingFang SC',monospace">{pred_html}</pre>
    <a href="/recommend" style="position:absolute;top:12px;right:14px;padding:5px 14px;background:linear-gradient(135deg,#c0392b,#8e44ad);color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:bold">查看完整AI推荐 →</a>
  </div>
</section>

<!-- 购彩录入 -->
<h2>📝 录入本次购彩</h2>
<div class="card">
  <!-- 模式选择 -->
  <div class="mode-tabs">
    <button class="mode-tab active" id="tab-single" onclick="switchMode('单式')">单式</button>
    <button class="mode-tab" id="tab-complex" onclick="switchMode('复式')">复式</button>
  </div>

  <!-- 期号 -->
  <div class="form-row" style="margin-top:14px">
    <div class="form-group">
      <label>期号</label>
      <input id="f-issue" type="text" placeholder="{snap.get('max_issue','26040')}" style="width:110px">
    </div>
    <!-- 注数（单式手填；复式自动计算） -->
    <div class="form-group" id="tickets-group">
      <label>购买注数</label>
      <input id="f-tickets" type="number" value="1" min="1" style="width:80px">
    </div>
    <!-- 复式规格选择 -->
    <div class="form-group" id="complex-spec" style="display:none">
      <label>红球数量</label>
      <select id="f-nred" onchange="buildInputs()" style="padding:7px 8px;border:1.5px solid #ddd;border-radius:6px;font-size:13px">
        <option value="7">7红</option>
        <option value="8">8红</option>
        <option value="9">9红</option>
        <option value="10">10红</option>
        <option value="11">11红</option>
        <option value="12">12红</option>
      </select>
    </div>
    <div class="form-group" id="complex-spec-blue" style="display:none">
      <label>蓝球数量</label>
      <select id="f-nblue" onchange="buildInputs()" style="padding:7px 8px;border:1.5px solid #ddd;border-radius:6px;font-size:13px">
        <option value="1">1蓝</option>
        <option value="2">2蓝</option>
        <option value="3">3蓝</option>
        <option value="4">4蓝</option>
        <option value="5">5蓝</option>
      </select>
    </div>
    <div class="form-group" id="complex-cost-display" style="display:none">
      <label>自动计算</label>
      <div id="complex-cost" style="padding:7px 12px;background:#f0f7ff;border-radius:6px;font-size:13px;color:#1565c0;font-weight:600">— 注 / ¥—</div>
    </div>
  </div>

  <!-- 红球输入区 -->
  <div class="form-group" style="margin-bottom:12px">
    <label id="red-label">选择红球（6个，1-33）</label>
    <div class="ball-input" id="red-inputs"></div>
  </div>

  <!-- 蓝球输入区 -->
  <div class="form-group" style="margin-bottom:14px">
    <label id="blue-label">选择蓝球（1个，1-16）</label>
    <div class="ball-input" id="blue-inputs"></div>
  </div>

  <div class="form-row">
    <button class="btn btn-success" onclick="addPurchase()">✚ 保存购彩记录</button>
    <span id="form-hint" style="font-size:12px;color:#888;align-self:center"></span>
  </div>
</div>

<!-- 购彩历史 -->
<h2>📊 购彩历史（最近30条）</h2>
<div class="card" style="padding:0;overflow:hidden">
  <table>
    <thead>
      <tr>
        <th>时间</th><th>期号</th><th>我的红球</th><th>我的蓝球</th>
        <th>注数/金额</th><th>开奖号码</th><th>命中</th><th>奖级</th><th>操作</th>
      </tr>
    </thead>
    <tbody id="purchase-tbody">
      {purchase_rows if purchase_rows else '<tr><td colspan="9" style="color:#bbb;padding:20px">暂无记录</td></tr>'}
    </tbody>
  </table>
</div>

<!-- 汇总行 -->
<div class="card" style="margin-top:0;border-radius:0 0 10px 10px;padding:10px 22px">
  <div class="roi-row">
    <span>累计投入：<strong>¥{total_cost:,}</strong></span>
    <span>累计获奖：<strong class="prize-win">¥{total_prize:,}</strong></span>
    <span>投资回报率：<strong class="{roi_cls}">{roi:+.1f}%</strong></span>
    <span style="color:#999;font-size:12px">（随机期望约 -50%）</span>
  </div>
</div>

<p style="text-align:center;color:#bbb;font-size:12px;margin-top:20px">双色球预测系统 · 仅供研究 · 请理性购彩</p>
</main>

<!-- 录入开奖结果 模态框 -->
<div class="modal-bg" id="result-modal">
  <div class="modal">
    <h3>录入开奖结果</h3>
    <input type="hidden" id="modal-pid">
    <div class="form-row" style="margin-bottom:8px">
      <div class="form-group"><label>期号</label><input id="modal-issue" readonly style="width:100px;background:#f5f5f5"></div>
    </div>
    <div class="form-group" style="margin-bottom:12px">
      <label>开奖红球（6个，1-33）</label>
      <div class="ball-input">
        <input class="actual-red" type="number" min="1" max="33" placeholder="01">
        <input class="actual-red" type="number" min="1" max="33" placeholder="02">
        <input class="actual-red" type="number" min="1" max="33" placeholder="03">
        <input class="actual-red" type="number" min="1" max="33" placeholder="04">
        <input class="actual-red" type="number" min="1" max="33" placeholder="05">
        <input class="actual-red" type="number" min="1" max="33" placeholder="06">
      </div>
    </div>
    <div class="form-group" style="margin-bottom:16px">
      <label>开奖蓝球（1-16）</label>
      <input id="modal-blue" type="number" min="1" max="16" placeholder="01" style="width:60px">
    </div>
    <div style="display:flex;gap:10px">
      <button class="btn btn-success" onclick="submitResult()">✓ 确认提交</button>
      <button class="btn" style="background:#eee;color:#555" onclick="closeModal()">取消</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
// ── 工具 ─────────────────────────────────────────────
function toast(msg, ms=2500){{
  const el=document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'), ms);
}}
function post(url, data){{
  return fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(data)}}).then(r=>r.json());
}}

// ── 应用快照数据到 DOM ────────────────────────────────
function applySnapshot(d){{
  if(d.total) document.getElementById('stat-total').textContent=d.total;
  if(d.max_issue) document.getElementById('stat-latest').textContent=d.max_issue;
  if(d.latest_html){{
    const tb=document.getElementById('latest-tbody');
    if(tb) tb.innerHTML=d.latest_html;
  }}
  if(d.top_red_html){{
    const el=document.getElementById('top-red-balls');
    if(el) el.innerHTML=d.top_red_html;
  }}
  if(d.top_blue_html){{
    const el=document.getElementById('top-blue-balls');
    if(el) el.innerHTML=d.top_blue_html;
  }}
  if(d.pred_html){{
    const pt=document.getElementById('pred-text');
    if(pt){{
      pt.textContent=d.pred_html;
      const sec=document.getElementById('pred-section');
      if(sec) sec.style.display='';
    }} else {{
      // 首次注入预测区块
      const main=document.querySelector('main')||document.body;
      const pcard=document.getElementById('pred-card-wrap');
      if(!pcard){{
        const wrap=document.createElement('section');
        wrap.id='pred-section';
        wrap.innerHTML='<h2>🎯 最新预测</h2><div class="card"><pre id="pred-text" style="white-space:pre-wrap;font-size:12px;line-height:1.8;color:#2c3e50">'+d.pred_html+'</pre></div>';
        const purchaseSection=document.querySelector('h2');
        if(purchaseSection) main.insertBefore(wrap,purchaseSection);
        else main.prepend(wrap);
      }}
    }}
  }}
  if(d.max_date){{
    const el=document.getElementById('refresh-status');
    if(el && !el.textContent.includes('刷新')) el.textContent='数据截至：'+d.max_date;
  }}
}}

// ── 页面加载时拉取实时快照 ────────────────────────────
(function loadSnapshot(){{
  fetch('/api/snapshot').then(r=>r.json()).then(applySnapshot).catch(()=>{{}});
}})();

// ── 刷新数据 ─────────────────────────────────────────
function doRefresh(){{
  const btn=document.getElementById('btn-refresh');
  const icon=document.getElementById('refresh-icon');
  const status=document.getElementById('refresh-status');
  btn.disabled=true;
  icon.textContent='⏳'; icon.classList.add('spin');
  status.textContent='正在拉取最新开奖数据及重新计算预测…';
  fetch('/api/refresh',{{method:'POST'}})
    .then(r=>r.json())
    .then(d=>{{
      if(d.ok){{
        let msg='刷新完成 ✓  新增 '+d.new_periods+' 期，最新期号 '+d.max_issue;
        if(d.auto_matched>0) msg+='，自动核销 '+d.auto_matched+' 条';
        status.textContent=msg;
        // 动态更新所有区块（无需整页刷新）
        applySnapshot(d);
        toast('✓ 数据及预测已更新，新增 '+d.new_periods+' 期'+(d.auto_matched>0?' | 核销 '+d.auto_matched+' 条':''));
        if(d.auto_matched>0) setTimeout(()=>location.reload(),1500);
      }}else{{
        status.textContent='刷新失败：'+d.msg;
        toast('⚠ '+d.msg, 3500);
      }}
    }})
    .catch(e=>{{ status.textContent='网络错误'; toast('⚠ 网络错误: '+e,3500); }})
    .finally(()=>{{ btn.disabled=false; icon.textContent='⟳'; icon.classList.remove('spin'); }});
}}

// ── 购彩模式：单式 / 复式 ──────────────────────────────
let _mode = '单式';

function C(n, k) {{
  if(k>n) return 0;
  let r=1; for(let i=0;i<k;i++) r=r*(n-i)/(i+1); return Math.round(r);
}}

function buildInputs() {{
  const nRed = _mode==='复式' ? parseInt(document.getElementById('f-nred').value) : 6;
  const nBlue = _mode==='复式' ? parseInt(document.getElementById('f-nblue').value) : 1;

  // 红球格子
  const ri = document.getElementById('red-inputs');
  ri.innerHTML = '';
  for(let i=0;i<nRed;i++) {{
    const inp = document.createElement('input');
    inp.className='my-red'; inp.type='number'; inp.min=1; inp.max=33;
    inp.placeholder=String(i+1).padStart(2,'0');
    inp.addEventListener('input', updateComplexSpec);
    ri.appendChild(inp);
  }}

  // 蓝球格子
  const bi = document.getElementById('blue-inputs');
  bi.innerHTML = '';
  for(let i=0;i<nBlue;i++) {{
    const inp = document.createElement('input');
    inp.className='my-blue-in'; inp.type='number'; inp.min=1; inp.max=16;
    inp.placeholder=String(i+1).padStart(2,'0');
    ri.style.flexWrap='wrap';
    bi.appendChild(inp);
  }}

  document.getElementById('red-label').textContent =
    _mode==='复式' ? `选择红球（${{nRed}}个，1-33）` : '选择红球（6个，1-33）';
  document.getElementById('blue-label').textContent =
    _mode==='复式' ? `选择蓝球（${{nBlue}}个，1-16）` : '选择蓝球（1个，1-16）';

  updateComplexSpec();
}}

function updateComplexSpec() {{
  if(_mode !== '复式') return;
  const nRed = parseInt(document.getElementById('f-nred').value);
  const nBlue = parseInt(document.getElementById('f-nblue').value);
  const combos = C(nRed,6) * nBlue;
  const cost = combos * 2;
  document.getElementById('complex-cost').textContent = `${{combos}} 注 / ¥${{cost}}`;
  document.getElementById('f-tickets').value = combos;
  document.getElementById('form-hint').textContent = `C(${{nRed}},6)×${{nBlue}} = ${{combos}} 注，¥${{cost}}`;
}}

function switchMode(mode) {{
  _mode = mode;
  document.getElementById('tab-single').classList.toggle('active', mode==='单式');
  document.getElementById('tab-complex').classList.toggle('active', mode==='复式');

  const isComplex = mode==='复式';
  document.getElementById('tickets-group').style.display = isComplex ? 'none' : '';
  document.getElementById('complex-spec').style.display = isComplex ? '' : 'none';
  document.getElementById('complex-spec-blue').style.display = isComplex ? '' : 'none';
  document.getElementById('complex-cost-display').style.display = isComplex ? '' : 'none';
  document.getElementById('form-hint').textContent = '';

  buildInputs();
}}

// 初始化单式输入
switchMode('单式');

// ── 录入购彩 ─────────────────────────────────────────
function addPurchase() {{
  const issue = document.getElementById('f-issue').value.trim();
  if(!issue) {{ toast('⚠ 请填写期号'); return; }}

  const reds = [...document.querySelectorAll('.my-red')].map(i=>parseInt(i.value));
  const blues = [...document.querySelectorAll('.my-blue-in')].map(i=>parseInt(i.value));

  // 验证红球
  const nExpRed = _mode==='复式' ? parseInt(document.getElementById('f-nred').value) : 6;
  if(reds.some(isNaN) || reds.length!==nExpRed) {{
    toast(`⚠ 请填写完整的${{nExpRed}}个红球`); return;
  }}
  if(new Set(reds).size!==nExpRed) {{ toast('⚠ 红球号码不能重复'); return; }}
  if(reds.some(n=>n<1||n>33)) {{ toast('⚠ 红球范围 1-33'); return; }}

  // 验证蓝球
  const nExpBlue = _mode==='复式' ? parseInt(document.getElementById('f-nblue').value) : 1;
  if(blues.some(isNaN) || blues.length!==nExpBlue) {{
    toast(`⚠ 请填写完整的${{nExpBlue}}个蓝球`); return;
  }}
  if(new Set(blues).size!==nExpBlue) {{ toast('⚠ 蓝球号码不能重复'); return; }}
  if(blues.some(n=>n<1||n>16)) {{ toast('⚠ 蓝球范围 1-16'); return; }}

  const tickets = parseInt(document.getElementById('f-tickets').value) || 1;
  const my_blue = _mode==='单式' ? blues[0] : blues;

  post('/api/purchase', {{
    issue, tickets, type: _mode,
    my_red: reds, my_blue
  }}).then(d=>{{
    if(d.ok) {{ toast('✓ 购彩记录已保存'); location.reload(); }}
    else toast('⚠ '+d.msg, 3000);
  }});
}}

// ── 录入开奖结果 ──────────────────────────────────────
function fillResult(pid, issue) {{
  document.getElementById('modal-pid').value=pid;
  document.getElementById('modal-issue').value=issue;
  document.querySelectorAll('.actual-red').forEach(i=>i.value='');
  document.getElementById('modal-blue').value='';
  document.getElementById('result-modal').classList.add('open');
}}
function closeModal() {{ document.getElementById('result-modal').classList.remove('open'); }}

function submitResult() {{
  const pid = document.getElementById('modal-pid').value;
  const reds = [...document.querySelectorAll('.actual-red')].map(i=>parseInt(i.value));
  const blue = parseInt(document.getElementById('modal-blue').value);

  if(reds.some(isNaN)||reds.length!==6) {{ toast('⚠ 请填写完整的6个开奖红球'); return; }}
  if(new Set(reds).size!==6) {{ toast('⚠ 红球不能重复'); return; }}
  if(reds.some(n=>n<1||n>33)) {{ toast('⚠ 红球范围 1-33'); return; }}
  if(isNaN(blue)||blue<1||blue>16) {{ toast('⚠ 蓝球范围 1-16'); return; }}

  post('/api/result', {{id:pid, actual_red:reds, actual_blue:blue}})
    .then(d=>{{
      if(d.ok) {{
        closeModal();
        const msg = d.combo_count > 1
          ? `✓ 复式共${{d.combo_count}}注，中奖${{d.winning_count}}注 → ${{d.prize}}（¥${{d.prize_money}}）`
          : `✓ 命中红${{d.hit_red}}个 蓝${{d.hit_blue?'✓':'✗'}} → ${{d.prize}}`;
        toast(msg, 3000);
        setTimeout(()=>location.reload(), 2200);
      }} else toast('⚠ '+d.msg, 3000);
    }});
}}

// ── 删除记录 ─────────────────────────────────────────
function deletePurchase(pid) {{
  if(!confirm('确认删除该记录？')) return;
  post('/api/delete', {{id:pid}})
    .then(d=>{{ if(d.ok) location.reload(); else toast('⚠ '+d.msg); }});
}}
</script>
</body>
</html>"""

    # ── 走势图数据 API ────────────────────────────────────
    def get_trend_data(n: int = 30, start_issue: str = "", end_issue: str = "") -> dict:
        """返回走势图所需数据：每期号码、遗漏值、统计。"""
        _mgr = DataManager()
        full_df = _mgr.load_local()
        if full_df is None or full_df.empty:
            return {"error": "no data"}

        full_df = full_df.sort_values("issue").reset_index(drop=True)

        if start_issue and end_issue:
            df = full_df[(full_df["issue"] >= start_issue) & (full_df["issue"] <= end_issue)]
        else:
            df = full_df.tail(n)

        df = df.reset_index(drop=True)

        red_cols = [f"r{i}" for i in range(1, 7)]
        rows = []
        # 遗漏计数器
        red_miss = {b: 0 for b in range(1, 34)}
        blue_miss = {b: 0 for b in range(1, 17)}

        # 先扫描 df 之前的数据来初始化遗漏（取全量最后 500 期）
        full_tail = full_df
        if full_tail is not None and not full_tail.empty:
            full_tail = full_tail.sort_values("issue")
            pre = full_tail[full_tail["issue"] < df.iloc[0]["issue"]].tail(500)
            for _, pr in pre.iterrows():
                drawn_red = set(int(pr[c]) for c in red_cols)
                drawn_blue = int(pr["blue"])
                for b in range(1, 34):
                    red_miss[b] = 0 if b in drawn_red else red_miss[b] + 1
                for b in range(1, 17):
                    blue_miss[b] = 0 if b == drawn_blue else blue_miss[b] + 1

        for _, row in df.iterrows():
            drawn_red = sorted(int(row[c]) for c in red_cols)
            drawn_blue = int(row["blue"])
            drawn_red_set = set(drawn_red)

            # 记录当前遗漏（开奖前）
            row_red_miss = dict(red_miss)
            row_blue_miss = dict(blue_miss)

            # 更新遗漏
            for b in range(1, 34):
                red_miss[b] = 0 if b in drawn_red_set else red_miss[b] + 1
            for b in range(1, 17):
                blue_miss[b] = 0 if b == drawn_blue else blue_miss[b] + 1

            rows.append({
                "issue": str(row["issue"]),
                "date": str(row.get("date", "")),
                "red": drawn_red,
                "blue": drawn_blue,
                "red_miss": row_red_miss,
                "blue_miss": row_blue_miss,
            })

        # 统计
        issues_list = [r["issue"] for r in rows]
        red_appear = {b: sum(1 for r in rows if b in r["red"]) for b in range(1, 34)}
        blue_appear = {b: sum(1 for r in rows if b == r["blue"]) for b in range(1, 17)}

        red_miss_vals = {b: [r["red_miss"][b] for r in rows] for b in range(1, 34)}
        blue_miss_vals = {b: [r["blue_miss"][b] for r in rows] for b in range(1, 17)}

        def _avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0
        def _max_consec(lst_drawn):
            mx = cur = 0
            for v in lst_drawn:
                cur = cur + 1 if v else 0
                mx = max(mx, cur)
            return mx

        red_stats = {}
        for b in range(1, 34):
            drawn_seq = [1 if b in r["red"] else 0 for r in rows]
            red_stats[b] = {
                "appear": red_appear[b],
                "avg_miss": _avg(red_miss_vals[b]),
                "max_miss": max(red_miss_vals[b]) if red_miss_vals[b] else 0,
                "max_consec": _max_consec(drawn_seq),
            }
        blue_stats = {}
        for b in range(1, 17):
            drawn_seq = [1 if r["blue"] == b else 0 for r in rows]
            blue_stats[b] = {
                "appear": blue_appear[b],
                "avg_miss": _avg(blue_miss_vals[b]),
                "max_miss": max(blue_miss_vals[b]) if blue_miss_vals[b] else 0,
                "max_consec": _max_consec(drawn_seq),
            }

        return {
            "rows": rows,
            "red_stats": {str(k): v for k, v in red_stats.items()},
            "blue_stats": {str(k): v for k, v in blue_stats.items()},
            "period_count": len(rows),
        }

    # ── 走势图页面 ────────────────────────────────────────
    def render_trend_html() -> str:
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>双色球走势图</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f5f5;font-size:12px;color:#333}
.top-bar{background:#fff;border-bottom:1px solid #ddd;padding:6px 10px;display:flex;align-items:center;flex-wrap:wrap;gap:6px;position:sticky;top:0;z-index:100}
.top-bar a{color:#c0392b;font-weight:bold;text-decoration:none;padding:3px 8px;border-radius:3px}
.top-bar a:hover,.top-bar a.active{background:#c0392b;color:#fff}
.sep{color:#ccc;padding:0 2px}
.period-btn{padding:3px 10px;border:1px solid #ccc;border-radius:3px;cursor:pointer;background:#fff;font-size:12px}
.period-btn.active,.period-btn:hover{background:#c0392b;color:#fff;border-color:#c0392b}
.period-input{border:1px solid #ccc;border-radius:3px;padding:3px 6px;width:72px;font-size:12px;text-align:center}
.query-btn{padding:3px 10px;background:#e74c3c;color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:12px}
.check-bar{background:#fff;padding:5px 10px;border-bottom:1px solid #eee;display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:12px}
.check-bar label{display:flex;align-items:center;gap:3px;cursor:pointer}
.wrapper{overflow-x:auto;padding:0}
table{border-collapse:collapse;background:#fff;white-space:nowrap;min-width:100%}
th,td{border:1px solid #e0e0e0;padding:0;text-align:center;height:22px;min-width:22px}
thead th{background:#f0e6e6;color:#333;font-weight:bold;height:24px;font-size:11px;position:sticky;top:86px;z-index:10}
thead th.blue-head{background:#e6eef8}
th.row-head{background:#f7f7f7;min-width:58px;position:sticky;left:0;z-index:11}
thead th.row-head{z-index:20}
td.row-head{background:#f7f7f7;font-size:11px;position:sticky;left:0}
.ball{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;font-size:11px;font-weight:bold;color:#fff;background:#c0392b}
.ball.blue{background:#2980b9}
.miss{color:#666;font-size:11px}
.miss.hot{color:#e74c3c;font-weight:bold}
/* 遗漏热度背景 */
td.l1{background:#fff5f5} td.l2{background:#ffe0e0} td.l3{background:#ffcccc}
td.b1{background:#f0f6ff} td.b2{background:#d6e9ff} td.b3{background:#b8d4ff}
/* 走势线 SVG */
.trend-cell{position:relative;overflow:visible}
svg.tline{position:absolute;pointer-events:none;overflow:visible;top:0;left:0;width:100%;height:100%}
/* 统计行 */
tr.stat-row td{background:#fafafa;font-size:11px;color:#555}
tr.stat-row.head td{background:#f0e6e6;font-weight:bold;color:#333}
tr.stat-row.blue-stat td{background:#e6eef8}
/* 模拟选号 */
tr.sim-row td{background:#fffde7;font-size:11px}
tr.sim-row td.row-head{background:#fff9c4}
.sim-ball{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;border-radius:50%;font-size:10px;font-weight:bold;color:#fff;background:#c0392b;margin:0 1px}
.sim-ball.blue{background:#2980b9}
.home-btn{margin-left:auto;padding:3px 10px;background:#27ae60;color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:12px;text-decoration:none}
</style>
</head>
<body>

<div class="top-bar">
  <a href="/" style="background:#27ae60;color:#fff">← 主页</a>
  <span class="sep">|</span>
  <a href="#" class="active" id="tab-redblue" onclick="setView('redblue',this)">红蓝走势</a>
  <a href="#" id="tab-zone3" onclick="setView('zone3',this)">红球三分区走势</a>
  <a href="#" id="tab-zone4" onclick="setView('zone4',this)">红球四分区走势</a>
  <a href="#" id="tab-zone7" onclick="setView('zone7',this)">红球七分区走势</a>
  <span class="sep" style="flex:1"></span>
  <button class="period-btn active" id="btn30" onclick="setPeriod(30,this)">最近30期</button>
  <button class="period-btn" id="btn50" onclick="setPeriod(50,this)">最近50期</button>
  <button class="period-btn" id="btn100" onclick="setPeriod(100,this)">最近100期</button>
  <input class="period-input" id="inp-start" placeholder="起始期号">
  <span>期 至</span>
  <input class="period-input" id="inp-end" placeholder="结束期号">
  <span>期</span>
  <button class="query-btn" onclick="queryRange()">查 看</button>
</div>

<div class="check-bar">
  <span>标注形式选择：</span>
  <label><input type="checkbox" id="chk-nomiss" onchange="renderTable()"> 不带遗漏数据</label>
  <label><input type="checkbox" id="chk-repeat" onchange="renderTable()"> 重号</label>
  <label><input type="checkbox" id="chk-consec" onchange="renderTable()"> 连号</label>
  <label><input type="checkbox" id="chk-edge" onchange="renderTable()"> 边号</label>
  <label><input type="checkbox" id="chk-layer" onchange="renderTable()"> 遗漏分层</label>
  <label><input type="checkbox" id="chk-line" checked onchange="renderTable()"> 辅助线</label>
</div>

<div class="wrapper">
  <div id="table-container">加载中…</div>
</div>

<script>
let _data = null;
let _period = 30;
let _view = 'redblue';
let _prevIssueRed = {};   // 上一行各红球位置 {ball: colIndex}
let _prevIssueBlue = {};  // 上一行蓝球位置

// 区间定义
const ZONES3 = [[1,11],[12,22],[23,33]];
const ZONES4 = [[1,8],[9,16],[17,25],[26,33]];
const ZONES7 = [[1,5],[6,10],[11,16],[17,21],[22,27],[28,30],[31,33]];

function setView(v, el) {
  _view = v;
  document.querySelectorAll('.top-bar a[id^=tab]').forEach(a=>a.classList.remove('active'));
  el.classList.add('active');
  renderTable();
}

function setPeriod(n, el) {
  _period = n;
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('inp-start').value='';
  document.getElementById('inp-end').value='';
  loadData();
}

function queryRange() {
  const s = document.getElementById('inp-start').value.trim();
  const e = document.getElementById('inp-end').value.trim();
  if(!s||!e){alert('请填写起止期号');return;}
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  fetch('/api/trend-data?start='+s+'&end='+e)
    .then(r=>r.json()).then(d=>{_data=d;renderTable();});
}

function loadData() {
  fetch('/api/trend-data?n='+_period)
    .then(r=>r.json()).then(d=>{_data=d;renderTable();});
}

// ── 主渲染 ────────────────────────────────────────────
function renderTable() {
  if(!_data||_data.error){document.getElementById('table-container').textContent='暂无数据';return;}
  const nomiss = document.getElementById('chk-nomiss').checked;
  const showLine = document.getElementById('chk-line').checked;
  const showRepeat = document.getElementById('chk-repeat').checked;
  const showConsec = document.getElementById('chk-consec').checked;
  const showEdge = document.getElementById('chk-edge').checked;
  const showLayer = document.getElementById('chk-layer').checked;

  const rows = _data.rows;
  const rs = _data.red_stats;
  const bs = _data.blue_stats;

  // 确定列定义
  let redCols, blueVisible;
  if(_view==='zone3'){
    redCols = ZONES3.map(z=>range(z[0],z[1]));
    blueVisible = false;
  } else if(_view==='zone4'){
    redCols = ZONES4.map(z=>range(z[0],z[1]));
    blueVisible = false;
  } else if(_view==='zone7'){
    redCols = ZONES7.map(z=>range(z[0],z[1]));
    blueVisible = false;
  } else {
    redCols = [range(1,33)];
    blueVisible = true;
  }
  const flatRed = redCols.flat();
  const blueCols = blueVisible ? range(1,16) : [];

  // 找重号（当前期与上一期有相同红球）
  const repeatSet = new Set();
  if(showRepeat){
    for(let i=1;i<rows.length;i++){
      const prev = new Set(rows[i-1].red);
      rows[i].red.forEach(b=>{if(prev.has(b)) repeatSet.add(rows[i].issue+'_'+b);});
    }
  }
  // 找连号（同一期内相邻号）
  const consecSet = new Set();
  if(showConsec){
    rows.forEach(r=>{
      const sorted = [...r.red].sort((a,b)=>a-b);
      for(let i=1;i<sorted.length;i++){
        if(sorted[i]-sorted[i-1]===1){
          consecSet.add(r.issue+'_'+sorted[i-1]);
          consecSet.add(r.issue+'_'+sorted[i]);
        }
      }
    });
  }
  // 边号 1,2,32,33
  const edgeBalls = new Set([1,2,32,33]);

  // 前一期红蓝位置（用于画线）
  let prevRedPos = {};   // ball -> colIndex in flatRed
  let prevBluePos = null; // ball number

  let html = '<table id="trend-table">';

  // ── 表头 ────────────────────────────────────────────
  html += '<thead><tr>';
  html += '<th class="row-head">期号</th>';
  if(_view==='redblue'){
    flatRed.forEach(b=>{html+=`<th>${b}</th>`;});
    blueCols.forEach(b=>{html+=`<th class="blue-head">${b}</th>`;});
  } else {
    redCols.forEach((zone,zi)=>{
      const zoneNames=['一区','二区','三区','四区','五区','六区','七区'];
      html+=`<th colspan="${zone.length}" style="background:#f0e6e6">${zoneNames[zi]||''}</th>`;
    });
    html += '<th colspan="0"></th>'; // placeholder fix
  }
  if(_view!=='redblue'){
    // zone view: individual cols
    html = '<table id="trend-table"><thead><tr><th class="row-head">期号</th>';
    flatRed.forEach(b=>{html+=`<th>${b}</th>`;});
    html += '</tr></thead>';
  } else {
    html += '</tr></thead>';
  }

  // ── 数据行 ──────────────────────────────────────────
  html += '<tbody>';
  rows.forEach((row,ri)=>{
    const drawnRed = new Set(row.red);
    const drawnBlue = row.blue;
    html += `<tr>`;
    html += `<td class="row-head">${row.issue}</td>`;

    // 红球列
    flatRed.forEach((b,ci)=>{
      const hit = drawnRed.has(b);
      const issKey = row.issue+'_'+b;
      let extra='', tdClass='';

      if(hit){
        let style='background:#c0392b';
        if(showRepeat && repeatSet.has(issKey)) style='background:#8e44ad';
        if(showConsec && consecSet.has(issKey)) style='background:#e67e22';
        if(showEdge && edgeBalls.has(b)) style='background:#16a085';
        // 画线标记
        extra=`<span class="ball" style="${style};position:relative;z-index:2">${b}</span>`;
        if(showLine){
          // 用 data 属性，JS 后处理画 SVG 线
          extra=`<span class="ball rball" data-ball="${b}" data-row="${ri}" data-col="${ci}" style="${style};position:relative;z-index:2">${b}</span>`;
        }
      } else {
        const mv = row.red_miss[b]||0;
        if(nomiss){extra='';tdClass='';}
        else {
          extra=`<span class="miss${mv>10?' hot':''}">${mv||''}</span>`;
          if(showLayer){
            const maxM = rs[b]?rs[b].max_miss:0;
            if(maxM>0){
              const ratio = mv/maxM;
              tdClass = ratio>0.7?'l3':ratio>0.4?'l2':ratio>0.15?'l1':'';
            }
          }
        }
      }
      html+=`<td class="${tdClass} trend-cell">${extra}</td>`;
    });

    // 蓝球列（仅红蓝走势模式）
    if(blueVisible){
      blueCols.forEach((b,ci)=>{
        const hit = (b===drawnBlue);
        let extra='', tdClass='';
        if(hit){
          extra=`<span class="ball blue bball" data-ball="${b}" data-row="${ri}" data-col="${ci}">${b}</span>`;
        } else {
          const mv = row.blue_miss[b]||0;
          if(nomiss){extra='';}
          else {
            extra=`<span class="miss${mv>8?' hot':''}">${mv||''}</span>`;
            if(showLayer){
              const maxM = bs[b]?bs[b].max_miss:0;
              if(maxM>0){
                const ratio=mv/maxM;
                tdClass=ratio>0.7?'b3':ratio>0.4?'b2':ratio>0.15?'b1':'';
              }
            }
          }
        }
        html+=`<td class="${tdClass} trend-cell">${extra}</td>`;
      });
    }
    html+='</tr>';
  });

  // ── 模拟选号（5组） ──────────────────────────────────
  const sims = generateSims(5, rows);
  sims.forEach((sim,si)=>{
    html+=`<tr class="sim-row"><td class="row-head">模拟选号${['一','二','三','四','五'][si]}</td>`;
    flatRed.forEach(b=>{
      const hit=sim.red.includes(b);
      html+=`<td>${hit?`<span class="sim-ball">${b}</span>`:''}</td>`;
    });
    if(blueVisible){
      blueCols.forEach(b=>{
        const hit=(b===sim.blue);
        html+=`<td>${hit?`<span class="sim-ball blue">${b}</span>`:''}</td>`;
      });
    }
    html+='</tr>';
  });

  // 您选择了提示
  html+=`<tr class="sim-row"><td class="row-head" colspan="${flatRed.length+(blueVisible?blueCols.length:0)+1}" style="text-align:left;padding:4px 8px;color:#666">`;
  sims.forEach((sim,si)=>{
    html+=`您选了：模拟${['一','二','三','四','五'][si]}（${sim.red.join('+')}红+${sim.blue}蓝）&nbsp;&nbsp;`;
  });
  html+='</td></tr>';

  // ── 统计区 ───────────────────────────────────────────
  const statLabels=['出现总次数','平均遗漏值','最大遗漏值','最大连出值'];
  const statKeys=['appear','avg_miss','max_miss','max_consec'];
  statLabels.forEach((label,li)=>{
    const k=statKeys[li];
    html+=`<tr class="stat-row${li===0?' head':''}"><td class="row-head">${label}</td>`;
    flatRed.forEach(b=>{html+=`<td>${rs[b]?rs[b][k]:''}</td>`;});
    if(blueVisible){
      blueCols.forEach(b=>{html+=`<td>${bs[b]?bs[b][k]:''}</td>`;});
    }
    html+='</tr>';
  });

  html+='</tbody></table>';
  document.getElementById('table-container').innerHTML=html;

  // ── 画蓝球走势线 ─────────────────────────────────────
  if(showLine && blueVisible) drawBlueLine();
  if(showLine) drawRedLines();
}

function drawBlueLine(){
  const balls = document.querySelectorAll('.bball');
  if(balls.length<2) return;
  const tbl = document.getElementById('trend-table');
  const tblRect = tbl.getBoundingClientRect();

  // 创建 SVG overlay
  let svg = document.getElementById('blue-svg-overlay');
  if(!svg){
    svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.id='blue-svg-overlay';
    svg.style.cssText='position:absolute;pointer-events:none;overflow:visible;top:0;left:0;z-index:5';
    tbl.parentElement.style.position='relative';
    tbl.parentElement.appendChild(svg);
  }
  svg.innerHTML='';
  const wrapRect = tbl.parentElement.getBoundingClientRect();
  svg.setAttribute('width', wrapRect.width);
  svg.setAttribute('height', wrapRect.height);

  const pts = [];
  balls.forEach(el=>{
    const r = el.getBoundingClientRect();
    pts.push({
      x: r.left - wrapRect.left + r.width/2,
      y: r.top - wrapRect.top + r.height/2,
    });
  });

  for(let i=1;i<pts.length;i++){
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',pts[i-1].x);line.setAttribute('y1',pts[i-1].y);
    line.setAttribute('x2',pts[i].x);  line.setAttribute('y2',pts[i].y);
    line.setAttribute('stroke','#2980b9');line.setAttribute('stroke-width','1.2');
    line.setAttribute('opacity','0.7');
    svg.appendChild(line);
  }
}

function drawRedLines(){
  // 对每个红球号，连接相邻出现行
  const allBalls = document.querySelectorAll('.rball');
  if(allBalls.length<2) return;
  const tbl = document.getElementById('trend-table');
  let svg = document.getElementById('red-svg-overlay');
  if(!svg){
    svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.id='red-svg-overlay';
    svg.style.cssText='position:absolute;pointer-events:none;overflow:visible;top:0;left:0;z-index:4';
    tbl.parentElement.style.position='relative';
    tbl.parentElement.appendChild(svg);
  }
  svg.innerHTML='';
  const wrapRect = tbl.parentElement.getBoundingClientRect();
  svg.setAttribute('width', wrapRect.width);
  svg.setAttribute('height', wrapRect.height);

  // 按 ball 号分组
  const groups={};
  allBalls.forEach(el=>{
    const b=el.dataset.ball;
    if(!groups[b]) groups[b]=[];
    const r=el.getBoundingClientRect();
    groups[b].push({x:r.left-wrapRect.left+r.width/2, y:r.top-wrapRect.top+r.height/2});
  });
  Object.values(groups).forEach(pts=>{
    for(let i=1;i<pts.length;i++){
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1',pts[i-1].x);line.setAttribute('y1',pts[i-1].y);
      line.setAttribute('x2',pts[i].x);  line.setAttribute('y2',pts[i].y);
      line.setAttribute('stroke','#c0392b');line.setAttribute('stroke-width','0.8');
      line.setAttribute('opacity','0.35');
      svg.appendChild(line);
    }
  });
}

function range(a,b){const r=[];for(let i=a;i<=b;i++)r.push(i);return r;}

function generateSims(n, rows){
  // 基于近期频率加权随机选号
  const freq={};
  for(let b=1;b<=33;b++) freq[b]=1;
  const bfreq={};
  for(let b=1;b<=16;b++) bfreq[b]=1;
  rows.forEach(r=>{
    r.red.forEach(b=>freq[b]=(freq[b]||0)+2);
    bfreq[r.blue]=(bfreq[r.blue]||0)+2;
  });
  const sims=[];
  for(let i=0;i<n;i++){
    const red = weightedSample(freq,6,[1,33]);
    const blue = weightedSample(bfreq,1,[1,16])[0];
    sims.push({red:red.sort((a,b)=>a-b), blue});
  }
  return sims;
}

function weightedSample(weights, k, [lo,hi]){
  const pool=[];
  for(let b=lo;b<=hi;b++) pool.push({b, w:weights[b]||1});
  const chosen=[];
  const used=new Set();
  for(let i=0;i<k;i++){
    const available=pool.filter(p=>!used.has(p.b));
    const total=available.reduce((s,p)=>s+p.w,0);
    let r=Math.random()*total;
    for(const p of available){r-=p.w; if(r<=0){chosen.push(p.b);used.add(p.b);break;}}
    if(chosen.length<=i) chosen.push(available[0].b); // fallback
  }
  return chosen;
}

// 初始化
loadData();
</script>
</body>
</html>"""

    # ── 自动核销：根据已抓取的开奖数据，更新待核销购彩记录 ──────
    def _auto_check_results(df) -> int:
        """扫描所有 actual_red==None 的购彩记录，若对应期号已有开奖数据则自动核销。
        返回本次自动核销的记录数。"""
        from itertools import combinations as _comb2
        if df is None or df.empty:
            return 0
        # 建立 issue -> (actual_red, actual_blue) 映射
        red_cols_local = ["r1", "r2", "r3", "r4", "r5", "r6"]
        issue_map = {}
        for _, row in df.iterrows():
            issue_map[str(row["issue"])] = {
                "red": sorted(int(row[c]) for c in red_cols_local),
                "blue": int(row["blue"]),
            }

        purchases = load_purchases()
        matched = 0
        changed = False
        for p in purchases:
            if p.get("actual_red") is not None:
                continue  # 已核销
            issue_key = str(p.get("issue", "")).strip()
            if issue_key not in issue_map:
                continue  # 还未开奖
            draw = issue_map[issue_key]
            actual_red = draw["red"]
            actual_blue = draw["blue"]
            act_set = set(actual_red)
            ticket_type = p.get("type", "单式")

            if ticket_type == "复式":
                blue_list = p["my_blue"] if isinstance(p["my_blue"], list) else [p["my_blue"]]
                total_prize = 0
                winning_count = 0
                best_prize_name = "未中奖"
                best_prize_money = 0
                max_hr, max_hb = 0, False
                for combo in _comb2(p["my_red"], 6):
                    hr = len(set(combo) & act_set)
                    for b in blue_list:
                        hb = (b == actual_blue)
                        pname, pmoney = prize_level(hr, hb)
                        total_prize += pmoney
                        if pmoney > 0:
                            winning_count += 1
                        if pmoney > best_prize_money:
                            best_prize_money = pmoney
                            best_prize_name = pname
                        if hr > max_hr or (hr == max_hr and hb and not max_hb):
                            max_hr, max_hb = hr, hb
                p["actual_red"] = actual_red
                p["actual_blue"] = actual_blue
                p["hit_red"] = max_hr
                p["hit_blue"] = max_hb
                p["prize"] = best_prize_name
                p["winning_count"] = winning_count
                p["prize_money"] = total_prize * p.get("tickets", 1)
            else:
                my_blue = p["my_blue"]
                if isinstance(my_blue, list):
                    my_blue = my_blue[0]
                hr = len(set(p["my_red"]) & act_set)
                hb = (my_blue == actual_blue)
                pname, pmoney = prize_level(hr, hb)
                p["actual_red"] = actual_red
                p["actual_blue"] = actual_blue
                p["hit_red"] = hr
                p["hit_blue"] = hb
                p["prize"] = pname
                p["winning_count"] = 1 if pmoney > 0 else 0
                p["prize_money"] = pmoney * p.get("tickets", 1)

            matched += 1
            changed = True

        if changed:
            save_purchases(purchases)
            # 后台优化权重
            threading.Thread(
                target=optimize_weights_from_feedback,
                args=(purchases,),
                daemon=True,
            ).start()
        return matched

    # ── 冷热分析数据 API ──────────────────────────────────
    def get_hot_cold_data(n: int = 0, start_issue: str = "", end_issue: str = "") -> dict:
        _mgr = DataManager()
        full_df = _mgr.load_local()
        if full_df is None or full_df.empty:
            return {"error": "no data"}
        full_df = full_df.sort_values("issue").reset_index(drop=True)
        if start_issue and end_issue:
            df = full_df[(full_df["issue"] >= start_issue) & (full_df["issue"] <= end_issue)]
        elif n > 0:
            df = full_df.tail(n)
        else:
            df = full_df
        total = len(df)
        red_cols = [f"r{i}" for i in range(1, 7)]
        red_counts = {}
        for b in range(1, 34):
            red_counts[b] = int(df[red_cols].apply(lambda row: b in row.values, axis=1).sum())
        blue_counts = {}
        for b in range(1, 17):
            blue_counts[b] = int((df["blue"] == b).sum())
        return {
            "total": total,
            "min_issue": str(df["issue"].min()),
            "max_issue": str(df["issue"].max()),
            "red": {str(b): {"count": red_counts[b],
                              "pct": round(red_counts[b] / (total * 6) * 100, 2) if total else 0}
                   for b in range(1, 34)},
            "blue": {str(b): {"count": blue_counts[b],
                               "pct": round(blue_counts[b] / total * 100, 2) if total else 0}
                    for b in range(1, 17)},
        }

    # ── 冷热分析页面 ──────────────────────────────────────
    def render_hot_cold_html() -> str:
        return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>双色球冷热分析</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f5f5;font-size:12px;color:#333}
.top-bar{background:#fff;border-bottom:2px solid #e0e0e0;padding:8px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;position:sticky;top:0;z-index:100}
.top-bar h2{font-size:15px;font-weight:bold;color:#333;margin-right:auto}
.period-btn{padding:4px 12px;border:1px solid #ccc;border-radius:3px;cursor:pointer;background:#fff;font-size:12px;color:#2980b9;text-decoration:none}
.period-btn.active,.period-btn:hover{background:#2980b9;color:#fff;border-color:#2980b9}
.period-input{border:1px solid #ccc;border-radius:3px;padding:3px 6px;width:72px;font-size:12px;text-align:center}
.query-btn{padding:4px 12px;background:#e74c3c;color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:12px}
.home-link{padding:4px 12px;background:#27ae60;color:#fff;border-radius:3px;text-decoration:none;font-size:12px}
.trend-link{padding:4px 12px;background:#8e44ad;color:#fff;border-radius:3px;text-decoration:none;font-size:12px}
.section{background:#fff;margin:12px;border:1px solid #ddd;border-radius:4px;overflow:hidden}
.section-title{background:#f7f7f7;padding:6px 14px;font-weight:bold;font-size:13px;border-bottom:1px solid #eee;color:#555}
.section-title.blue{background:#eef3fb}
.tbl-wrap{overflow-x:auto}
table{border-collapse:collapse;min-width:100%}
td,th{border:1px solid #e8e8e8;text-align:center;padding:0;min-width:36px;height:26px}
tr.label-row td{background:#fafafa;color:#666;font-size:11px;height:22px;font-weight:normal;padding:2px 0}
tr.count-row td{font-size:12px;color:#333;height:26px}
tr.pct-row td{font-size:11px;color:#888;height:22px}
tr.chart-row td{height:80px;vertical-align:bottom;padding:0 3px 0}
tr.ball-row td{height:32px;background:#f9f9f9}
td.row-head{background:#f5f5f5;font-weight:bold;color:#555;padding:0 8px;min-width:50px;white-space:nowrap;position:sticky;left:0;z-index:1;font-size:11px}
.bar{background:#4a90d9;border-radius:2px 2px 0 0;width:18px;margin:0 auto;display:block;min-height:2px}
.bar.hot{background:#e74c3c}
.bar.cold{background:#95a5a6}
.rball{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;background:#c0392b;color:#fff;font-size:11px;font-weight:bold}
.bball{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;background:#2980b9;color:#fff;font-size:11px;font-weight:bold}
.heat-legend{display:flex;gap:12px;padding:6px 14px;font-size:11px;align-items:center;border-top:1px solid #eee;background:#fafafa}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:3px}
</style>
</head>
<body>
<div class="top-bar">
  <h2>双色球冷热分析</h2>
  <a href="/" class="home-link">← 主页</a>
  <a href="/trend" class="trend-link">走势图</a>
  <a href="#" class="period-btn active" id="btn-all" onclick="setPeriod(0,this)">全部</a>
  <a href="#" class="period-btn" id="btn30" onclick="setPeriod(30,this)">最近30期</a>
  <a href="#" class="period-btn" id="btn50" onclick="setPeriod(50,this)">最近50期</a>
  <a href="#" class="period-btn" id="btn100" onclick="setPeriod(100,this)">最近100期</a>
  <input class="period-input" id="inp-start" placeholder="起始期号">
  <span>期 至</span>
  <input class="period-input" id="inp-end" placeholder="结束期号">
  <span>期</span>
  <button class="query-btn" onclick="queryRange()">查 看</button>
</div>

<div id="info-bar" style="padding:6px 14px;font-size:11px;color:#888;background:#fff;border-bottom:1px solid #eee"></div>

<div class="section" id="sec-red">
  <div class="section-title">红球：</div>
  <div class="tbl-wrap" id="red-table">加载中…</div>
  <div class="heat-legend">
    <span><span class="dot" style="background:#e74c3c"></span>热号（出现频率前1/3）</span>
    <span><span class="dot" style="background:#4a90d9"></span>温号</span>
    <span><span class="dot" style="background:#95a5a6"></span>冷号（出现频率后1/3）</span>
  </div>
</div>

<div class="section" id="sec-blue">
  <div class="section-title blue">蓝球：</div>
  <div class="tbl-wrap" id="blue-table">加载中…</div>
  <div class="heat-legend">
    <span><span class="dot" style="background:#e74c3c"></span>热号</span>
    <span><span class="dot" style="background:#4a90d9"></span>温号</span>
    <span><span class="dot" style="background:#95a5a6"></span>冷号</span>
  </div>
</div>

<script>
let _period = 0;

function setPeriod(n, el) {
  _period = n;
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('inp-start').value='';
  document.getElementById('inp-end').value='';
  loadData('/api/hot-cold-data?n='+n);
}

function queryRange() {
  const s = document.getElementById('inp-start').value.trim();
  const e = document.getElementById('inp-end').value.trim();
  if(!s||!e){alert('请填写起止期号');return;}
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  loadData('/api/hot-cold-data?start='+s+'&end='+e);
}

function loadData(url) {
  fetch(url).then(r=>r.json()).then(render);
}

function render(d) {
  if(d.error){document.getElementById('red-table').textContent='暂无数据';return;}
  document.getElementById('info-bar').textContent =
    `共 ${d.total} 期数据，期号 ${d.min_issue} — ${d.max_issue}`;
  renderBalls('red-table', d.red, 33, 'r');
  renderBalls('blue-table', d.blue, 16, 'b');
}

function renderBalls(containerId, data, maxN, type) {
  const counts = [];
  for(let b=1;b<=maxN;b++) counts.push(data[b]?data[b].count:0);
  const sorted = [...counts].sort((a,b)=>a-b);
  const lo = sorted[Math.floor(maxN/3)];
  const hi = sorted[Math.floor(maxN*2/3)];
  const maxCount = Math.max(...counts,1);

  const BAR_MAX = 70; // px

  let html = '<table><tbody>';

  // 次数行
  html += '<tr class="count-row"><td class="row-head">次数</td>';
  for(let b=1;b<=maxN;b++){
    html += `<td>${counts[b-1]}</td>`;
  }
  html += '</tr>';

  // 比例行
  html += '<tr class="pct-row"><td class="row-head">比例</td>';
  for(let b=1;b<=maxN;b++){
    const pct = data[b]?data[b].pct:0;
    html += `<td>(${pct}%)</td>`;
  }
  html += '</tr>';

  // 柱状图行
  html += '<tr class="chart-row"><td class="row-head"></td>';
  for(let b=1;b<=maxN;b++){
    const c = counts[b-1];
    const h = Math.max(2, Math.round(c/maxCount*BAR_MAX));
    const cls = c>=hi?'hot':c<=lo?'cold':'';
    html += `<td style="vertical-align:bottom;padding:0 3px 2px"><div class="bar ${cls}" style="height:${h}px"></div></td>`;
  }
  html += '</tr>';

  // 号码行
  html += '<tr class="ball-row"><td class="row-head">号码</td>';
  const ballCls = type==='r'?'rball':'bball';
  for(let b=1;b<=maxN;b++){
    html += `<td><span class="${ballCls}">${String(b).padStart(2,'0')}</span></td>`;
  }
  html += '</tr>';

  html += '</tbody></table>';
  document.getElementById(containerId).innerHTML = html;
}

loadData('/api/hot-cold-data?n=0');
</script>
</body>
</html>"""

    # ── 智能推荐数据 ──────────────────────────────────────
    def get_recommend_data(n: int = 100) -> dict:
        import math, statistics
        _mgr = DataManager()
        full_df = _mgr.load_local()
        if full_df is None or full_df.empty:
            return {"error": "no data"}
        full_df = full_df.sort_values("issue").reset_index(drop=True)
        red_cols = [f"r{i}" for i in range(1, 7)]
        total = len(full_df)
        recent = full_df.tail(n)

        # ── 红球评分 ─────────────────────────────────────
        scores = {}
        alerts = []
        for b in range(1, 34):
            # 1. 历史频率得分（Dirichlet-Multinomial后验）
            hist_count = int(full_df[red_cols].apply(lambda r: b in r.values, axis=1).sum())
            freq_score = (hist_count + 1) / (total * 6 + 33)

            # 2. 遗漏得分（当前遗漏 / 历史最大遗漏）
            miss = 0
            max_miss = 0
            cur_miss = 0
            for _, row in full_df.iterrows():
                if b in [row[c] for c in red_cols]:
                    max_miss = max(max_miss, cur_miss)
                    cur_miss = 0
                else:
                    cur_miss += 1
            miss = cur_miss
            max_miss = max(max_miss, cur_miss)
            miss_score = min(miss / max(max_miss, 1), 1.0)

            # 均值遗漏
            miss_intervals = []
            cur_gap = 0
            for _, row in full_df.iterrows():
                if b in [row[c] for c in red_cols]:
                    miss_intervals.append(cur_gap)
                    cur_gap = 0
                else:
                    cur_gap += 1
            avg_miss = statistics.mean(miss_intervals) if miss_intervals else 0
            std_miss = statistics.stdev(miss_intervals) if len(miss_intervals) > 1 else 0
            if miss > avg_miss + 2 * std_miss and std_miss > 0:
                alerts.append({"ball": b, "type": "red", "miss": miss,
                               "avg": round(avg_miss, 1), "threshold": round(avg_miss + 2*std_miss, 1)})

            # 3. 近期动量得分（最近n期频率 vs 历史频率）
            recent_count = int(recent[red_cols].apply(lambda r: b in r.values, axis=1).sum())
            recent_freq = recent_count / (len(recent) * 6)
            momentum_score = recent_freq / max(freq_score, 0.001)

            # 综合得分（权重：历史35% + 遗漏35% + 近期动量20% + 随机探索10%）
            import random as _rand
            composite = 0.35 * freq_score * 33 + 0.35 * miss_score + 0.20 * min(momentum_score / 3, 1.0)
            scores[b] = {
                "ball": b,
                "hist_count": hist_count,
                "hist_pct": round(hist_count / (total * 6) * 100, 2),
                "recent_count": recent_count,
                "miss": miss,
                "avg_miss": round(avg_miss, 1),
                "max_miss": max_miss,
                "score": round(composite, 4),
                "freq_score": round(freq_score * 33, 3),
                "miss_score": round(miss_score, 3),
                "momentum": round(min(momentum_score, 3.0), 3),
            }

        # ── 蓝球 Markov 转移概率 ──────────────────────────
        blue_seq = list(full_df["blue"].astype(int))
        trans = {b: {nb: 0 for nb in range(1, 17)} for b in range(1, 17)}
        for i in range(len(blue_seq) - 1):
            trans[blue_seq[i]][blue_seq[i+1]] += 1
        last_blue = blue_seq[-1] if blue_seq else 1
        raw_next = trans[last_blue]
        total_trans = sum(raw_next.values())
        # 加平滑
        blue_probs = {b: round((raw_next[b] + 1) / (total_trans + 16) * 100, 2) for b in range(1, 17)}
        # 二阶（如果数据足够）
        if len(blue_seq) >= 2:
            prev2 = (blue_seq[-2], blue_seq[-1])
            trans2 = {}
            for i in range(len(blue_seq) - 2):
                k = (blue_seq[i], blue_seq[i+1])
                nxt = blue_seq[i+2]
                trans2.setdefault(k, {b: 0 for b in range(1, 17)})
                trans2[k][nxt] += 1
            if prev2 in trans2:
                raw2 = trans2[prev2]
                t2 = sum(raw2.values())
                blue_probs2 = {b: round((raw2[b] + 1) / (t2 + 16) * 100, 2) for b in range(1, 17)}
                # 混合一阶+二阶
                blue_probs = {b: round(0.4 * blue_probs[b] + 0.6 * blue_probs2[b], 2) for b in range(1, 17)}

        # ── 生成推荐号码组合 ──────────────────────────────
        import random as _rand
        def weighted_pick(score_dict, k, exclude=set()):
            pool = [(b, v["score"]) for b, v in score_dict.items() if b not in exclude]
            total_w = sum(w for _, w in pool)
            chosen = []
            used = set(exclude)
            for _ in range(k):
                r = _rand.random() * sum(w for b, w in pool if b not in used)
                acc = 0
                for b, w in pool:
                    if b in used: continue
                    acc += w
                    if acc >= r:
                        chosen.append(b)
                        used.add(b)
                        break
                else:
                    remaining = [b for b, _ in pool if b not in used]
                    if remaining:
                        chosen.append(remaining[0])
                        used.add(remaining[0])
            return sorted(chosen)

        def pick_blue(probs):
            pool = list(probs.items())
            total_w = sum(w for _, w in pool)
            r = _rand.random() * total_w
            acc = 0
            for b, w in pool:
                acc += w
                if acc >= r: return b
            return pool[-1][0]

        def ac_value(balls):
            diffs = set(abs(balls[i]-balls[j]) for i in range(len(balls)) for j in range(i+1,len(balls)))
            return len(diffs) - (len(balls) - 1)

        alert_ball_set = {a["ball"] for a in alerts}
        max_score = max(v["score"] for v in scores.values())

        _rand.seed()
        combos = []
        for _ in range(8):
            red = weighted_pick(scores, 6)
            blue = pick_blue(blue_probs)
            z1 = sum(1 for b in red if 1 <= b <= 11)
            z2 = sum(1 for b in red if 12 <= b <= 22)
            z3 = sum(1 for b in red if 23 <= b <= 33)
            balance = "均衡" if all(z > 0 for z in [z1, z2, z3]) else "偏区"
            odd = sum(1 for b in red if b % 2 == 1)
            s = sum(red)
            span = max(red) - min(red)
            ac = ac_value(red)
            combo_score = sum(scores[b]["score"] for b in red) / 6
            conf_pct = round(combo_score / max_score * 100 * 0.06, 1)  # 归一化到合理显示区间
            # 生成选号理由
            reasons = []
            high_conf = [b for b in red if scores[b]["score"] >= max_score * 0.65]
            if high_conf: reasons.append(f"高置信度红球 {high_conf}")
            miss_balls = [b for b in red if b in alert_ball_set]
            if miss_balls: reasons.append(f"含遗漏回补号 {miss_balls}")
            hot = [b for b in red if scores[b]["momentum"] > 1.4]
            if hot: reasons.append(f"近期热号 {hot}")
            if blue_probs[blue] >= max(blue_probs.values()) * 0.75:
                reasons.append(f"蓝球 {blue} 处于Markov高概率区间")
            combos.append({
                "red": red, "blue": blue, "sum": s,
                "odd_even": f"{odd}奇{6-odd}偶",
                "balance": balance,
                "z1": z1, "z2": z2, "z3": z3,
                "span": span, "ac": ac,
                "conf": conf_pct,
                "reasons": reasons,
                # 每颗红球的评分档位
                "ball_tiers": {str(b): ("hot" if scores[b]["score"]>=max_score*0.65 else "warm" if scores[b]["score"]>=max_score*0.35 else "cool") for b in red},
            })

        return {
            "scores": {str(b): scores[b] for b in range(1, 34)},
            "blue_probs": {str(b): blue_probs[b] for b in range(1, 17)},
            "combos": combos,
            "alerts": alerts,
            "last_blue": last_blue,
            "total": total,
            "n": n,
        }

    # ── 统计分析数据 ──────────────────────────────────────
    def get_stats_data(n: int = 0) -> dict:
        _mgr = DataManager()
        full_df = _mgr.load_local()
        if full_df is None or full_df.empty:
            return {"error": "no data"}
        full_df = full_df.sort_values("issue").reset_index(drop=True)
        df = full_df.tail(n) if n > 0 else full_df
        red_cols = [f"r{i}" for i in range(1, 7)]
        sums, spans, odds, highs = [], [], [], []
        for _, row in df.iterrows():
            reds = sorted(int(row[c]) for c in red_cols)
            sums.append(sum(reds))
            spans.append(reds[-1] - reds[0])
            odds.append(sum(1 for b in reds if b % 2 == 1))
            highs.append(sum(1 for b in reds if b > 16))

        def hist(vals, lo, hi, step=1):
            buckets = {}
            for v in vals:
                k = round(v // step * step)
                buckets[k] = buckets.get(k, 0) + 1
            return [{"x": x, "y": buckets.get(x, 0)} for x in range(lo, hi + 1, step)]

        import statistics as _st
        return {
            "total": len(df),
            "sum": {
                "data": hist(sums, 21, 183, 5),
                "mean": round(_st.mean(sums), 1),
                "std": round(_st.stdev(sums) if len(sums) > 1 else 0, 1),
                "mode_range": f"{round(_st.mean(sums)-_st.stdev(sums) if len(sums)>1 else 0)}-{round(_st.mean(sums)+_st.stdev(sums) if len(sums)>1 else 0)}",
            },
            "span": {
                "data": hist(spans, 0, 32, 1),
                "mean": round(_st.mean(spans), 1),
            },
            "odd": {
                "data": [{"x": i, "y": odds.count(i)} for i in range(7)],
                "mean": round(_st.mean(odds), 2),
            },
            "high": {
                "data": [{"x": i, "y": highs.count(i)} for i in range(7)],
                "mean": round(_st.mean(highs), 2),
            },
            "blue_dist": {
                str(b): int((df["blue"].astype(int) == b).sum()) for b in range(1, 17)
            },
        }

    # ── 共现热图数据 ──────────────────────────────────────
    def get_cooccur_data(n: int = 0) -> dict:
        _mgr = DataManager()
        full_df = _mgr.load_local()
        if full_df is None or full_df.empty:
            return {"error": "no data"}
        full_df = full_df.sort_values("issue").reset_index(drop=True)
        df = full_df.tail(n) if n > 0 else full_df
        red_cols = [f"r{i}" for i in range(1, 7)]
        matrix = [[0] * 33 for _ in range(33)]
        for _, row in df.iterrows():
            reds = [int(row[c]) - 1 for c in red_cols]
            for i in range(len(reds)):
                for j in range(len(reds)):
                    if i != j:
                        matrix[reds[i]][reds[j]] += 1
        # 对角线填频次
        for i in range(33):
            matrix[i][i] = int(df[red_cols].apply(lambda r: (i+1) in r.values, axis=1).sum())
        max_val = max(max(r) for r in matrix)
        return {"matrix": matrix, "max_val": max_val, "total": len(df)}

    # ── 智能推荐页面（Liquid Glass AI科技风） ───────────────
    def render_recommend_html() -> str:
        return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>AI智能推荐 - 双色球</title>
<style>
:root{
  --red:#ff375f;--red2:#ff6b85;--rs:rgba(255,55,95,.55);
  --blue:#0a84ff;--blue2:#5ac8fa;--bs:rgba(10,132,255,.5);
  --cyan:#32d74b;--cs:rgba(50,215,75,.4);
  --gold:#ffd60a;--gs:rgba(255,214,10,.45);
  --warn:#ff9f0a;--ws:rgba(255,159,10,.4);
  --purple:#bf5af2;--ps:rgba(191,90,242,.4);
  --glass:rgba(255,255,255,0.07);
  --glass-border:rgba(255,255,255,0.14);
  --glass-hover:rgba(255,255,255,0.12);
  --text:#f5f5f7;--dim:rgba(255,255,255,0.45);--dimmer:rgba(255,255,255,0.22);
}
*{box-sizing:border-box;margin:0;padding:0}
html{height:100%}
body{min-height:100%;font-family:-apple-system,'SF Pro Display','PingFang SC','Helvetica Neue',sans-serif;
  background:#000;color:var(--text);font-size:13px;overflow-x:hidden}

/* ═══ Aurora BG ═══ */
.aurora{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.aurora-blob{position:absolute;border-radius:50%;filter:blur(80px);animation:drift ease-in-out infinite alternate}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(var(--dx),var(--dy)) scale(var(--ds))}}
.a1{width:700px;height:700px;top:-200px;left:-100px;background:radial-gradient(circle,rgba(10,132,255,.35),transparent 70%);--dx:80px;--dy:60px;--ds:1.15;animation-duration:18s}
.a2{width:600px;height:600px;top:-100px;right:-80px;background:radial-gradient(circle,rgba(191,90,242,.28),transparent 70%);--dx:-60px;--dy:80px;--ds:1.1;animation-duration:22s}
.a3{width:500px;height:500px;bottom:100px;left:20%;background:radial-gradient(circle,rgba(255,55,95,.22),transparent 70%);--dx:40px;--dy:-50px;--ds:1.2;animation-duration:16s}
.a4{width:400px;height:400px;bottom:0;right:10%;background:radial-gradient(circle,rgba(50,215,75,.18),transparent 70%);--dx:-30px;--dy:-40px;--ds:1.08;animation-duration:20s}

/* ═══ Nav ═══ */
nav{position:sticky;top:0;z-index:200;display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:10px 20px;
  background:rgba(0,0,0,0.55);
  backdrop-filter:blur(28px) saturate(180%);
  -webkit-backdrop-filter:blur(28px) saturate(180%);
  border-bottom:1px solid var(--glass-border)}
.logo{font-size:15px;font-weight:700;letter-spacing:.5px;margin-right:auto;
  background:linear-gradient(135deg,var(--blue2),var(--blue),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-a{padding:4px 11px;border:1px solid var(--glass-border);color:var(--dim);border-radius:20px;
  text-decoration:none;font-size:11px;transition:all .2s;backdrop-filter:blur(8px)}
.nav-a:hover{border-color:var(--blue2);color:var(--blue2);background:rgba(90,200,250,.08)}
.prd-btn{padding:4px 12px;border:1px solid var(--glass-border);color:var(--dimmer);
  border-radius:20px;cursor:pointer;background:transparent;font-size:11px;
  transition:all .2s;font-family:inherit}
.prd-btn.active{border-color:var(--blue);color:var(--blue);background:rgba(10,132,255,.12);
  box-shadow:0 0 12px rgba(10,132,255,.25)}
.prd-btn:hover{border-color:var(--blue2);color:var(--blue2)}
.gen-btn{padding:5px 16px;border:none;border-radius:20px;cursor:pointer;font-family:inherit;
  font-size:12px;font-weight:600;color:#fff;letter-spacing:.3px;
  background:linear-gradient(135deg,var(--red),#c0003e);
  box-shadow:0 0 18px var(--rs),inset 0 1px 0 rgba(255,255,255,.2);
  transition:all .25s}
.gen-btn:hover{box-shadow:0 0 28px var(--rs);transform:translateY(-1px)}

/* ═══ Layout ═══ */
main{position:relative;z-index:1;max-width:1300px;margin:0 auto;padding:20px 16px;
  display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:860px){main{grid-template-columns:1fr}}

/* ═══ Glass Card ═══ */
.gc{position:relative;border-radius:20px;padding:20px;overflow:hidden;
  background:var(--glass);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(24px) saturate(160%);
  -webkit-backdrop-filter:blur(24px) saturate(160%);
  box-shadow:0 8px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.1);
  transition:border-color .3s,box-shadow .3s}
.gc:hover{border-color:rgba(255,255,255,.22);box-shadow:0 12px 40px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.15)}
.gc.full{grid-column:1/-1}
/* 顶部高光条 */
.gc::after{content:'';position:absolute;top:0;left:15%;right:15%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent)}
/* 色调角标 */
.gc.red-tint{box-shadow:0 8px 32px rgba(0,0,0,.4),0 0 0 1px rgba(255,55,95,.2),inset 0 1px 0 rgba(255,255,255,.1)}
.gc.blue-tint{box-shadow:0 8px 32px rgba(0,0,0,.4),0 0 0 1px rgba(10,132,255,.2),inset 0 1px 0 rgba(255,255,255,.1)}
.gc.gold-tint{box-shadow:0 8px 32px rgba(0,0,0,.4),0 0 0 1px rgba(255,214,10,.18),inset 0 1px 0 rgba(255,255,255,.1)}

/* ═══ Section Header ═══ */
.sh{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.sh-icon{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.sh-icon.red{background:linear-gradient(135deg,rgba(255,55,95,.3),rgba(255,55,95,.1));border:1px solid rgba(255,55,95,.3)}
.sh-icon.blue{background:linear-gradient(135deg,rgba(10,132,255,.3),rgba(10,132,255,.1));border:1px solid rgba(10,132,255,.3)}
.sh-icon.gold{background:linear-gradient(135deg,rgba(255,214,10,.3),rgba(255,214,10,.1));border:1px solid rgba(255,214,10,.3)}
.sh-icon.cyan{background:linear-gradient(135deg,rgba(50,215,75,.3),rgba(50,215,75,.1));border:1px solid rgba(50,215,75,.3)}
.sh-title{font-size:14px;font-weight:700;letter-spacing:.2px}
.sh-title.red{color:var(--red2);text-shadow:0 0 20px var(--rs)}
.sh-title.blue{color:var(--blue2);text-shadow:0 0 20px var(--bs)}
.sh-title.gold{color:var(--gold);text-shadow:0 0 20px var(--gs)}
.sh-title.cyan{color:var(--cyan);text-shadow:0 0 20px var(--cs)}
.sh-sub{font-size:10px;color:var(--dimmer);margin-left:auto;text-align:right;line-height:1.5}

/* ═══ Score Grid ═══ */
.sg{display:grid;grid-template-columns:repeat(11,1fr);gap:4px}
@media(max-width:700px){.sg{grid-template-columns:repeat(6,1fr)}}
.si{text-align:center;cursor:pointer;transition:transform .2s}
.si:hover{transform:translateY(-4px)}
.sbar-bg{height:68px;display:flex;align-items:flex-end;justify-content:center;
  background:rgba(255,255,255,0.04);border-radius:6px;margin-bottom:5px;position:relative;overflow:hidden}
/* scanline shimmer */
.sbar-bg::before{content:'';position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(255,255,255,.04) 0%,transparent 100%)}
.sbar{width:18px;border-radius:4px 4px 0 0;min-height:4px;position:relative;
  transition:height .7s cubic-bezier(.34,1.56,.64,1)}
.sbar::after{content:'';position:absolute;top:2px;left:2px;right:2px;height:40%;
  background:rgba(255,255,255,0.3);border-radius:3px 3px 0 0}
.sbar.t1{background:linear-gradient(0deg,var(--red),#ff8fa3);box-shadow:0 0 10px var(--rs)}
.sbar.t2{background:linear-gradient(0deg,var(--warn),#ffc947);box-shadow:0 0 8px var(--ws)}
.sbar.t3{background:linear-gradient(0deg,#1d4ed8,var(--blue2));box-shadow:0 0 5px var(--bs)}
.snum{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
  border-radius:50%;font-size:10px;font-weight:700;color:#fff;margin-bottom:2px}
.snum.r{background:radial-gradient(circle at 33% 33%,#ff8fa3,var(--red) 60%,#8b0020);
  box-shadow:0 2px 6px rgba(0,0,0,.4),0 0 8px var(--rs)}
.snum.alert-ball{background:radial-gradient(circle at 33% 33%,#ffd60a,var(--warn) 60%,#7d4500);
  box-shadow:0 2px 6px rgba(0,0,0,.4);animation:glow-warn 1.5s infinite}
@keyframes glow-warn{0%,100%{box-shadow:0 2px 6px rgba(0,0,0,.4),0 0 6px var(--ws)}
  50%{box-shadow:0 2px 6px rgba(0,0,0,.4),0 0 18px var(--ws),0 0 32px rgba(255,159,10,.3)}}
.sval{font-size:9px;color:var(--dimmer);font-variant-numeric:tabular-nums}

/* ═══ Alert Chips ═══ */
.alert-wrap{display:flex;flex-wrap:wrap;gap:8px}
.ach{display:flex;align-items:center;gap:8px;
  background:rgba(255,159,10,0.08);
  border:1px solid rgba(255,159,10,0.28);
  border-radius:14px;padding:8px 12px;
  animation:ach-pulse 2s ease-in-out infinite}
@keyframes ach-pulse{0%,100%{border-color:rgba(255,159,10,.28)}50%{border-color:rgba(255,159,10,.7);box-shadow:0 0 12px rgba(255,159,10,.2)}}
.ach-info{font-size:10px;color:var(--warn)}
.ach-info b{color:#fff;font-size:12px}
.ach-bar-track{width:90px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;margin-top:4px;overflow:hidden}
.ach-bar-fill{height:100%;background:linear-gradient(90deg,var(--warn),var(--red));border-radius:2px}
.no-alert{color:var(--cyan);font-size:12px;display:flex;align-items:center;gap:6px}

/* ═══ Blue Probs ═══ */
.bg16{display:grid;grid-template-columns:repeat(8,1fr);gap:6px}
@media(max-width:600px){.bg16{grid-template-columns:repeat(4,1fr)}}
.bi{text-align:center}
.bbar-bg{height:60px;display:flex;align-items:flex-end;justify-content:center;
  background:rgba(255,255,255,.04);border-radius:6px;margin-bottom:4px}
.bbar{width:20px;border-radius:4px 4px 0 0;min-height:3px;transition:height .5s}
.bbar.btop{background:linear-gradient(0deg,#004ac2,var(--blue2));box-shadow:0 0 12px var(--bs)}
.bbar.bmid{background:linear-gradient(0deg,#001f6e,#0a84ff);box-shadow:0 0 5px rgba(10,132,255,.3)}
.bbar.blow{background:rgba(255,255,255,.08)}
.bnum{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
  border-radius:50%;font-size:10px;font-weight:700;color:#fff;margin-bottom:2px;
  background:radial-gradient(circle at 33% 33%,var(--blue2),var(--blue) 60%,#001f6e);
  box-shadow:0 2px 5px rgba(0,0,0,.5),0 0 8px var(--bs)}
.bpct{font-size:9px;color:var(--dimmer);font-variant-numeric:tabular-nums}

/* ═══ Combo Cards ═══ */
.cl{display:flex;flex-direction:column;gap:12px}
.cc{position:relative;border-radius:16px;overflow:hidden;
  background:rgba(255,255,255,0.05);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(12px);
  transition:transform .25s,box-shadow .25s,border-color .25s;
  animation:card-in .4s ease both}
@keyframes card-in{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.cc:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.25);
  box-shadow:0 16px 40px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.15)}
/* 渐变侧条 */
.cc-side{position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:3px 0 0 3px}
/* 顶部扫光 */
.cc::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);opacity:.7}

.cc-header{display:flex;align-items:center;gap:10px;padding:14px 16px 10px 20px}
.cc-rank{font-size:11px;font-weight:700;color:var(--dimmer);width:28px;flex-shrink:0;
  font-variant-numeric:tabular-nums}
.balls-row{display:flex;align-items:center;gap:5px;flex:1;flex-wrap:wrap}
.rb{display:inline-flex;align-items:center;justify-content:center;
  width:34px;height:34px;border-radius:50%;
  font-size:12px;font-weight:700;color:#fff;
  transition:transform .18s}
.rb:hover{transform:scale(1.15)}
.rb.hot{background:radial-gradient(circle at 33% 28%,#ff8fa3,var(--red) 55%,#6e001a);
  box-shadow:0 3px 8px rgba(0,0,0,.5),0 0 10px var(--rs)}
.rb.warm{background:radial-gradient(circle at 33% 28%,#ffd347,var(--warn) 55%,#6e3600);
  box-shadow:0 3px 8px rgba(0,0,0,.5),0 0 8px var(--ws)}
.rb.cool{background:radial-gradient(circle at 33% 28%,#82b4ff,var(--blue) 55%,#001d6e);
  box-shadow:0 3px 8px rgba(0,0,0,.5),0 0 6px var(--bs)}
.sep{color:rgba(255,255,255,.2);font-size:18px;margin:0 2px}
.bb{display:inline-flex;align-items:center;justify-content:center;
  width:34px;height:34px;border-radius:50%;
  font-size:12px;font-weight:700;color:#fff;
  background:radial-gradient(circle at 33% 28%,#82cfff,var(--blue) 55%,#001d6e);
  box-shadow:0 3px 8px rgba(0,0,0,.5),0 0 12px var(--bs)}

/* 置信度条 */
.conf-bar{margin:0 16px 0 20px;height:3px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden}
.conf-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--blue),var(--purple),var(--red));
  box-shadow:0 0 8px rgba(191,90,242,.6)}

/* 底部标签 */
.cc-footer{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:8px 16px 12px 20px}
.tag{display:inline-flex;align-items:center;gap:3px;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:500}
.tag.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);color:var(--dim)}
.tag.ok{background:rgba(50,215,75,.1);border:1px solid rgba(50,215,75,.3);color:var(--cyan)}
.tag.warn{background:rgba(255,159,10,.1);border:1px solid rgba(255,159,10,.3);color:var(--warn)}
.tag.reason{background:rgba(10,132,255,.1);border:1px solid rgba(10,132,255,.25);color:var(--blue2)}
.tag.hot-tag{background:rgba(255,55,95,.1);border:1px solid rgba(255,55,95,.3);color:var(--red2)}
.conf-label{margin-left:auto;font-size:10px;color:var(--dimmer)}

/* ═══ Loading ═══ */
.ld{grid-column:1/-1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:80px 20px;gap:20px}
.ld-ring{width:56px;height:56px;border-radius:50%;
  border:2px solid rgba(255,255,255,.1);
  border-top-color:var(--blue2);
  animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.ld-dots{display:flex;gap:6px}
.ld-dot{width:6px;height:6px;border-radius:50%;background:var(--blue2);opacity:.3;
  animation:dotpulse 1.2s ease-in-out infinite}
.ld-dot:nth-child(2){animation-delay:.2s}.ld-dot:nth-child(3){animation-delay:.4s}
@keyframes dotpulse{0%,100%{opacity:.3;transform:scale(1)}50%{opacity:1;transform:scale(1.4)}}
.ld-txt{font-size:13px;color:var(--dim);letter-spacing:2px}

footer{position:relative;z-index:1;text-align:center;padding:20px;
  color:var(--dimmer);font-size:10px;letter-spacing:.5px}
</style>
</head>
<body>

<div class="aurora">
  <div class="aurora-blob a1"></div>
  <div class="aurora-blob a2"></div>
  <div class="aurora-blob a3"></div>
  <div class="aurora-blob a4"></div>
</div>

<nav>
  <span class="logo">◈ AI PREDICT · 双色球智能分析</span>
  <a href="/" class="nav-a">← 主页</a>
  <a href="/trend" class="nav-a">走势图</a>
  <a href="/hot-cold" class="nav-a">冷热</a>
  <a href="/stats-analysis" class="nav-a">统计</a>
  <a href="/cooccur" class="nav-a">热图</a>
  <button class="prd-btn active" id="btn100" onclick="setPeriod(100,this)">近100期</button>
  <button class="prd-btn" id="btn200" onclick="setPeriod(200,this)">近200期</button>
  <button class="prd-btn" id="btn500" onclick="setPeriod(500,this)">近500期</button>
  <button class="prd-btn" id="btnall" onclick="setPeriod(0,this)">全历史</button>
  <button class="gen-btn" onclick="reload()">⟳ 重新生成</button>
</nav>

<main id="grid">
  <div class="gc full ld-wrap" style="padding:0">
    <div class="ld">
      <div class="ld-ring"></div>
      <div class="ld-dots"><div class="ld-dot"></div><div class="ld-dot"></div><div class="ld-dot"></div></div>
      <div class="ld-txt">AI 模型推理中</div>
    </div>
  </div>
</main>

<footer>仅供概率统计研究 · 彩票开奖随机 · 理性购彩 · 切勿沉迷</footer>

<script>
const SIDE_COLORS=[
  'linear-gradient(180deg,#ff375f,#0a84ff)',
  'linear-gradient(180deg,#bf5af2,#32d74b)',
  'linear-gradient(180deg,#ff9f0a,#0a84ff)',
  'linear-gradient(180deg,#32d74b,#bf5af2)',
  'linear-gradient(180deg,#0a84ff,#ff375f)',
  'linear-gradient(180deg,#ffd60a,#ff375f)',
  'linear-gradient(180deg,#5ac8fa,#bf5af2)',
  'linear-gradient(180deg,#ff375f,#32d74b)',
];

let _period=100;
function setPeriod(n,el){
  _period=n;
  document.querySelectorAll('.prd-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');load();
}
function reload(){load();}

function load(){
  document.getElementById('grid').innerHTML=`<div class="gc full" style="padding:0"><div class="ld">
    <div class="ld-ring"></div>
    <div class="ld-dots"><div class="ld-dot"></div><div class="ld-dot"></div><div class="ld-dot"></div></div>
    <div class="ld-txt">AI 模型推理中</div></div></div>`;
  fetch('/api/recommend-data?n='+_period).then(r=>r.json()).then(render);
}

function render(d){
  if(d.error){document.getElementById('grid').innerHTML='<div class="gc full" style="text-align:center;padding:60px;color:#ff375f">暂无数据，请先初始化</div>';return;}
  const sc=d.scores,bp=d.blue_probs;
  const vals=Object.values(sc).map(v=>v.score);
  const maxS=Math.max(...vals),minS=Math.min(...vals),rng=Math.max(maxS-minS,.001);
  const maxBP=Math.max(...Object.values(bp));
  const alertSet=new Set(d.alerts.map(a=>a.ball));

  /* ── 红球评分 */
  let rH='';
  for(let b=1;b<=33;b++){
    const s=sc[b];if(!s)continue;
    const n=(s.score-minS)/rng;
    const h=Math.max(4,Math.round(n*62));
    const t=n>=.7?'t1':n>=.38?'t2':'t3';
    const ia=alertSet.has(b);
    rH+=`<div class="si" title="历史${s.hist_count}次·遗漏${s.miss}期·近期${s.recent_count}次·动量${s.momentum}x">
      <div class="sbar-bg"><div class="sbar ${t}" style="height:${h}px"></div></div>
      <div class="snum r${ia?' alert-ball':''}">${String(b).padStart(2,'0')}</div>
      <div class="sval">${(s.score*100).toFixed(1)}</div>
    </div>`;
  }

  /* ── 遗漏预警 */
  let aH=d.alerts.length===0
    ?'<div class="no-alert">✓ 当前无超长遗漏号码</div>'
    :d.alerts.map(a=>{
      const pct=Math.min(100,Math.round(a.miss/a.threshold*100));
      return `<div class="ach">
        <div class="snum r alert-ball">${String(a.ball).padStart(2,'0')}</div>
        <div><div class="ach-info">遗漏 <b>${a.miss}</b> 期 / 均值 ${a.avg} / 阈 ${a.threshold}</div>
          <div class="ach-bar-track"><div class="ach-bar-fill" style="width:${pct}%"></div></div></div>
      </div>`;
    }).join('');

  /* ── 蓝球概率 */
  let bH='';
  for(let b=1;b<=16;b++){
    const p=bp[b]||0;
    const h=Math.max(3,Math.round(p/maxBP*54));
    const t=p>=maxBP*.75?'btop':p>=maxBP*.4?'bmid':'blow';
    bH+=`<div class="bi">
      <div class="bbar-bg"><div class="bbar ${t}" style="height:${h}px"></div></div>
      <div class="bnum">${String(b).padStart(2,'0')}</div>
      <div class="bpct">${p}%</div>
    </div>`;
  }

  /* ── 推荐组合 */
  let cH='';
  d.combos.forEach((c,i)=>{
    const delay=i*55;
    const rballs=c.red.map(b=>{
      const t=(c.ball_tiers&&c.ball_tiers[b])||'cool';
      return `<div class="rb ${t}" title="${t==='hot'?'高置信':t==='warm'?'中置信':'稳健'}">${String(b).padStart(2,'0')}</div>`;
    }).join('');
    const bball=`<div class="bb">${String(c.blue).padStart(2,'0')}</div>`;
    const confW=Math.max(5,Math.min(98,c.conf*14));
    const balTag=c.balance==='均衡'
      ?'<span class="tag ok">✓ 三区均衡</span>'
      :'<span class="tag warn">⚡ 偏区</span>';
    const reasons=(c.reasons||[]).map(r=>`<span class="tag reason">▸ ${r}</span>`).join('');
    cH+=`<div class="cc" style="animation-delay:${delay}ms">
      <div class="cc-side" style="background:${SIDE_COLORS[i%SIDE_COLORS.length]}"></div>
      <div class="cc-header">
        <div class="cc-rank">#${String(i+1).padStart(2,'0')}</div>
        <div class="balls-row">${rballs}<div class="sep">+</div>${bball}</div>
        <div class="conf-label">置信度 ${c.conf}%</div>
      </div>
      <div class="conf-bar"><div class="conf-fill" style="width:${confW}%"></div></div>
      <div class="cc-footer">
        <span class="tag stat">和值 ${c.sum}</span>
        <span class="tag stat">跨度 ${c.span}</span>
        <span class="tag stat">AC ${c.ac}</span>
        <span class="tag stat">${c.odd_even}</span>
        <span class="tag stat">${c.z1}·${c.z2}·${c.z3}</span>
        ${balTag}
        ${reasons}
      </div>
    </div>`;
  });

  document.getElementById('grid').innerHTML=`
    <div class="gc full red-tint">
      <div class="sh">
        <div class="sh-icon red">🎯</div>
        <div><div class="sh-title red">红球综合置信度</div></div>
        <div class="sh-sub">历史频率 35% · 遗漏回归 35% · 近期动量 20%<br>分析期数：${d.n||'全量 '+d.total+' 期'}</div>
      </div>
      <div class="sg">${rH}</div>
    </div>

    <div class="gc gold-tint">
      <div class="sh">
        <div class="sh-icon gold">⚡</div>
        <div><div class="sh-title gold">遗漏预警</div></div>
        <div class="sh-sub">当前遗漏 &gt; 均值 + 2σ<br>均值回归概率偏高</div>
      </div>
      <div class="alert-wrap">${aH}</div>
    </div>

    <div class="gc blue-tint">
      <div class="sh">
        <div class="sh-icon blue">🔵</div>
        <div><div class="sh-title blue">蓝球 · Markov 转移概率</div></div>
        <div class="sh-sub">上期蓝球：${String(d.last_blue).padStart(2,'0')}<br>混合一阶+二阶转移矩阵</div>
      </div>
      <div class="bg16">${bH}</div>
    </div>

    <div class="gc full">
      <div class="sh">
        <div class="sh-icon cyan">🎱</div>
        <div><div class="sh-title cyan">AI 智能推荐 · ${d.combos.length} 注</div></div>
        <div class="sh-sub">综合评分加权抽样 · 每次点击「重新生成」得到新组合<br>
          <span style="color:rgba(255,55,95,.8)">●</span> 高置信&nbsp;
          <span style="color:rgba(255,159,10,.8)">●</span> 中置信&nbsp;
          <span style="color:rgba(10,132,255,.8)">●</span> 稳健号</div>
      </div>
      <div class="cl">${cH}</div>
    </div>`;
}

/* ═══ Paywall Modal ═══ */
.pw-overlay{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;
  background:rgba(0,0,0,0.75);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
.pw-card{position:relative;width:min(480px,92vw);border-radius:28px;overflow:hidden;
  background:rgba(28,28,30,0.92);border:1px solid rgba(255,255,255,.15);
  box-shadow:0 32px 80px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.12);
  padding:36px 32px 28px}
.pw-card::before{content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent)}
.pw-aurora{position:absolute;top:-60px;right:-60px;width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(10,132,255,.35),transparent 70%);filter:blur(40px);pointer-events:none}
.pw-aurora2{position:absolute;bottom:-40px;left:-40px;width:160px;height:160px;border-radius:50%;
  background:radial-gradient(circle,rgba(191,90,242,.3),transparent 70%);filter:blur(40px);pointer-events:none}
.pw-lock{font-size:40px;text-align:center;margin-bottom:8px}
.pw-title{font-size:22px;font-weight:700;text-align:center;
  background:linear-gradient(135deg,#fff,rgba(255,255,255,.7));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px}
.pw-sub{font-size:13px;color:rgba(255,255,255,.45);text-align:center;margin-bottom:24px;line-height:1.6}
.pw-plans{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}
.pw-plan{border-radius:14px;padding:14px 12px;border:1px solid rgba(255,255,255,.1);
  background:rgba(255,255,255,.05);cursor:pointer;transition:all .2s;text-align:center}
.pw-plan:hover,.pw-plan.selected{border-color:var(--blue);background:rgba(10,132,255,.12);
  box-shadow:0 0 16px rgba(10,132,255,.25)}
.pw-plan .plan-name{font-size:11px;color:rgba(255,255,255,.5);margin-bottom:4px}
.pw-plan .plan-price{font-size:20px;font-weight:700;color:#fff}
.pw-plan .plan-desc{font-size:10px;color:rgba(255,255,255,.35);margin-top:2px}
.pw-plan.featured{border-color:var(--purple);background:rgba(191,90,242,.1)}
.pw-plan.featured:hover,.pw-plan.featured.selected{border-color:var(--purple);background:rgba(191,90,242,.18);
  box-shadow:0 0 16px rgba(191,90,242,.3)}
.pw-divider{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.pw-divider::before,.pw-divider::after{content:'';flex:1;height:1px;background:rgba(255,255,255,.1)}
.pw-divider span{font-size:11px;color:rgba(255,255,255,.3)}
.pw-input{width:100%;padding:12px 16px;border-radius:12px;border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);color:#fff;font-size:14px;font-family:monospace;
  letter-spacing:1px;outline:none;transition:border-color .2s}
.pw-input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(10,132,255,.15)}
.pw-input::placeholder{color:rgba(255,255,255,.25);letter-spacing:0;font-family:inherit}
.pw-btn{width:100%;padding:14px;border-radius:14px;border:none;cursor:pointer;
  font-size:15px;font-weight:600;color:#fff;margin-top:12px;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  box-shadow:0 0 20px rgba(10,132,255,.3);transition:all .25s;font-family:inherit}
.pw-btn:hover{box-shadow:0 0 30px rgba(10,132,255,.5);transform:translateY(-1px)}
.pw-err{text-align:center;font-size:12px;color:var(--red);margin-top:8px;min-height:18px}
.pw-note{text-align:center;font-size:10px;color:rgba(255,255,255,.2);margin-top:12px;line-height:1.7}
.pw-note a{color:rgba(10,132,255,.7);text-decoration:none}
</style>
<!-- paywall styles injected above -->
<style>
/* override: paywall styles were above, nothing here */
</style>
<div id="paywall" class="pw-overlay" style="display:none">
  <div class="pw-card">
    <div class="pw-aurora"></div>
    <div class="pw-aurora2"></div>
    <div class="pw-lock">🔐</div>
    <div class="pw-title">解锁 AI 智能推荐</div>
    <div class="pw-sub">基于贝叶斯 · 马尔科夫 · 集成模型<br>综合置信度评分 · 每期精选推荐</div>
    <div class="pw-plans">
      <div class="pw-plan" id="plan-once" onclick="selectPlan('once')">
        <div class="plan-name">单次体验</div>
        <div class="plan-price">¥9.9</div>
        <div class="plan-desc">有效期 24 小时</div>
      </div>
      <div class="pw-plan featured" id="plan-monthly" onclick="selectPlan('monthly')">
        <div class="plan-name">⭐ 月度会员</div>
        <div class="plan-price">¥29.9</div>
        <div class="plan-desc">有效期 30 天</div>
      </div>
      <div class="pw-plan" id="plan-yearly" onclick="selectPlan('yearly')">
        <div class="plan-name">年度会员</div>
        <div class="plan-price">¥199</div>
        <div class="plan-desc">有效期 365 天</div>
      </div>
      <div class="pw-plan" id="plan-lifetime" onclick="selectPlan('lifetime')">
        <div class="plan-name">永久会员</div>
        <div class="plan-price">¥499</div>
        <div class="plan-desc">无限期使用</div>
      </div>
    </div>
    <div class="pw-divider"><span>付款后输入收到的 Token</span></div>
    <input class="pw-input" id="pw-token" placeholder="粘贴您的访问 Token（如 A3F8B2...）" maxlength="24">
    <button class="pw-btn" onclick="submitToken()">验 证 并 解 锁</button>
    <div class="pw-err" id="pw-err"></div>
    <div class="pw-note">
      付款请联系管理员获取 Token<br>
      已有 Token？直接粘贴上方即可 · <a href="/">返回主页</a>
    </div>
  </div>
</div>

<script>
// ── Paywall logic
let _selectedPlan = 'monthly';
function selectPlan(p){
  _selectedPlan=p;
  document.querySelectorAll('.pw-plan').forEach(el=>el.classList.remove('selected'));
  document.getElementById('plan-'+p).classList.add('selected');
}
selectPlan('monthly');

function showPaywall(){document.getElementById('paywall').style.display='flex';}
function hidePaywall(){document.getElementById('paywall').style.display='none';}

function submitToken(){
  const tok = document.getElementById('pw-token').value.trim();
  if(!tok){document.getElementById('pw-err').textContent='请输入 Token';return;}
  document.getElementById('pw-err').textContent='验证中…';
  fetch('/api/verify-token',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:tok})})
  .then(r=>r.json()).then(d=>{
    if(d.valid){
      localStorage.setItem('ssq_token', tok);
      hidePaywall();
      load();
    } else {
      document.getElementById('pw-err').textContent='❌ '+d.msg;
    }
  }).catch(()=>document.getElementById('pw-err').textContent='网络错误，请重试');
}

function checkAccess(cb){
  const tok = localStorage.getItem('ssq_token');
  if(!tok){showPaywall();return;}
  fetch('/api/verify-token',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:tok})})
  .then(r=>r.json()).then(d=>{
    if(d.valid){cb();}
    else{localStorage.removeItem('ssq_token');showPaywall();}
  }).catch(()=>cb()); // 网络异常时放行避免误拦
}

// ── original page logic (modified to require token)
</script>
<script>
checkAccess(load);
</script>
</body>
</html>"""

    def render_stats_html() -> str:
        return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>统计分析 - 双色球</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;font-size:12px;color:#333}
.top-bar{background:linear-gradient(135deg,#8e44ad,#6c3483);color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.top-bar h2{font-size:15px;font-weight:bold;margin-right:auto}
.nav-link{padding:5px 12px;background:rgba(255,255,255,0.2);color:#fff;border-radius:5px;text-decoration:none;font-size:12px}
.period-btn{padding:4px 10px;border:1px solid rgba(255,255,255,0.5);border-radius:3px;cursor:pointer;background:transparent;font-size:12px;color:#fff}
.period-btn.active{background:rgba(255,255,255,0.3)}
main{max-width:1200px;margin:0 auto;padding:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){main{grid-template-columns:1fr}}
.card{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
h3{font-size:13px;font-weight:bold;color:#2c3e50;margin-bottom:4px}
.subtitle{font-size:11px;color:#888;margin-bottom:12px}
.chart-wrap{overflow-x:auto}
.bar-chart{display:flex;align-items:flex-end;gap:2px;height:120px;border-bottom:1px solid #eee;padding-bottom:2px}
.bar-col{display:flex;flex-direction:column;align-items:center;min-width:14px}
.bar-col .bar{background:#8e44ad;border-radius:2px 2px 0 0;width:100%;min-height:2px}
.bar-col .bar.highlight{background:#e74c3c}
.bar-col .xlbl{font-size:8px;color:#aaa;margin-top:2px;writing-mode:vertical-rl;transform:rotate(180deg);height:22px}
.pie-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.pie-item{text-align:center;flex:1;min-width:60px}
.pie-bar{height:80px;display:flex;align-items:flex-end;justify-content:center;background:#f8f8f8;border-radius:4px;margin-bottom:4px}
.pie-inner{width:28px;background:#8e44ad;border-radius:3px 3px 0 0;min-height:3px}
.stat-summary{display:flex;gap:16px;margin-top:8px;font-size:12px;flex-wrap:wrap}
.stat-val{color:#8e44ad;font-weight:bold;font-size:14px}
</style>
</head>
<body>
<div class="top-bar">
  <h2>📊 统计特征分析</h2>
  <a href="/" class="nav-link">← 主页</a>
  <a href="/recommend" class="nav-link">智能推荐</a>
  <a href="/cooccur" class="nav-link">共现热图</a>
  <button class="period-btn active" id="btnall" onclick="setPeriod(0,this)">全部</button>
  <button class="period-btn" id="btn100" onclick="setPeriod(100,this)">近100期</button>
  <button class="period-btn" id="btn300" onclick="setPeriod(300,this)">近300期</button>
  <button class="period-btn" id="btn500" onclick="setPeriod(500,this)">近500期</button>
</div>
<main id="main-grid">
  <div style="grid-column:1/-1;text-align:center;padding:30px;color:#888" id="loading">加载中…</div>
</main>
<script>
let _period=0;
function setPeriod(n,el){
  _period=n;
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  load();
}
function load(){
  document.getElementById('loading').style.display='block';
  fetch('/api/stats-data?n='+_period).then(r=>r.json()).then(render);
}
function barChart(data, highlightFn, maxH=110){
  const maxY=Math.max(...data.map(d=>d.y),1);
  return '<div class="bar-chart">' + data.filter(d=>d.y>0||true).map(d=>{
    const h=Math.max(2,Math.round(d.y/maxY*maxH));
    const hl=highlightFn?highlightFn(d.x):'';
    return `<div class="bar-col" title="${d.x}: ${d.y}次">
      <div class="bar${hl?' highlight':''}" style="height:${h}px"></div>
      <div class="xlbl">${d.x}</div>
    </div>`;
  }).join('')+'</div>';
}
function pieChart(data, colors){
  const maxY=Math.max(...data.map(d=>d.y),1);
  return '<div class="pie-row">'+data.map((d,i)=>{
    const h=Math.max(3,Math.round(d.y/maxY*72));
    const c=colors?colors[i%colors.length]:'#8e44ad';
    return `<div class="pie-item">
      <div class="pie-bar"><div class="pie-inner" style="height:${h}px;background:${c}"></div></div>
      <div style="font-size:10px;color:#555">${d.x}奇</div>
      <div style="font-size:11px;font-weight:bold">${d.y}</div>
    </div>`;
  }).join('')+'</div>';
}
function render(d){
  if(d.error){document.getElementById('loading').textContent='暂无数据';return;}
  document.getElementById('loading').style.display='none';
  const g=document.getElementById('main-grid');
  const meanSum=d.sum.mean, stdSum=d.sum.std;
  g.innerHTML=`
  <div class="card">
    <h3>红球和值分布</h3>
    <div class="subtitle">6个红球之和 · 历史均值 ${d.sum.mean}，标准差 ${d.sum.std}，黄金区间 ${d.sum.mode_range}</div>
    <div class="chart-wrap">${barChart(d.sum.data, x=>x>=meanSum-stdSum&&x<=meanSum+stdSum)}</div>
    <div class="stat-summary"><span>均值 <span class="stat-val">${d.sum.mean}</span></span><span>标准差 <span class="stat-val">±${d.sum.std}</span></span><span>建议选和值在 <span class="stat-val">${d.sum.mode_range}</span> 范围内</span></div>
  </div>
  <div class="card">
    <h3>红球跨度分布</h3>
    <div class="subtitle">最大红球 - 最小红球 · 历史均值 ${d.span.mean}</div>
    <div class="chart-wrap">${barChart(d.span.data.filter(p=>p.x>=5), x=>x>=20&&x<=30)}</div>
    <div class="stat-summary"><span>均值跨度 <span class="stat-val">${d.span.mean}</span></span><span>高频区间 <span class="stat-val">20-30</span>（橙色）</span></div>
  </div>
  <div class="card">
    <h3>奇偶比分布</h3>
    <div class="subtitle">6个红球中奇数个数的历史分布 · 均值 ${d.odd.mean} 个奇数</div>
    ${pieChart(d.odd.data, ['#95a5a6','#3498db','#2ecc71','#e74c3c','#8e44ad','#e67e22','#1abc9c'])}
    <div class="stat-summary"><span>最优组合 <span class="stat-val">3奇3偶 或 4奇2偶</span></span></div>
  </div>
  <div class="card">
    <h3>大小号比分布</h3>
    <div class="subtitle">红球 &gt;16 为大号，≤16 为小号 · 均值 ${d.high.mean} 个大号</div>
    ${pieChart(d.high.data, ['#95a5a6','#3498db','#2ecc71','#e74c3c','#8e44ad','#e67e22','#1abc9c'])}
    <div class="stat-summary"><span>最优组合 <span class="stat-val">3大3小 或 2大4小</span></span></div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h3>蓝球历史频率分布</h3>
    <div class="subtitle">蓝球1-16各号码出现次数，共${d.total}期</div>
    <div class="chart-wrap"><div class="bar-chart" style="height:100px">
    ${Object.entries(d.blue_dist).map(([b,cnt])=>{
      const mx=Math.max(...Object.values(d.blue_dist));
      const h=Math.max(3,Math.round(cnt/mx*90));
      return `<div class="bar-col" style="min-width:28px" title="蓝球${b}: ${cnt}次">
        <div class="bar" style="height:${h}px;background:#2980b9"></div>
        <div style="font-size:9px;color:#555;margin-top:2px">${b}</div>
      </div>`;
    }).join('')}
    </div></div>
  </div>`;
}
load();
</script>
</body>
</html>"""

    # ── 共现热图页面 ──────────────────────────────────────
    def render_cooccur_html() -> str:
        return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>共现热图 - 双色球</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;font-size:11px;color:#333}
.top-bar{background:linear-gradient(135deg,#16a085,#0e6251);color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.top-bar h2{font-size:15px;font-weight:bold;margin-right:auto}
.nav-link{padding:5px 12px;background:rgba(255,255,255,0.2);color:#fff;border-radius:5px;text-decoration:none;font-size:12px}
.period-btn{padding:4px 10px;border:1px solid rgba(255,255,255,0.5);border-radius:3px;cursor:pointer;background:transparent;font-size:12px;color:#fff}
.period-btn.active{background:rgba(255,255,255,0.3)}
.info-bar{padding:8px 16px;background:#fff;border-bottom:1px solid #eee;font-size:11px;color:#888}
.legend{display:flex;align-items:center;gap:6px;padding:8px 16px;background:#fff;border-bottom:1px solid #eee}
.legend-bar{height:12px;width:160px;background:linear-gradient(to right,#eaf4fb,#1a5276);border-radius:2px}
.wrap{overflow:auto;padding:10px}
table{border-collapse:collapse}
td,th{width:22px;height:22px;font-size:9px;text-align:center;padding:0;border:none}
th.axis{background:#f7f7f7;font-weight:bold;font-size:10px;color:#555;position:sticky}
th.col-head{top:0;z-index:10}
th.row-head{left:0;z-index:9;min-width:24px}
td.heat{cursor:default}
#tooltip{position:fixed;background:rgba(0,0,0,.75);color:#fff;padding:5px 8px;border-radius:4px;font-size:11px;pointer-events:none;display:none;z-index:999}
</style>
</head>
<body>
<div class="top-bar">
  <h2>🔥 红球共现热图</h2>
  <a href="/" class="nav-link">← 主页</a>
  <a href="/recommend" class="nav-link">智能推荐</a>
  <a href="/stats-analysis" class="nav-link">统计分析</a>
  <button class="period-btn active" id="btnall" onclick="setPeriod(0,this)">全部</button>
  <button class="period-btn" id="btn300" onclick="setPeriod(300,this)">近300期</button>
  <button class="period-btn" id="btn100" onclick="setPeriod(100,this)">近100期</button>
</div>
<div class="info-bar" id="info-bar">加载中…</div>
<div class="legend">
  <span style="font-size:11px">共现次数：低</span>
  <div class="legend-bar"></div>
  <span style="font-size:11px">高</span>
  <span style="margin-left:20px;font-size:11px;color:#888">对角线 = 该号码单独出现次数；悬停查看详情</span>
</div>
<div class="wrap" id="heat-wrap">加载中…</div>
<div id="tooltip"></div>
<script>
let _period=0;
function setPeriod(n,el){
  _period=n;
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  load();
}
function load(){
  document.getElementById('info-bar').textContent='计算中…';
  fetch('/api/cooccur-data?n='+_period).then(r=>r.json()).then(render);
}
function heatColor(v,max){
  const t=v/Math.max(max,1);
  const r=Math.round(26+(1-t)*200), g=Math.round(82+(1-t)*150), b=Math.round(118+(1-t)*130);
  return `rgb(${255-Math.round(t*200)},${255-Math.round(t*160)},${255-Math.round(t*80)})`;
}
function heatColorDiag(v,max){
  const t=v/Math.max(max,1);
  return `rgba(192,57,43,${0.2+t*0.7})`;
}
function render(d){
  if(d.error){document.getElementById('heat-wrap').textContent='暂无数据';return;}
  document.getElementById('info-bar').textContent=
    `共 ${d.total} 期数据 · 数值=两号码同期出现次数 · 对角线=该号码出现总次数`;
  const m=d.matrix, mx=d.max_val;
  // 非对角线最大值（用于颜色归一）
  let offMax=0;
  for(let i=0;i<33;i++) for(let j=0;j<33;j++) if(i!==j) offMax=Math.max(offMax,m[i][j]);
  let html='<table><thead><tr><th class="axis col-head row-head"></th>';
  for(let b=1;b<=33;b++) html+=`<th class="axis col-head">${b}</th>`;
  html+='</tr></thead><tbody>';
  for(let i=0;i<33;i++){
    html+=`<tr><th class="axis row-head">${i+1}</th>`;
    for(let j=0;j<33;j++){
      const v=m[i][j];
      const isDiag=(i===j);
      const bg=isDiag?heatColorDiag(v,mx):heatColor(v,offMax);
      const txt=v>0?v:'';
      html+=`<td class="heat" style="background:${bg}" data-i="${i+1}" data-j="${j+1}" data-v="${v}">${isDiag?`<b>${v}</b>`:txt}</td>`;
    }
    html+='</tr>';
  }
  html+='</tbody></table>';
  document.getElementById('heat-wrap').innerHTML=html;

  const tip=document.getElementById('tooltip');
  document.querySelectorAll('.heat').forEach(td=>{
    td.addEventListener('mousemove',e=>{
      const i=td.dataset.i, j=td.dataset.j, v=td.dataset.v;
      const txt=i===j?`红球 ${i} 出现 ${v} 次`:
        `红球 ${i} 与 ${j} 同期出现 ${v} 次`;
      tip.textContent=txt;
      tip.style.display='block';
      tip.style.left=(e.clientX+10)+'px';
      tip.style.top=(e.clientY-24)+'px';
    });
    td.addEventListener('mouseleave',()=>tip.style.display='none');
  });
}
load();
</script>
</body>
</html>"""

    # ── 管理员页面 ────────────────────────────────────────
    def render_admin_html() -> str:
        return r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>管理后台</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'PingFang SC',sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh;padding:24px}
h2{font-size:18px;font-weight:700;margin-bottom:20px;background:linear-gradient(135deg,#0a84ff,#bf5af2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:24px;margin-bottom:16px;max-width:560px}
label{display:block;font-size:12px;color:rgba(255,255,255,.5);margin-bottom:5px}
input,select{width:100%;padding:10px 14px;border-radius:10px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.06);color:#fff;font-size:14px;margin-bottom:14px;outline:none;font-family:inherit}
button{padding:12px 24px;border-radius:10px;border:none;cursor:pointer;font-size:14px;font-weight:600;background:linear-gradient(135deg,#0a84ff,#bf5af2);color:#fff;width:100%;font-family:inherit;transition:all .2s}
button:hover{transform:translateY(-1px);box-shadow:0 0 24px rgba(10,132,255,.4)}
.result{margin-top:14px;padding:14px;background:rgba(0,0,0,.4);border-radius:10px;font-family:monospace;font-size:13px;color:#32d74b;border:1px solid rgba(50,215,75,.2);display:none}
.result.err{color:#ff375f;border-color:rgba(255,55,95,.2)}
.tok{display:flex;align-items:center;gap:8px;padding:7px 10px;background:rgba(10,132,255,.08);border:1px solid rgba(10,132,255,.2);border-radius:7px;margin-bottom:5px}
.tok span{flex:1;font-family:monospace;font-size:13px}
.cpbtn{padding:3px 10px;background:rgba(10,132,255,.2);border:1px solid rgba(10,132,255,.4);color:#5ac8fa;border-radius:6px;cursor:pointer;font-size:11px;flex-shrink:0;font-family:inherit;width:auto}
a.back{color:rgba(255,255,255,.35);font-size:12px;display:block;margin-bottom:16px;text-decoration:none}
a.back:hover{color:#0a84ff}
</style></head><body>
<a href="/" class="back">← 返回主页</a>
<h2>⚙️ 管理后台 · Token 管理</h2>
<div class="card">
  <label>管理员密钥（ADMIN_KEY 环境变量）</label>
  <input id="ak" type="password" placeholder="输入管理员密钥">
  <label>套餐</label>
  <select id="pl">
    <option value="trial">试用 1小时</option>
    <option value="once">单次 24小时</option>
    <option value="monthly" selected>月度 30天</option>
    <option value="yearly">年度 365天</option>
    <option value="lifetime">永久</option>
  </select>
  <label>生成数量</label>
  <input id="cnt" type="number" value="1" min="1" max="50">
  <button onclick="gen()">生成 Token</button>
  <div id="res" class="result"></div>
</div>
<script>
function gen(){
  const ak=document.getElementById('ak').value.trim();
  if(!ak){alert('请输入密钥');return;}
  const pl=document.getElementById('pl').value;
  const cnt=parseInt(document.getElementById('cnt').value)||1;
  fetch('/api/admin/generate-token',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({admin_key:ak,plan:pl,count:cnt})})
  .then(r=>r.json()).then(d=>{
    const el=document.getElementById('res');
    el.style.display='block';
    if(!d.ok){el.className='result err';el.textContent='❌ '+d.msg;return;}
    el.className='result';
    el.innerHTML='✅ 已生成 '+d.tokens.length+' 个Token  套餐：'+d.plan+'  有效期：'+d.expire+'\n\n'
      +d.tokens.map((t,i)=>`<div class="tok"><span>${i+1}. ${t}</span>
        <button class="cpbtn" onclick="navigator.clipboard.writeText('${t}').then(()=>this.textContent='✓已复制')">复制</button></div>`).join('');
  }).catch(()=>{const el=document.getElementById('res');el.style.display='block';el.className='result err';el.textContent='网络错误';});
}
</script></body></html>"""

    # ── HTTP 请求处理器 ───────────────────────────────────
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data: dict, code: int = 200):
            body = _json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_str: str):
            body = html_str.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return _json.loads(self.rfile.read(length))

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)
            if path == "/trend":
                self._send_html(render_trend_html())
            elif path == "/api/trend-data":
                n = int(qs.get("n", [30])[0])
                start = qs.get("start", [""])[0]
                end = qs.get("end", [""])[0]
                self._send_json(get_trend_data(n, start, end))
            elif path == "/hot-cold":
                self._send_html(render_hot_cold_html())
            elif path == "/api/hot-cold-data":
                n = int(qs.get("n", [0])[0])
                start = qs.get("start", [""])[0]
                end = qs.get("end", [""])[0]
                self._send_json(get_hot_cold_data(n, start, end))
            elif path == "/recommend":
                self._send_html(render_recommend_html())
            elif path == "/api/recommend-data":
                n = int(qs.get("n", [100])[0])
                self._send_json(get_recommend_data(n))
            elif path == "/stats-analysis":
                self._send_html(render_stats_html())
            elif path == "/api/stats-data":
                n = int(qs.get("n", [0])[0])
                self._send_json(get_stats_data(n))
            elif path == "/cooccur":
                self._send_html(render_cooccur_html())
            elif path == "/api/cooccur-data":
                n = int(qs.get("n", [0])[0])
                self._send_json(get_cooccur_data(n))
            elif path == "/admin":
                self._send_html(render_admin_html())
            elif path == "/api/snapshot":
                self._handle_snapshot()
            else:
                self._send_html(render_html())

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path

            if path == "/api/refresh":
                self._handle_refresh()
            elif path == "/api/purchase":
                self._handle_purchase()
            elif path == "/api/result":
                self._handle_result()
            elif path == "/api/delete":
                self._handle_delete()
            elif path == "/api/verify-token":
                data = self._read_body()
                result = verify_token(str(data.get("token", "")))
                self._send_json(result)
            elif path == "/api/admin/generate-token":
                data = self._read_body()
                if data.get("admin_key") != ADMIN_KEY:
                    self._send_json({"ok": False, "msg": "Unauthorized"}, 403)
                    return
                plan = data.get("plan", "monthly")
                count = max(1, int(data.get("count", 1)))
                import time
                tokens_out = [generate_token(plan) for _ in range(count)]
                expire_days = {"trial": "1h", "once": "24h", "monthly": "30天",
                               "yearly": "365天", "lifetime": "永久"}.get(plan, "30天")
                self._send_json({"ok": True, "tokens": tokens_out,
                                 "plan": plan, "expire": expire_days})
            else:
                self._send_json({"ok": False, "msg": "Unknown endpoint"}, 404)

        def _build_live_snapshot(self) -> dict:
            """构建实时快照（供 /api/snapshot 和刷新后返回）。"""
            snap = build_snapshot()
            red_cols_local = ["r1", "r2", "r3", "r4", "r5", "r6"]

            def bh(nums, cls):
                return "".join(f'<span class="{cls}">{int(n):02d}</span>' for n in nums)

            latest10 = snap.get("latest10", [])
            rows_html = ""
            for r in reversed(latest10):
                reds = bh([r[c] for c in red_cols_local], "red")
                blue = f'<span class="blue">{int(r["blue"]):02d}</span>'
                rows_html += f"<tr><td>{r['issue']}</td><td>{r['date']}</td><td>{reds}</td><td>{blue}</td></tr>"

            top_red_html = "".join(
                f'<span class="red">{n:02d}</span><small class="freq-cnt">×{cnt}</small> '
                for n, cnt in snap.get("top10_red", [])
            )
            top_blue_html = "".join(
                f'<span class="blue">{n:02d}</span><small class="freq-cnt">×{cnt}</small> '
                for n, cnt in snap.get("top5_blue", [])
            )

            # 实时生成轻量预测（无需ML模型，纯统计）
            pred_html = ""
            try:
                rec = get_recommend_data(n=100)
                combos = rec.get("combos", [])[:5]
                alerts = rec.get("alerts", [])
                last_blue = rec.get("last_blue", "?")
                bp = rec.get("blue_probs", {})
                top_blue_pred = sorted(bp.items(), key=lambda x: x[1], reverse=True)[:3]

                lines = []
                lines.append(f"▶ 基于近100期综合分析（贝叶斯·遗漏·动量）")
                if alerts:
                    ab = "、".join(f"{a['ball']:02d}（遗漏{a['miss']}期）" for a in alerts[:4])
                    lines.append(f"⚡ 超长遗漏预警：{ab}")
                lines.append(f"🔵 蓝球高概率：" + "  ".join(f"{b}号{p}%" for b, p in top_blue_pred))
                lines.append("")
                lines.append("▶ 推荐号码（按综合置信度排序）：")
                for i, c in enumerate(combos, 1):
                    red_str = " ".join(f"{b:02d}" for b in c["red"])
                    lines.append(
                        f"  第{i:02d}注：红球[{red_str}] + 蓝球[{c['blue']:02d}]  "
                        f"和值:{c['sum']}  {c['odd_even']}  三区{c['z1']}/{c['z2']}/{c['z3']}  {c['balance']}"
                    )
                pred_text = "\n".join(lines)
                pred_html = pred_text
            except Exception as ex:
                logger.warning("实时预测生成失败：%s", ex)
                pred_html = snap.get("pred_content", "")

            return {
                "total": snap.get("total", 0),
                "max_issue": str(snap.get("max_issue", "")),
                "max_date": str(snap.get("max_date", "")),
                "latest_html": rows_html,
                "top_red_html": top_red_html,
                "top_blue_html": top_blue_html,
                "pred_html": pred_html,
            }

        def _handle_snapshot(self):
            try:
                self._send_json(self._build_live_snapshot())
            except Exception as e:
                self._send_json({"error": str(e)})

        def _handle_refresh(self):
            with state_lock:
                if state["refreshing"]:
                    self._send_json({"ok": False, "msg": "正在刷新中，请稍候..."})
                    return
                state["refreshing"] = True

            try:
                mgr = DataManager()
                df, new_count = mgr.update()
                if df.empty:
                    self._send_json({"ok": False, "msg": "数据获取失败，请检查网络"})
                    return

                # 自动对照开奖号码，更新未核销的购彩记录
                auto_matched = _auto_check_results(df)

                # 构建实时快照（含最新预测）
                snap = self._build_live_snapshot()

                self._send_json({
                    "ok": True,
                    "new_periods": new_count,
                    "total": len(df),
                    "max_issue": str(df["issue"].max()),
                    "latest_html": snap["latest_html"],
                    "top_red_html": snap["top_red_html"],
                    "top_blue_html": snap["top_blue_html"],
                    "pred_html": snap["pred_html"],
                    "auto_matched": auto_matched,
                })
            except Exception as e:
                logger.exception("刷新失败：%s", e)
                self._send_json({"ok": False, "msg": str(e)})
            finally:
                with state_lock:
                    state["refreshing"] = False

        def _handle_purchase(self):
            data = self._read_body()
            issue = str(data.get("issue", "")).strip()
            tickets = max(1, int(data.get("tickets", 1)))
            ticket_type = data.get("type", "单式")  # "单式" or "复式"
            my_red = [int(x) for x in data.get("my_red", [])]
            raw_blue = data.get("my_blue", 0)

            if ticket_type == "复式":
                # 复式：my_blue is a list; my_red may be 7-12 balls
                if isinstance(raw_blue, list):
                    my_blue = [int(b) for b in raw_blue]
                else:
                    my_blue = [int(raw_blue)]
                if len(my_red) < 7 or len(my_red) > 12:
                    self._send_json({"ok": False, "msg": "复式红球数量应为7-12个"})
                    return
                if not my_blue or any(not (1 <= b <= 16) for b in my_blue):
                    self._send_json({"ok": False, "msg": "蓝球号码不合法"})
                    return
                from math import comb
                combo_count = comb(len(my_red), 6) * len(my_blue)
            else:
                my_blue = int(raw_blue)
                if len(my_red) != 6 or not (1 <= my_blue <= 16):
                    self._send_json({"ok": False, "msg": "数据不合法"})
                    return
                combo_count = 1

            purchases = load_purchases()
            import uuid
            record = {
                "id": str(uuid.uuid4())[:8],
                "time": datetime.now().isoformat(),
                "issue": issue,
                "tickets": tickets,
                "type": ticket_type,
                "combo_count": combo_count,
                "my_red": sorted(my_red),
                "my_blue": my_blue,
                "actual_red": None,
                "actual_blue": None,
                "hit_red": None,
                "hit_blue": None,
                "prize": None,
                "prize_money": None,
            }
            purchases.append(record)
            save_purchases(purchases)
            self._send_json({"ok": True, "combo_count": combo_count})

        def _handle_result(self):
            from itertools import combinations as _comb
            data = self._read_body()
            pid = data.get("id")
            actual_red = [int(x) for x in data.get("actual_red", [])]
            actual_blue = int(data.get("actual_blue", 0))

            if len(actual_red) != 6 or not (1 <= actual_blue <= 16):
                self._send_json({"ok": False, "msg": "开奖号码不合法"})
                return

            purchases = load_purchases()
            updated = False
            p_ref = None
            for p in purchases:
                if p["id"] == pid:
                    ticket_type = p.get("type", "单式")
                    act_set = set(actual_red)

                    if ticket_type == "复式":
                        blue_list = p["my_blue"] if isinstance(p["my_blue"], list) else [p["my_blue"]]
                        total_prize = 0
                        winning_count = 0
                        best_prize_name = "未中奖"
                        best_prize_money = 0
                        max_hr, max_hb = 0, False
                        for combo in _comb(p["my_red"], 6):
                            hr = len(set(combo) & act_set)
                            for b in blue_list:
                                hb = (b == actual_blue)
                                pname, pmoney = prize_level(hr, hb)
                                total_prize += pmoney
                                if pmoney > 0:
                                    winning_count += 1
                                if pmoney > best_prize_money:
                                    best_prize_money = pmoney
                                    best_prize_name = pname
                                if hr > max_hr or (hr == max_hr and hb and not max_hb):
                                    max_hr, max_hb = hr, hb
                        p["actual_red"] = sorted(actual_red)
                        p["actual_blue"] = actual_blue
                        p["hit_red"] = max_hr
                        p["hit_blue"] = max_hb
                        p["prize"] = best_prize_name
                        p["winning_count"] = winning_count
                        p["prize_money"] = total_prize * p.get("tickets", 1)
                    else:
                        my_blue = p["my_blue"]
                        if isinstance(my_blue, list):
                            my_blue = my_blue[0]
                        hr = len(set(p["my_red"]) & act_set)
                        hb = (my_blue == actual_blue)
                        pname, pmoney = prize_level(hr, hb)
                        p["actual_red"] = sorted(actual_red)
                        p["actual_blue"] = actual_blue
                        p["hit_red"] = hr
                        p["hit_blue"] = hb
                        p["prize"] = pname
                        p["winning_count"] = 1 if pmoney > 0 else 0
                        p["prize_money"] = pmoney * p.get("tickets", 1)

                    updated = True
                    p_ref = p
                    break

            if not updated:
                self._send_json({"ok": False, "msg": "记录未找到"})
                return

            save_purchases(purchases)

            # 触发模型权重优化（后台线程，不阻塞响应）
            threading.Thread(
                target=optimize_weights_from_feedback,
                args=(purchases,),
                daemon=True,
            ).start()

            self._send_json({
                "ok": True,
                "hit_red": p_ref["hit_red"],
                "hit_blue": p_ref["hit_blue"],
                "prize": p_ref["prize"],
                "prize_money": p_ref["prize_money"],
                "winning_count": p_ref.get("winning_count", 0),
                "combo_count": p_ref.get("combo_count", 1),
            })

        def _handle_delete(self):
            data = self._read_body()
            pid = data.get("id")
            purchases = load_purchases()
            purchases = [p for p in purchases if p["id"] != pid]
            save_purchases(purchases)
            self._send_json({"ok": True})

        def log_message(self, fmt, *a):
            pass  # 静默日志

    # 启动时自动核销已有开奖数据
    try:
        _init_df = DataManager().load_local()
        if _init_df is not None and not _init_df.empty:
            _n = _auto_check_results(_init_df)
            if _n:
                logger.info("启动时自动核销 %d 条购彩记录", _n)
    except Exception:
        pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n✓ 控制面板已启动 → http://localhost:{port}")
    print("  功能：数据刷新 · 购彩录入 · 命中追踪 · 模型优化")
    print("  按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def cmd_update(args) -> int:
    """增量更新数据并生成新预测。"""
    from auto_updater import AutoUpdater

    print("\n执行增量更新...")
    updater = AutoUpdater()
    summary = updater.run_update_cycle(force_retrain=args.retrain)

    print(f"\n更新摘要：")
    print(f"  新增期数：{summary['new_periods']}")
    print(f"  模型重训：{'是' if summary['retrained'] else '否'}")
    print(f"  预测生成：{'是' if summary['prediction_generated'] else '否'}")
    if summary.get("next_issue"):
        print(f"  预测期号：{summary['next_issue']}")
    if summary.get("report_file"):
        print(f"  报告文件：{summary['report_file']}")
    if summary.get("error"):
        print(f"  错误信息：{summary['error']}")
        return 1

    return 0


def cmd_predict(args) -> int:
    """生成下期预测号码。"""
    from data_scraper import DataManager
    from data_analyzer import DataAnalyzer, build_features
    from prediction_model import EnsemblePredictor, format_prediction_output
    from auto_updater import save_prediction, _load_model, _save_model

    print("\n正在生成预测...")

    mgr = DataManager()
    df = mgr.df
    if df.empty:
        print("错误：未找到历史数据，请先运行 `python main.py init`")
        return 1

    # 尝试加载已有模型
    model = _load_model()
    if model is None or args.retrain:
        print("  训练模型中（首次可能较慢）...")
        t0 = time.time()

        try:
            feat_df = build_features(df, windows=[10, 20, 30])
        except Exception as e:
            logger.warning("特征构建失败：%s", e)
            feat_df = None

        model = EnsemblePredictor()
        model.fit(df, features_df=feat_df)
        _save_model(model)
        print(f"  训练完成（用时 {time.time() - t0:.1f}s）")

    # 最新特征
    try:
        feat_df = build_features(df, windows=[10, 20, 30])
        latest_feat = feat_df.tail(1).reset_index(drop=True)
    except Exception:
        latest_feat = None

    n_single = args.count or NUM_SINGLE_PREDICTIONS
    pred = model.predict(
        n_single=n_single,
        complex_specs=COMPLEX_SPECS,
        latest_features=latest_feat,
    )

    output = format_prediction_output(pred)
    print("\n" + output)

    # 保存
    if not args.no_save:
        latest_issue = df["issue"].max()
        try:
            next_issue = str(int(latest_issue) + 1)
        except ValueError:
            next_issue = latest_issue + "_next"
        save_prediction(pred, next_issue)

        report_file = REPORT_DIR / f"prediction_{next_issue}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.write_text(output, encoding="utf-8")
        print(f"\n报告已保存：{report_file}")

    return 0


def cmd_analyze(args) -> int:
    """运行统计分析并输出报告。"""
    from data_scraper import DataManager
    from data_analyzer import DataAnalyzer, visualize_frequency, visualize_heatmap

    print("\n运行统计分析...")

    mgr = DataManager()
    df = mgr.df
    if df.empty:
        print("错误：未找到历史数据，请先运行 `python main.py init`")
        return 1

    analyzer = DataAnalyzer(df)

    # 文本报告
    report = analyzer.full_report()
    print("\n" + report)

    # 可视化（可选）
    if not args.no_plot:
        print("\n生成可视化图表...")
        try:
            freq_plot = REPORT_DIR / "frequency_analysis.png"
            visualize_frequency(analyzer.frequency(), save_path=str(freq_plot))
            print(f"  频率图：{freq_plot}")

            heat_plot = REPORT_DIR / "cooccurrence_heatmap.png"
            visualize_heatmap(analyzer.cooccurrence(), save_path=str(heat_plot))
            print(f"  热力图：{heat_plot}")
        except Exception as e:
            logger.warning("可视化生成失败：%s", e)

    # 保存文本报告
    report_file = REPORT_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"\n分析报告已保存：{report_file}")

    return 0


def cmd_backtest(args) -> int:
    """回测评估预测模型性能。"""
    from data_scraper import DataManager
    from prediction_model import Backtester

    print("\n运行回测评估...")

    mgr = DataManager()
    df = mgr.df
    if df.empty:
        print("错误：未找到历史数据，请先运行 `python main.py init`")
        return 1

    n_test = min(args.periods, len(df) - 100)
    if n_test < 10:
        print(f"历史数据不足，无法进行回测（需要至少110期，当前 {len(df)} 期）。")
        return 1

    print(f"  回测期数：{n_test}")
    print("  训练中（每期都重新训练，可能需要较长时间）...")

    bt = Backtester()
    result = bt.evaluate(df, n_test=n_test)

    print(f"\n【回测结果（统计学习模型）】")
    print(f"  测试期数：{result['total_tests']}")
    print(f"  平均红球命中：{result['avg_red_hits']:.3f} 个/注")
    print(f"  蓝球命中率：{result['blue_hit_rate'] * 100:.2f}% （随机期望：6.25%）")
    print(f"  投入成本：¥{result['total_cost']}")
    print(f"  获奖总额：¥{result['total_prize']}")
    print(f"  ROI：{result['roi']:.2f}%")
    print(f"  奖级分布：{result['prize_distribution']}")

    # 与随机对比
    print("\n  对比随机选号...")
    rand_result = bt.compare_random(df, n_test=n_test)
    print(f"  随机平均红球命中：{rand_result['avg_red_hits']:.3f} 个/注")
    print(f"  随机蓝球命中率：{rand_result['blue_hit_rate'] * 100:.2f}%")
    print(f"  随机 ROI：{rand_result['roi']:.2f}%")

    # 理论期望
    print("\n  理论随机期望：")
    print("    平均红球命中 ≈ 6×6/33 ≈ 1.091 个/注")
    print("    蓝球命中率 ≈ 1/16 ≈ 6.25%")
    print("    理论 ROI ≈ -50%（彩票整体返奖率约50%）")

    # 保存报告
    report_lines = [
        "双色球预测系统 — 回测评估报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"回测期数：{n_test}",
        f"",
        "【模型结果】",
        f"平均红球命中：{result['avg_red_hits']:.3f}",
        f"蓝球命中率：{result['blue_hit_rate'] * 100:.2f}%",
        f"ROI：{result['roi']:.2f}%",
        f"",
        "【随机基准】",
        f"平均红球命中：{rand_result['avg_red_hits']:.3f}",
        f"蓝球命中率：{rand_result['blue_hit_rate'] * 100:.2f}%",
        f"ROI：{rand_result['roi']:.2f}%",
        "",
        "⚠ 说明：彩票开奖为随机过程，模型与随机选号长期ROI均为负。",
        "  本回测结果仅具统计参考价值，不构成任何投资建议。",
    ]
    report_file = REPORT_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n回测报告已保存：{report_file}")

    return 0


def cmd_report(args) -> int:
    """生成综合报告（分析 + 预测 + 历史验证）。"""
    from data_scraper import DataManager
    from data_analyzer import DataAnalyzer
    from auto_updater import verify_prediction

    print("\n生成综合报告...")

    mgr = DataManager()
    df = mgr.df
    if df.empty:
        print("错误：未找到历史数据，请先运行 `python main.py init`")
        return 1

    analyzer = DataAnalyzer(df)
    analysis_text = analyzer.full_report()

    verify_df = verify_prediction(df)
    verify_text = ""
    if verify_df is not None and not verify_df.empty:
        verify_text = (
            f"\n【历史预测验证】\n"
            f"  验证期数：{len(verify_df)}\n"
            f"  平均红球命中：{verify_df['hit_red'].mean():.2f}\n"
            f"  蓝球命中率：{verify_df['hit_blue'].mean() * 100:.1f}%\n"
            f"  奖级分布：{verify_df['prize'].value_counts().to_dict()}\n"
        )

    full_report = "\n".join([
        DISCLAIMER,
        analysis_text,
        verify_text,
        f"\n报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    report_file = REPORT_DIR / f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.write_text(full_report, encoding="utf-8")

    print(analysis_text)
    if verify_text:
        print(verify_text)
    print(f"\n完整报告已保存：{report_file}")

    return 0


def cmd_daemon(args) -> int:
    """启动定时守护进程。"""
    from auto_updater import start_scheduler

    print("\n启动定时守护进程（按 Ctrl+C 停止）...")
    print("  调度策略：每天 21:30 检查并更新")
    print("  开奖日：周二、周四、周日")
    start_scheduler(run_now=args.run_now)
    return 0


# ════════════════════════════════════════════════════════
# CLI 参数解析
# ════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssq",
        description=(
            "双色球预测系统 — 概率统计研究工具\n"
            "⚠ 仅供学术研究，彩票具有高度随机性，请理性购彩。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="输出详细日志"
    )

    sub = parser.add_subparsers(title="子命令", dest="command")

    # init
    p_init = sub.add_parser("init", help="初始化：从网络获取历史数据")
    p_init.add_argument("--refresh", action="store_true", help="强制重新获取（忽略本地缓存）")
    p_init.set_defaults(func=cmd_init)

    # update
    p_upd = sub.add_parser("update", help="增量更新数据并生成新预测")
    p_upd.add_argument("--retrain", action="store_true", help="强制重训练模型")
    p_upd.set_defaults(func=cmd_update)

    # predict
    p_pred = sub.add_parser("predict", help="生成下期预测号码")
    p_pred.add_argument("--count", "-n", type=int, default=None, help="单式预测注数（默认10）")
    p_pred.add_argument("--retrain", action="store_true", help="强制重训练模型")
    p_pred.add_argument("--no-save", action="store_true", help="不保存预测结果")
    p_pred.set_defaults(func=cmd_predict)

    # analyze
    p_ana = sub.add_parser("analyze", help="运行统计分析")
    p_ana.add_argument("--no-plot", action="store_true", help="不生成可视化图表")
    p_ana.set_defaults(func=cmd_analyze)

    # backtest
    p_bt = sub.add_parser("backtest", help="回测评估模型性能")
    p_bt.add_argument("--periods", "-p", type=int, default=50, help="回测期数（默认50）")
    p_bt.set_defaults(func=cmd_backtest)

    # report
    p_rep = sub.add_parser("report", help="生成综合分析报告")
    p_rep.set_defaults(func=cmd_report)

    # daemon
    p_dm = sub.add_parser("daemon", help="启动定时守护进程")
    p_dm.add_argument("--run-now", action="store_true", help="立即执行一次后再定时")
    p_dm.set_defaults(func=cmd_daemon)

    return parser


# ════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=getattr(args, "verbose", False))

    print(DISCLAIMER)

    if not hasattr(args, "func"):
        # 无子命令时默认执行 predict
        if len(sys.argv) == 1:
            print("提示：未指定子命令，使用 --help 查看用法。")
            print("快速开始：")
            print("  python main.py init        # 第一步：获取历史数据")
            print("  python main.py predict     # 生成预测")
            print("  python main.py analyze     # 统计分析")
            print("  python main.py backtest    # 回测评估")
            parser.print_help()
            sys.exit(0)
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n操作已取消。")
        sys.exit(0)
    except Exception as e:
        logger.exception("程序异常退出：%s", e)
        print(f"\n错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
