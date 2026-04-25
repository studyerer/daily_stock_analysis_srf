# -*- coding: utf-8 -*-
"""
===================================
Market Scanner — 全市场量化预筛选
===================================

两阶段流水线的 Stage 1：
- 批量拉取 Tushare 全市场日线（日期维度，一次调用 ≈ 全市场 ~5000 只股票）
- 本地 pandas 计算 MA / 量能 / 动量指标
- 规则分层过滤 + 打分排序
- 输出 Top N 候选股到 data/scan_candidates_YYYYMMDD.json

设计要点：
- 30 个交易日仅需 30 次 API 调用 + stock_basic 1 次 + daily_basic 1 次
- 全部指标计算在 pandas 内完成，无 LLM 调用
- 预计运行耗时 2~5 分钟
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 默认输出目录
DEFAULT_OUTPUT_DIR = "data"

# 过滤参数默认值
DEFAULT_FILTERS: Dict = {
    "min_market_cap_yi": 20,     # 最小总市值（亿元）
    "min_list_days": 90,         # 上市天数下限
    "turnover_min": 0.5,         # 换手率下限 %
    "turnover_max": 20.0,        # 换手率上限 %
    "bias_ma5_max": 10.0,        # MA5 乖离率绝对值上限 %
    "change_5d_max": 20.0,       # 5日涨幅上限 %
    "volume_ratio_min": 0.8,     # 量比下限
    "top_n": 30,                 # 输出候选数
    "exclude_bj": True,          # 排除北交所
    "exclude_cyb": True,         # 排除创业板（300/301, 需要权限）
    "exclude_kcb": True,         # 排除科创板（688/689, 需要权限）
    "exclude_st": True,          # 排除 ST / *ST / 退市整理
    "history_days": 30,          # 拉取历史天数（至少 20 以满足 MA20）
}


def _get_tushare_api():
    """获取已 patch 过端点的 Tushare Pro API 实例（复用 TushareFetcher 逻辑）"""
    from data_provider.tushare_fetcher import TushareFetcher

    fetcher = TushareFetcher()
    if not fetcher.is_available():
        raise RuntimeError("Tushare API 不可用，请检查 TUSHARE_TOKEN 配置")
    return fetcher._api, fetcher


def _get_recent_trade_dates(api, n: int = 30) -> List[str]:
    """获取最近 N 个 A 股交易日列表（降序：最新在前）"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - pd.Timedelta(days=max(n * 2, 60))).strftime("%Y%m%d")
    cal = api.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    dates = sorted(cal["cal_date"].tolist(), reverse=True)
    return dates[:n]


def _fetch_all_daily(api, fetcher, trade_dates: List[str]) -> pd.DataFrame:
    """批量拉取多个交易日的全市场日线（OHLCV）"""
    frames: List[pd.DataFrame] = []
    total = len(trade_dates)
    for i, date in enumerate(trade_dates):
        try:
            fetcher._check_rate_limit()
            df = api.daily(trade_date=date)
            if df is not None and not df.empty:
                frames.append(df)
            count = len(df) if df is not None else 0
            logger.info(f"[Scanner] 拉取日线 {date} ({i + 1}/{total})，{count} 只股票")
        except Exception as e:
            logger.warning(f"[Scanner] 拉取 {date} 失败: {e}")

    if not frames:
        raise RuntimeError("未获取到任何日线数据")

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# 公开接口
# ------------------------------------------------------------------


def _default_output_path() -> str:
    """按日期生成输出路径，如 data/scan_candidates_20260425.json"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(DEFAULT_OUTPUT_DIR, f"scan_candidates_{today}.json")


def scan_market(
    top_n: int = 30,
    output_path: Optional[str] = None,
    filters: Optional[Dict] = None,
) -> List[str]:
    """
    全市场扫描，返回候选股代码列表。

    流程:
    1. 获取 A 股上市列表，排除 ST / 北交所 / 新股
    2. 拉取最近 30 个交易日全市场日线（30 次 API 调用）
    3. 拉取最新交易日 daily_basic（PE/PB/换手率/市值，1 次调用）
    4. 本地计算 MA5/10/20、乖离率、量比、5日涨幅
    5. 规则分层过滤
    6. 打分排序 → Top N
    7. 保存 JSON 结果
    """
    t0 = time.time()
    if output_path is None:
        output_path = _default_output_path()
    f: Dict = {**DEFAULT_FILTERS, **(filters or {})}
    f["top_n"] = top_n

    api, fetcher = _get_tushare_api()

    # ================================================================
    # Step 1: 股票列表
    # ================================================================
    logger.info("[Scanner] Step 1/6: 获取全市场股票列表...")
    fetcher._check_rate_limit()
    stock_list = api.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name,list_date,market",
    )
    total_listed = len(stock_list)

    # 排除 ST / 退市整理
    if f.get("exclude_st", True):
        stock_list = stock_list[~stock_list["name"].str.contains("ST|退", na=False)]

    # 排除北交所
    if f.get("exclude_bj", True):
        stock_list = stock_list[~stock_list["ts_code"].str.endswith(".BJ")]

    # 排除创业板（300xxx / 301xxx）
    if f.get("exclude_cyb", True):
        stock_list = stock_list[stock_list["market"] != "创业板"]

    # 排除科创板（688xxx / 689xxx）
    if f.get("exclude_kcb", True):
        stock_list = stock_list[stock_list["market"] != "科创板"]

    # 排除上市不足 N 天
    stock_list["list_date_dt"] = pd.to_datetime(stock_list["list_date"], format="%Y%m%d")
    stock_list = stock_list[
        (pd.Timestamp.now() - stock_list["list_date_dt"]).dt.days >= f["min_list_days"]
    ]

    valid_codes = set(stock_list["ts_code"].tolist())
    code_name_map = dict(zip(stock_list["ts_code"], stock_list["name"]))
    excluded = "/".join(
        x for x, flag in [
            ("ST", f.get("exclude_st")),
            ("北交所", f.get("exclude_bj")),
            ("创业板", f.get("exclude_cyb")),
            ("科创板", f.get("exclude_kcb")),
        ] if flag
    )
    logger.info(
        f"[Scanner] 有效股票: {len(valid_codes)} "
        f"(总上市: {total_listed}, 排除 {excluded}/上市<{f['min_list_days']}天)"
    )

    # ================================================================
    # Step 2: 批量拉取日线
    # ================================================================
    history_days = max(f.get("history_days", 30), 25)  # 至少 25 天确保 MA20 可算
    logger.info(f"[Scanner] Step 2/6: 拉取最近 {history_days} 个交易日全市场日线...")
    trade_dates = _get_recent_trade_dates(api, n=history_days)
    if not trade_dates:
        raise RuntimeError("未获取到交易日历")
    logger.info(f"[Scanner] 交易日范围: {trade_dates[-1]} ~ {trade_dates[0]}")

    df_daily = _fetch_all_daily(api, fetcher, trade_dates)
    df_daily = df_daily[df_daily["ts_code"].isin(valid_codes)].copy()
    df_daily = df_daily.sort_values(["ts_code", "trade_date"])

    logger.info(
        f"[Scanner] 日线数据: {len(df_daily)} 行, "
        f"{df_daily['ts_code'].nunique()} 只股票"
    )

    # ================================================================
    # Step 3: daily_basic（最新交易日）
    # ================================================================
    latest_date = trade_dates[0]
    logger.info(f"[Scanner] Step 3/6: 获取 {latest_date} daily_basic...")
    fetcher._check_rate_limit()
    df_basic = api.daily_basic(
        trade_date=latest_date,
        fields="ts_code,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv",
    )
    if df_basic is not None and not df_basic.empty:
        df_basic = df_basic[df_basic["ts_code"].isin(valid_codes)]
        logger.info(f"[Scanner] daily_basic: {len(df_basic)} 只股票")
    else:
        logger.warning("[Scanner] daily_basic 返回空，估值/换手过滤将被跳过")
        df_basic = pd.DataFrame()

    # ================================================================
    # Step 4: 本地计算指标
    # ================================================================
    logger.info("[Scanner] Step 4/6: 计算技术指标...")

    grouped = df_daily.groupby("ts_code")

    # 均线
    df_daily["ma5"] = grouped["close"].transform(lambda x: x.rolling(5).mean())
    df_daily["ma10"] = grouped["close"].transform(lambda x: x.rolling(10).mean())
    df_daily["ma20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())

    # 5 日均量
    df_daily["vol_ma5"] = grouped["vol"].transform(lambda x: x.rolling(5).mean())

    # 取最新交易日截面
    latest = df_daily[df_daily["trade_date"] == latest_date].copy()
    if latest.empty:
        raise RuntimeError(f"最新交易日 {latest_date} 无日线数据")

    # 5 日涨幅
    if len(trade_dates) >= 6:
        date_5d_ago = trade_dates[5]
        df_5d = df_daily[df_daily["trade_date"] == date_5d_ago][["ts_code", "close"]].rename(
            columns={"close": "close_5d_ago"}
        )
        latest = latest.merge(df_5d, on="ts_code", how="left")
        latest["change_5d"] = (
            (latest["close"] - latest["close_5d_ago"]) / latest["close_5d_ago"] * 100
        )
    else:
        latest["change_5d"] = 0.0

    # 乖离率 MA5
    latest["bias_ma5"] = (latest["close"] - latest["ma5"]) / latest["ma5"] * 100

    # 自算量比
    latest["vol_ratio_calc"] = latest["vol"] / latest["vol_ma5"]

    # 合并 daily_basic
    if not df_basic.empty:
        latest = latest.merge(df_basic, on="ts_code", how="left")

    logger.info(f"[Scanner] 最新截面: {len(latest)} 只股票，指标计算完成")

    # ================================================================
    # Step 5: 规则分层过滤
    # ================================================================
    logger.info("[Scanner] Step 5/6: 规则过滤...")
    cand = latest.copy()

    def _log_filter(label: str, before: int, after: int):
        logger.info(f"  [过滤] {label}: {before} -> {after} (-{before - after})")

    # --- 第一层：硬过滤 ---
    n0 = len(cand)
    cand = cand[cand["vol"] > 0]
    _log_filter("排除停牌/零成交", n0, len(cand))

    if "total_mv" in cand.columns:
        n0 = len(cand)
        min_mv = f["min_market_cap_yi"] * 10000  # 亿 → 万元（daily_basic 单位）
        cand = cand[cand["total_mv"] >= min_mv]
        _log_filter(f"总市值 >= {f['min_market_cap_yi']}亿", n0, len(cand))

    # --- 第二层：多头排列 ---
    n0 = len(cand)
    cand = cand[
        cand["ma5"].notna()
        & cand["ma10"].notna()
        & cand["ma20"].notna()
        & (cand["ma5"] > cand["ma10"])
        & (cand["ma10"] > cand["ma20"])
        & (cand["close"] > cand["ma5"])
    ]
    _log_filter("多头排列 (MA5>MA10>MA20) + 站上MA5", n0, len(cand))

    # --- 第三层：量能过滤 ---
    if "turnover_rate" in cand.columns:
        n0 = len(cand)
        cand = cand[
            (cand["turnover_rate"] >= f["turnover_min"])
            & (cand["turnover_rate"] <= f["turnover_max"])
        ]
        _log_filter(f"换手率 {f['turnover_min']}%~{f['turnover_max']}%", n0, len(cand))

    n0 = len(cand)
    vr = (
        cand["volume_ratio"].fillna(cand["vol_ratio_calc"])
        if "volume_ratio" in cand.columns
        else cand["vol_ratio_calc"]
    )
    cand = cand[vr >= f["volume_ratio_min"]]
    _log_filter(f"量比 >= {f['volume_ratio_min']}", n0, len(cand))

    # --- 第四层：动量过滤 ---
    n0 = len(cand)
    cand = cand[cand["change_5d"] < f["change_5d_max"]]
    _log_filter(f"5日涨幅 < {f['change_5d_max']}%", n0, len(cand))

    n0 = len(cand)
    cand = cand[cand["bias_ma5"].abs() < f["bias_ma5_max"]]
    _log_filter(f"|乖离率MA5| < {f['bias_ma5_max']}%", n0, len(cand))

    # ================================================================
    # Step 6: 打分排序
    # ================================================================
    logger.info(f"[Scanner] Step 6/6: 打分排序，从 {len(cand)} 只中取 Top {top_n}...")

    if cand.empty:
        logger.warning("[Scanner] 过滤后无候选股，保存空结果")
        result_codes: List[str] = []
        result_details: list = []
    else:
        # 重新取量比列（合并后可能列名不同）
        if "volume_ratio" in cand.columns:
            vr = cand["volume_ratio"].fillna(cand["vol_ratio_calc"])
        else:
            vr = cand["vol_ratio_calc"]

        cand = cand.copy()
        cand["score"] = 0.0

        # 多头强度：(MA5 - MA20) / MA20 * 100，越发散越好  (0~40 分)
        ma_spread = ((cand["ma5"] - cand["ma20"]) / cand["ma20"] * 100).clip(0, 10)
        cand["score"] += ma_spread * 4

        # 量能活跃度  (0~20 分)
        vr_norm = vr.clip(0.8, 3.0)
        cand["score"] += (vr_norm - 0.8) / 2.2 * 20

        # 低乖离奖励  (0~20 分)
        bias_abs = cand["bias_ma5"].abs()
        cand["score"] += (f["bias_ma5_max"] - bias_abs) / f["bias_ma5_max"] * 20

        # 5 日涨幅适中  (0~20 分)：靠近 5% 最高分
        chg_score = 20 - (cand["change_5d"] - 5).abs() * 2
        cand["score"] += chg_score.clip(0, 20)

        # 取 Top N
        cand = cand.sort_values("score", ascending=False).head(top_n)

        result_codes = [c.split(".")[0] for c in cand["ts_code"].tolist()]
        result_details = []
        for _, row in cand.iterrows():
            code = row["ts_code"].split(".")[0]
            result_details.append(
                {
                    "code": code,
                    "name": code_name_map.get(row["ts_code"], ""),
                    "score": round(float(row["score"]), 1),
                    "close": round(float(row["close"]), 2),
                    "change_5d": round(float(row.get("change_5d", 0)), 2),
                    "bias_ma5": round(float(row["bias_ma5"]), 2),
                    "turnover_rate": round(float(row.get("turnover_rate", 0)), 2),
                }
            )

    # ================================================================
    # 保存结果
    # ================================================================
    output = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": latest_date,
        "total_listed": total_listed,
        "valid_stocks": len(valid_codes),
        "after_filter": len(result_codes),
        "candidates": result_details,
        "codes": result_codes,
        "filters_used": f,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info(f"[Scanner] ✅ 扫描完成: {len(result_codes)} 只候选股, 耗时 {elapsed:.1f}s")
    logger.info(f"[Scanner] 结果已保存到: {output_path}")
    for d in result_details[:10]:
        logger.info(
            f"  {d['code']} {d['name']}: "
            f"评分={d['score']}, 5日涨幅={d['change_5d']}%, "
            f"乖离={d['bias_ma5']}%, 换手={d['turnover_rate']}%"
        )
    if len(result_details) > 10:
        logger.info(f"  ... 共 {len(result_details)} 只，仅显示前 10")

    return result_codes


def _find_latest_scan_file() -> Optional[str]:
    """在 data/ 下找到最新的 scan_candidates_YYYYMMDD.json 文件。"""
    import glob

    pattern = os.path.join(DEFAULT_OUTPUT_DIR, "scan_candidates_*.json")
    files = sorted(glob.glob(pattern), reverse=True)  # 按文件名降序 → 最新日期在前
    return files[0] if files else None


def load_scan_candidates(path: Optional[str] = None, latest: bool = False) -> List[str]:
    """读取扫描结果文件，返回候选股代码列表。

    Args:
        path: 指定文件路径。为 None 时按 latest 参数决定策略。
        latest: True 时查找 data/ 下最新的扫描文件；False 时查找当天文件。

    文件不存在返回空列表。
    """
    if path is None:
        if latest:
            path = _find_latest_scan_file()
            if path is None:
                logger.warning("[Scanner] data/ 下未找到任何扫描结果文件")
                return []
        else:
            path = _default_output_path()
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        codes = data.get("codes", [])
        scan_date = data.get("scan_date", "unknown")
        trade_date = data.get("trade_date", "unknown")
        logger.info(
            f"[Scanner] 读取扫描候选: {len(codes)} 只 "
            f"(扫描时间: {scan_date}, 交易日: {trade_date}, 文件: {path})"
        )
        return codes
    except Exception as e:
        logger.warning(f"[Scanner] 读取扫描结果失败: {e}")
        return []
