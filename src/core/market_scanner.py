# -*- coding: utf-8 -*-
"""
===================================
Market Scanner v2 — v27b 策略选股
===================================

基于 v27b 回测策略的全市场量化预筛选，6 层流水线：

  1. 基础过滤（ST/北交所/创业板/科创板/次新）
  2. 技术面 5 条件（MA多头 + 回踩确认 + 放量 + 动量区间 + ATR波动）
  3. 基本面硬门槛（ROE/毛利率/营收增速/PE/PB/市值/负债率）
  4. 行业热度（多周期动量聚合 + 百分位排名）
  5. 概念热度（v18c：东财概念板块涨幅百分位，与行业热度各占50%）
  6. 综合评分（60%技术面含热度 + 40%基本面）
  7. 排序取 Top N

数据需求：
  - Tushare 日线 ~80 天（~80 次 API，用于 MA60）
  - daily_basic 1 次（PE/PB/市值/换手率）
  - fina_indicator 按股票查询（仅技术面通过的候选股，~100-300 次）
  - cyq_perf 可选（筹码胜率加分，需 5000 积分）

预计运行耗时：5~15 分钟
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# v27b 策略参数
# ============================================================

# ── 技术面买入 ──────────────────────────────────
_MIN_MOM20 = 5.0               # 20日动量下限 %
_MAX_MOM20 = 40.0              # 20日动量上限 %
_MOM_PREFERRED = 25.0          # 5~25% 满分区，25~40% 衰减
_VOL_RATIO_MIN = 1.2           # 5日均量/20日均量 下限
_BREAKOUT_DAYS = 20            # 突破窗口：20日新高
_PULLBACK_WINDOW = 7           # 回踩确认窗口（天）
_ATR_THRESHOLD = 0.07          # ATR(14)/close < 7%

# ── 基本面门槛 ──────────────────────────────────
_MIN_ROE = 8.0                 # 扣非 ROE %
_MIN_GROSS_MARGIN = 20.0       # 毛利率 %
_MAX_DEBT_RATIO = 70.0         # 资产负债率 %（非金融）
_PE_RANGE = (0, 80)
_PB_RANGE = (0, 15)
_MIN_REVENUE_GROWTH = 5.0      # 营收同比增速 %
_MIN_MARKET_CAP_YI = 100       # 最小市值（亿元）

# 科技股放宽
_TECH_PE_RANGE = (0, 200)
_TECH_MIN_ROE = 3.0
_TECH_MIN_GROSS_MARGIN = 15.0
_TECH_MIN_MARKET_CAP_YI = 50

# ── 基本面评分权重 ──────────────────────────────
_FUND_W_ROE = 0.40
_FUND_W_REVENUE = 0.30
_FUND_W_MARGIN = 0.30
_FUND_BONUS_CASHFLOW = 0.10
_FUND_BONUS_PEG = 0.05

# ── 综合评分权重 ──────────────────────────────
_SCORE_W_TECH = 0.60
_SCORE_W_FUND = 0.40
_LOW_ELASTIC_TECH_W = 0.80     # 低弹性行业（白酒等）技术面权重
_LOW_ELASTIC_FUND_W = 0.20

# ── 行业分类 ──────────────────────────────────
_FINANCIAL_INDUSTRIES = {"银行", "保险", "证券", "多元金融", "银行I", "非银金融I"}

_TECH_INDUSTRIES = {
    "半导体", "元器件", "光学光电子", "电子制造", "其他电子",
    "计算机应用", "计算机设备", "IT服务", "软件开发",
    "通信设备", "通信服务", "通信运营",
    "电气设备", "电源设备", "电机", "电力设备",
    "专用设备", "通用设备", "仪器仪表",
    "航天装备", "航空装备", "地面兵装", "船舶制造",
    "医疗器械", "化学制药", "生物制品", "中药",
    "汽车配件", "汽车整车",
}

_LOW_ELASTIC_INDUSTRIES = {"白酒", "啤酒", "乳制品", "环境保护", "路桥", "园区开发"}

# ── 默认参数 ──────────────────────────────────
DEFAULT_OUTPUT_DIR = "data"

DEFAULT_FILTERS: Dict = {
    "min_list_days": 90,
    "exclude_bj": True,
    "exclude_cyb": True,
    "exclude_kcb": True,
    "exclude_st": True,
    "history_days": 80,           # MA60 需要 ≥60 天
    "top_n": 30,
    "enable_winner_rate": False,  # cyq_perf 需 5000 积分
}


# ============================================================
# Tushare 工具函数
# ============================================================

def _get_tushare_api():
    """获取已 patch 过端点的 Tushare Pro API 实例"""
    from data_provider.tushare_fetcher import TushareFetcher

    fetcher = TushareFetcher()
    if not fetcher.is_available():
        raise RuntimeError("Tushare API 不可用，请检查 TUSHARE_TOKEN 配置")
    return fetcher._api, fetcher


def _get_recent_trade_dates(api, n: int = 80) -> List[str]:
    """获取最近 N 个 A 股交易日（降序）"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - pd.Timedelta(days=max(n * 2, 180))).strftime("%Y%m%d")
    cal = api.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    dates = sorted(cal["cal_date"].tolist(), reverse=True)
    return dates[:n]


def _fetch_all_daily(api, fetcher, trade_dates: List[str]) -> pd.DataFrame:
    """批量拉取多个交易日的全市场日线"""
    frames: List[pd.DataFrame] = []
    total = len(trade_dates)
    for i, date in enumerate(trade_dates):
        try:
            fetcher._check_rate_limit()
            df = api.daily(trade_date=date)
            if df is not None and not df.empty:
                frames.append(df)
            count = len(df) if df is not None else 0
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"  [日线] {i + 1}/{total}，当日 {count} 只")
        except Exception as e:
            logger.warning(f"  [日线] {date} 失败: {e}")

    if not frames:
        raise RuntimeError("未获取到任何日线数据")
    return pd.concat(frames, ignore_index=True)


# ============================================================
# v27b 指标计算（向量化）
# ============================================================

def _calc_indicators(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算 v27b 全部技术指标。

    新增列：ma5/10/20/60, high20, recent_breakout, not_at_high,
           vol_ratio_5_20, mom5, mom20, atr14, atr_ratio
    """
    df = df_daily.sort_values(["ts_code", "trade_date"]).copy()

    # 强制数值类型（防止 Tushare 偶发返回字符串）
    for col in ["open", "high", "low", "close", "vol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grp = "ts_code"  # 每次 groupby 都用新实例，避免旧引用访问新列

    # ── 均线 ──
    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = df.groupby(grp)["close"].transform(
            lambda x, _n=n: x.rolling(_n, min_periods=_n).mean()
        )

    # ── 20日新高 ──
    df["high20"] = df.groupby(grp)["close"].transform(
        lambda x: x.rolling(_BREAKOUT_DAYS, min_periods=_BREAKOUT_DAYS).max()
    )

    # ── 回踩确认 ──
    # 昨日20日新高（shift 避免当日看到当日新高）
    high20_prev = df.groupby(grp)["high20"].shift(1)
    # 当日是否触及新高
    is_bk = (df["close"] >= high20_prev * 0.99).fillna(False).astype(int)
    # 过去 PULLBACK_WINDOW 天内是否有过突破
    df["recent_breakout"] = is_bk.groupby(df[grp]).transform(
        lambda x: x.rolling(_PULLBACK_WINDOW, min_periods=1).max()
    )
    # 当前不在 PULLBACK_WINDOW 日最高点 → 已回踩
    rolling_high_pw = df.groupby(grp)["close"].transform(
        lambda x: x.rolling(_PULLBACK_WINDOW, min_periods=1).max()
    )
    df["not_at_high"] = (df["close"] < rolling_high_pw).astype(int)

    # ── 量比（5日均量 / 20日均量）──
    df["vol_ma5"] = df.groupby(grp)["vol"].transform(
        lambda x: x.rolling(5, min_periods=5).mean()
    )
    df["vol_ma20"] = df.groupby(grp)["vol"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df["vol_ratio_5_20"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)

    # ── 动量 ──
    df["mom5"] = df.groupby(grp)["close"].transform(lambda x: (x / x.shift(5) - 1) * 100)
    df["mom20"] = df.groupby(grp)["close"].transform(lambda x: (x / x.shift(20) - 1) * 100)

    # ── ATR(14) ──
    prev_close = df.groupby(grp)["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    df["atr14"] = tr.groupby(df[grp]).transform(
        lambda x: x.rolling(14, min_periods=14).mean()
    )
    df["atr_ratio"] = df["atr14"] / df["close"]

    return df


# ============================================================
# v27b 买入信号 + 技术面评分
# ============================================================

def _filter_and_score_tech(latest: pd.DataFrame) -> pd.DataFrame:
    """
    对最新截面逐层过滤（v27b 5 条件），并计算 tech_score。
    返回通过技术面筛选的 DataFrame，含 tech_score 列。
    """
    df = latest.copy()
    n_start = len(df)

    # ── 条件 1：MA 多头（close > MA20 > MA60）──
    n0 = len(df)
    df = df[
        df["ma20"].notna() & df["ma60"].notna()
        & (df["close"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
    ]
    logger.info(f"  [v27b 条件1] MA多头 close>MA20>MA60: {n0} -> {len(df)}")

    # ── 条件 2：回踩确认（7天内突破过20日新高）──
    n0 = len(df)
    df = df[df["recent_breakout"].fillna(0) >= 1]
    logger.info(f"  [v27b 条件2] 回踩确认 (7天内破20日新高): {n0} -> {len(df)}")

    # ── 条件 3：放量（5日均量/20日均量 ≥ 1.2）──
    n0 = len(df)
    df = df[df["vol_ratio_5_20"].notna() & (df["vol_ratio_5_20"] >= _VOL_RATIO_MIN)]
    logger.info(f"  [v27b 条件3] 放量 (5d/20d量比 >= {_VOL_RATIO_MIN}): {n0} -> {len(df)}")

    # ── 条件 4：动量区间（20日动量 5%~40%）──
    n0 = len(df)
    df = df[
        df["mom20"].notna()
        & (df["mom20"] >= _MIN_MOM20)
        & (df["mom20"] <= _MAX_MOM20)
    ]
    logger.info(f"  [v27b 条件4] 动量区间 ({_MIN_MOM20}%~{_MAX_MOM20}%): {n0} -> {len(df)}")

    # ── 条件 5：ATR 波动过滤 ──
    n0 = len(df)
    df = df[df["atr_ratio"].notna() & (df["atr_ratio"] < _ATR_THRESHOLD)]
    logger.info(f"  [v27b 条件5] ATR波动 (<{_ATR_THRESHOLD * 100:.0f}%): {n0} -> {len(df)}")

    logger.info(f"  [v27b] 技术面通过: {n_start} -> {len(df)}")
    if df.empty:
        return df

    # ── 计算 tech_score ──
    df = df.copy()
    mom = df["mom20"]
    vr = df["vol_ratio_5_20"]

    # 动量得分（满分100）
    mom_score = pd.Series(0.0, index=df.index)
    lo = mom <= _MOM_PREFERRED
    mom_score.loc[lo] = (mom[lo] - _MIN_MOM20) / (_MOM_PREFERRED - _MIN_MOM20) * 100
    hi = mom > _MOM_PREFERRED
    mom_score.loc[hi] = (
        100 - (mom[hi] - _MOM_PREFERRED) / (_MAX_MOM20 - _MOM_PREFERRED) * 60
    ).clip(lower=0)

    # 放量得分（满分100，超2.5x衰减）
    vol_score = pd.Series(0.0, index=df.index)
    normal = vr <= 2.5
    vol_score.loc[normal] = (vr[normal] / _VOL_RATIO_MIN * 50).clip(upper=100)
    excess = vr > 2.5
    vol_score.loc[excess] = (100 - (vr[excess] - 2.5) / 2.5 * 80).clip(lower=0)

    # 回踩质量加分（不在7日最高点 +15）
    pullback_bonus = df["not_at_high"].fillna(0) * 15

    # 筹码胜率加分（T-1 winner_rate > 50% → +10）
    wr_bonus = pd.Series(0.0, index=df.index)
    if "winner_rate" in df.columns:
        wr_bonus = (df["winner_rate"].fillna(0) > 50).astype(float) * 10

    df["tech_score"] = mom_score * 0.5 + vol_score * 0.3 + pullback_bonus + wr_bonus

    return df


# ============================================================
# 基本面评分
# ============================================================

def _normalize(s: pd.Series, clip_low=None, clip_high=None) -> pd.Series:
    """归一化到 0~100"""
    s = s.copy()
    if clip_low is not None:
        s = s.clip(lower=clip_low)
    if clip_high is not None:
        s = s.clip(upper=clip_high)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100


def _fetch_fina_indicators(api, fetcher, ts_codes: List[str]) -> pd.DataFrame:
    """逐股获取最新一期财务指标"""
    all_data = []
    total = len(ts_codes)
    t0 = time.time()
    for i, code in enumerate(ts_codes):
        try:
            fetcher._check_rate_limit()
            df = api.fina_indicator(
                ts_code=code,
                fields="ts_code,ann_date,end_date,"
                       "roe_dt,grossprofit_margin,debt_to_assets,"
                       "or_yoy,netprofit_yoy,ocfps",
            )
            if df is not None and not df.empty:
                # 取最新一期
                all_data.append(df.sort_values("end_date").tail(1))
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            logger.info(f"  [基本面] {i + 1}/{total}  已用 {elapsed:.0f}s  剩余 ~{eta:.0f}s")

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def _score_fundamentals(
    fina_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    industry_map: Dict[str, str],
) -> Dict[str, dict]:
    """
    基本面门槛 + 评分。

    Returns:
        {ts_code: {"passed": bool, "fund_score": float,
                    "is_tech": bool, "industry": str}}
    """
    result: Dict[str, dict] = {}
    if fina_df.empty:
        return result

    fina = fina_df.copy()
    fina["industry"] = fina["ts_code"].map(industry_map).fillna("")
    fina["is_tech"] = fina["industry"].isin(_TECH_INDUSTRIES)

    # ── 门槛 ──
    std_ok = (
        (~fina["is_tech"])
        & (fina["roe_dt"].fillna(0) >= _MIN_ROE)
        & (fina["grossprofit_margin"].fillna(0) >= _MIN_GROSS_MARGIN)
        & (fina["or_yoy"].fillna(0) >= _MIN_REVENUE_GROWTH)
    )
    tech_ok = (
        (fina["is_tech"])
        & (fina["roe_dt"].fillna(0) >= _TECH_MIN_ROE)
        & (fina["grossprofit_margin"].fillna(0) >= _TECH_MIN_GROSS_MARGIN)
        & (fina["or_yoy"].fillna(0) >= _MIN_REVENUE_GROWTH)
    )
    pass_mask = std_ok | tech_ok

    # 负债率（非金融）
    non_fin = ~fina["industry"].isin(_FINANCIAL_INDUSTRIES)
    debt_ok = (fina["debt_to_assets"].fillna(0) < _MAX_DEBT_RATIO) | (~non_fin)
    pass_mask = pass_mask & debt_ok

    # ── 估值 ──
    val_codes: set = set()
    if val_df is not None and not val_df.empty:
        vdf = val_df.copy()
        vdf["is_tech"] = vdf["ts_code"].map(industry_map).fillna("").isin(_TECH_INDUSTRIES)

        std_v = vdf[~vdf["is_tech"]]
        std_pass = std_v[
            (std_v["pe_ttm"].fillna(0) > _PE_RANGE[0])
            & (std_v["pe_ttm"].fillna(999) < _PE_RANGE[1])
            & (std_v["pb"].fillna(0) > _PB_RANGE[0])
            & (std_v["pb"].fillna(999) < _PB_RANGE[1])
            & (std_v["total_mv"].fillna(0) > _MIN_MARKET_CAP_YI * 10000)
        ]
        tech_v = vdf[vdf["is_tech"]]
        tech_pass = tech_v[
            (tech_v["pe_ttm"].fillna(0) > _TECH_PE_RANGE[0])
            & (tech_v["pe_ttm"].fillna(999) < _TECH_PE_RANGE[1])
            & (tech_v["pb"].fillna(0) > _PB_RANGE[0])
            & (tech_v["pb"].fillna(999) < _PB_RANGE[1])
            & (tech_v["total_mv"].fillna(0) > _TECH_MIN_MARKET_CAP_YI * 10000)
        ]
        val_codes = set(std_pass["ts_code"]) | set(tech_pass["ts_code"])

    # ── 评分 ──
    passed_df = fina[pass_mask]
    if passed_df.empty:
        return result

    roe_s = _normalize(passed_df["roe_dt"].fillna(0), 0, 50)
    rev_s = _normalize(passed_df["or_yoy"].fillna(0), -10, 80)
    gm_s = _normalize(passed_df["grossprofit_margin"].fillna(0), 10, 80)
    scores = _FUND_W_ROE * roe_s + _FUND_W_REVENUE * rev_s + _FUND_W_MARGIN * gm_s

    ocf_s = _normalize(passed_df["ocfps"].fillna(0), -1, 10)
    np_s = _normalize(passed_df["netprofit_yoy"].fillna(0), -20, 100)
    scores = scores + _FUND_BONUS_CASHFLOW * ocf_s + _FUND_BONUS_PEG * np_s

    for idx, row in passed_df.iterrows():
        code = row["ts_code"]
        passed = True
        if val_codes and code not in val_codes:
            passed = False
        result[code] = {
            "passed": passed,
            "fund_score": float(scores.get(idx, 0)),
            "is_tech": bool(row["is_tech"]),
            "industry": row["industry"],
        }

    return result


# ============================================================
# 行业热度
# ============================================================

def _calc_industry_heat(
    latest_all: pd.DataFrame,
    industry_map: Dict[str, str],
) -> Dict[str, float]:
    """
    多周期行业热度 + 百分位排名。

    使用全市场最新截面的 mom5/mom20/vol_ratio_5_20 聚合。
    Returns {industry: heat_percentile (0~100)}
    """
    df = latest_all.copy()
    df["industry"] = df["ts_code"].map(industry_map).fillna("")
    df = df[df["industry"] != ""]

    if df.empty:
        return {}

    agg = df.groupby("industry").agg(
        m5=("mom5", "mean"),
        m20=("mom20", "mean"),
        vr=("vol_ratio_5_20", "mean"),
        cnt=("ts_code", "count"),
    )
    agg = agg[agg["cnt"] >= 3]
    if agg.empty:
        return {}

    # 原始热度 = 0.5*短期 + 0.5*中期
    agg["raw"] = 0.5 * agg["m5"].fillna(0) + 0.5 * agg["m20"].fillna(0)
    # 行业放量加成
    agg.loc[agg["vr"] > 1.2, "raw"] *= 1.1
    # 百分位
    agg["heat"] = agg["raw"].rank(pct=True) * 100

    return agg["heat"].to_dict()


# ============================================================
# 东财概念板块热度（v18c）
# ============================================================

_CONCEPT_MAP_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "concept_map.json")
_CONCEPT_MAP_MAX_AGE_DAYS = 20


def _load_or_fetch_concept_mapping(
    api, fetcher, trade_date: str,
) -> Dict[str, list]:
    """
    加载或获取东财概念板块成分映射：{stock_ts_code: [concept_codes]}。

    缓存到 data/concept_map.json，20 天内有效。
    """
    # 检查缓存
    if os.path.exists(_CONCEPT_MAP_FILE):
        try:
            with open(_CONCEPT_MAP_FILE, "r", encoding="utf-8") as fp:
                cached = json.load(fp)
            cached_date = cached.get("_date", "")
            if cached_date:
                days_since = (pd.Timestamp(trade_date) - pd.Timestamp(cached_date)).days
                if days_since <= _CONCEPT_MAP_MAX_AGE_DAYS:
                    mapping = {k: v for k, v in cached.items() if k != "_date"}
                    logger.info(
                        f"  [概念映射] 使用缓存（{cached_date}，{days_since}天前）："
                        f"{len(mapping)} 只股票"
                    )
                    return mapping
        except Exception:
            pass

    # 重新下载
    logger.info("  [概念映射] 缓存过期或不存在，重新下载...")
    try:
        fetcher._check_rate_limit()
        idx = api.dc_index(trade_date=trade_date, fields="ts_code,name")
        if idx is None or idx.empty:
            logger.warning("  [概念映射] dc_index 返回空")
            return {}
    except Exception as e:
        logger.warning(f"  [概念映射] dc_index 失败: {e}")
        return {}

    concept_codes = idx["ts_code"].unique().tolist()
    logger.info(f"  [概念映射] 共 {len(concept_codes)} 个概念板块，开始下载成分...")

    stock_to_concepts: Dict[str, list] = {}
    for i, ccode in enumerate(concept_codes):
        try:
            fetcher._check_rate_limit()
            df = api.dc_member(
                trade_date=trade_date, ts_code=ccode,
                fields="ts_code,con_code",
            )
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    con = row["con_code"]
                    if con not in stock_to_concepts:
                        stock_to_concepts[con] = []
                    stock_to_concepts[con].append(ccode)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            logger.info(f"    [概念映射] 进度 {i + 1}/{len(concept_codes)}")

    logger.info(f"  [概念映射] 完成：{len(stock_to_concepts)} 只股票有概念归属")

    # 保存缓存
    try:
        cache_data = dict(stock_to_concepts)
        cache_data["_date"] = trade_date
        os.makedirs(os.path.dirname(_CONCEPT_MAP_FILE) or ".", exist_ok=True)
        with open(_CONCEPT_MAP_FILE, "w", encoding="utf-8") as fp:
            json.dump(cache_data, fp, ensure_ascii=False)
        logger.info(f"  [概念映射] 已缓存到 {_CONCEPT_MAP_FILE}")
    except Exception as e:
        logger.warning(f"  [概念映射] 缓存保存失败: {e}")

    return stock_to_concepts


def _calc_concept_heat(
    api, fetcher, trade_date: str,
    stock_concepts: Dict[str, list],
) -> Dict[str, float]:
    """
    计算每只股票的概念热度百分位。

    取每只股票所属概念中涨幅排名最高的百分位作为该股票的概念热度。
    Returns {ts_code: concept_heat_percentile (0~100)}
    """
    if not stock_concepts:
        return {}

    try:
        fetcher._check_rate_limit()
        dc = api.dc_index(trade_date=trade_date, fields="ts_code,pct_change")
        if dc is None or dc.empty:
            return {}
    except Exception as e:
        logger.warning(f"  [概念热度] dc_index 行情获取失败: {e}")
        return {}

    dc["pct_change"] = pd.to_numeric(dc["pct_change"], errors="coerce")
    dc["pct_rank"] = dc["pct_change"].rank(pct=True) * 100
    concept_rank = dict(zip(dc["ts_code"], dc["pct_rank"]))

    result: Dict[str, float] = {}
    for stock, concepts in stock_concepts.items():
        ranks = [concept_rank[c] for c in concepts if c in concept_rank]
        if ranks:
            result[stock] = max(ranks)

    return result


# ============================================================
# 筹码胜率（可选）
# ============================================================

def _fetch_winner_rate(
    api, fetcher, ts_codes: List[str], trade_date: str,
) -> Dict[str, float]:
    """
    获取指定交易日的 winner_rate（获利比例）。
    Returns {ts_code: winner_rate}
    """
    result: Dict[str, float] = {}
    total = len(ts_codes)
    t0 = time.time()
    for i, code in enumerate(ts_codes):
        try:
            fetcher._check_rate_limit()
            df = api.cyq_perf(
                ts_code=code,
                start_date=trade_date,
                end_date=trade_date,
                fields="ts_code,trade_date,winner_rate",
            )
            if df is not None and not df.empty:
                wr = pd.to_numeric(df.iloc[0]["winner_rate"], errors="coerce")
                if pd.notna(wr):
                    result[code] = float(wr)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [筹码] {i + 1}/{total}  已用 {elapsed:.0f}s")

    logger.info(f"  [筹码] 获取 winner_rate: {len(result)}/{total} 只有数据")
    return result


# ============================================================
# 公开接口
# ============================================================

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
    v27b 策略全市场扫描，返回候选股代码列表。

    流水线：
      Step 1  股票列表（排除 ST/北交所/创业板/科创板/次新）
      Step 2  拉取 ~80 天全市场日线
      Step 3  daily_basic（PE/PB/市值/换手）
      Step 4  计算 v27b 技术指标
      Step 5  技术面 5 条件过滤 + tech_score
      Step 6  基本面数据获取 + 门槛过滤 + fund_score
      Step 7  行业热度 + 综合评分
      Step 8  排序取 Top N → 保存 JSON
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
    logger.info("[Scanner] Step 1/8: 获取全市场股票列表...")
    fetcher._check_rate_limit()
    stock_list = api.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name,list_date,market,industry",
    )
    total_listed = len(stock_list)

    if f.get("exclude_st", True):
        stock_list = stock_list[~stock_list["name"].str.contains("ST|退", na=False)]
    if f.get("exclude_bj", True):
        stock_list = stock_list[~stock_list["ts_code"].str.endswith(".BJ")]
    if f.get("exclude_cyb", True):
        stock_list = stock_list[stock_list["market"] != "创业板"]
    if f.get("exclude_kcb", True):
        stock_list = stock_list[stock_list["market"] != "科创板"]

    stock_list["list_date_dt"] = pd.to_datetime(stock_list["list_date"], format="%Y%m%d")
    stock_list = stock_list[
        (pd.Timestamp.now() - stock_list["list_date_dt"]).dt.days >= f["min_list_days"]
    ]

    valid_codes = set(stock_list["ts_code"].tolist())
    code_name_map = dict(zip(stock_list["ts_code"], stock_list["name"]))
    industry_map = dict(zip(stock_list["ts_code"], stock_list["industry"].fillna("")))

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
    # Step 2: 批量拉取日线（~80天，满足 MA60）
    # ================================================================
    history_days = max(f.get("history_days", 80), 65)
    logger.info(f"[Scanner] Step 2/8: 拉取最近 {history_days} 个交易日全市场日线...")
    trade_dates = _get_recent_trade_dates(api, n=history_days)
    if not trade_dates:
        raise RuntimeError("未获取到交易日历")
    logger.info(f"[Scanner] 交易日范围: {trade_dates[-1]} ~ {trade_dates[0]}")

    df_daily = _fetch_all_daily(api, fetcher, trade_dates)
    df_daily = df_daily[df_daily["ts_code"].isin(valid_codes)].copy()
    df_daily = df_daily.sort_values(["ts_code", "trade_date"])

    logger.info(
        f"[Scanner] 日线数据: {len(df_daily):,} 行, "
        f"{df_daily['ts_code'].nunique()} 只股票"
    )

    # ================================================================
    # Step 3: daily_basic
    # ================================================================
    latest_date = trade_dates[0]
    logger.info(f"[Scanner] Step 3/8: 获取 {latest_date} daily_basic...")
    fetcher._check_rate_limit()
    df_basic = api.daily_basic(
        trade_date=latest_date,
        fields="ts_code,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv",
    )
    if df_basic is not None and not df_basic.empty:
        df_basic = df_basic[df_basic["ts_code"].isin(valid_codes)]
        logger.info(f"[Scanner] daily_basic: {len(df_basic)} 只股票")
    else:
        logger.warning("[Scanner] daily_basic 返回空")
        df_basic = pd.DataFrame()

    # ================================================================
    # Step 4: v27b 技术指标
    # ================================================================
    logger.info("[Scanner] Step 4/8: 计算 v27b 技术指标（MA/动量/ATR/回踩确认）...")
    df_daily = _calc_indicators(df_daily)

    # 取最新截面
    latest = df_daily[df_daily["trade_date"] == latest_date].copy()
    if latest.empty:
        raise RuntimeError(f"最新交易日 {latest_date} 无日线数据")

    # 合并 daily_basic（PE/PB/市值/换手）
    if not df_basic.empty:
        latest = latest.merge(df_basic, on="ts_code", how="left", suffixes=("", "_basic"))

    logger.info(f"[Scanner] 最新截面: {len(latest)} 只股票")

    # ================================================================
    # Step 5: 技术面 5 条件过滤 + tech_score
    # ================================================================
    logger.info(f"[Scanner] Step 5/8: v27b 技术面过滤...")
    # 先排除停牌/零成交
    latest = latest[latest["vol"] > 0]

    candidates = _filter_and_score_tech(latest)

    if candidates.empty:
        logger.warning("[Scanner] 技术面过滤后无候选股，保存空结果")
        _save_empty_result(output_path, latest_date, total_listed, len(valid_codes), f)
        return []

    tech_codes = candidates["ts_code"].tolist()
    logger.info(f"[Scanner] 技术面通过: {len(tech_codes)} 只")

    # ================================================================
    # Step 6: 基本面数据 + 门槛过滤 + fund_score
    # ================================================================
    logger.info(f"[Scanner] Step 6/8: 获取 {len(tech_codes)} 只候选股基本面...")
    fina_df = _fetch_fina_indicators(api, fetcher, tech_codes)
    logger.info(f"[Scanner] 获取到 {len(fina_df)} 条财务数据")

    # 基本面评分（含门槛）
    val_df = df_basic[df_basic["ts_code"].isin(set(tech_codes))] if not df_basic.empty else None
    fund_scores = _score_fundamentals(fina_df, val_df, industry_map)

    # 基本面通过的代码集
    fund_passed = {c for c, s in fund_scores.items() if s["passed"]}
    n_before_fund = len(candidates)
    candidates = candidates[candidates["ts_code"].isin(fund_passed)].copy()
    logger.info(
        f"[Scanner] 基本面门槛: {n_before_fund} -> {len(candidates)} "
        f"(ROE/毛利率/营收/PE/PB/市值)"
    )

    if candidates.empty:
        logger.warning("[Scanner] 基本面过滤后无候选股")
        _save_empty_result(output_path, latest_date, total_listed, len(valid_codes), f)
        return []

    # ================================================================
    # Step 6.5（可选）: 筹码胜率
    # ================================================================
    if f.get("enable_winner_rate", False) and len(trade_dates) >= 2:
        prev_date = trade_dates[1]
        logger.info(f"[Scanner] Step 6.5: 获取 {prev_date} 筹码胜率 (cyq_perf)...")
        wr_map = _fetch_winner_rate(api, fetcher, candidates["ts_code"].tolist(), prev_date)
        if wr_map:
            candidates["winner_rate"] = candidates["ts_code"].map(wr_map)
            # 重算 tech_score 加上 winner_rate bonus
            wr_bonus = (candidates["winner_rate"].fillna(0) > 50).astype(float) * 10
            candidates["tech_score"] = candidates["tech_score"] + wr_bonus

    # ================================================================
    # Step 7: 行业热度 + 综合评分
    # ================================================================
    logger.info("[Scanner] Step 7/8: 行业热度 + 概念热度 + 综合评分...")

    # 行业热度（全市场最新截面，不只是候选股）
    industry_heat = _calc_industry_heat(
        df_daily[df_daily["trade_date"] == latest_date],
        industry_map,
    )
    n_ind = len(industry_heat)
    logger.info(f"  [行业热度] 有效行业 {n_ind} 个")

    # 概念热度（v18c：东财概念板块）
    stock_concepts = _load_or_fetch_concept_mapping(api, fetcher, latest_date)
    concept_heat = _calc_concept_heat(api, fetcher, latest_date, stock_concepts)
    logger.info(f"  [概念热度] {len(concept_heat)} 只股票有概念热度")

    # 综合评分
    candidates["industry"] = candidates["ts_code"].map(industry_map).fillna("")
    ind_heat_s = candidates["industry"].map(industry_heat).fillna(50.0)
    # v18c：融合概念热度（50%行业 + 50%概念）
    if concept_heat:
        cpt_heat_s = candidates["ts_code"].map(concept_heat).fillna(50.0)
        candidates["heat"] = ind_heat_s * 0.5 + cpt_heat_s * 0.5
    else:
        candidates["heat"] = ind_heat_s
    fund_score_map = {c: s["fund_score"] for c, s in fund_scores.items() if s.get("passed")}
    candidates["fund_score"] = candidates["ts_code"].map(fund_score_map).fillna(0.0)

    # tech_with_heat = tech_score * 0.7 + heat * 0.3
    candidates["tech_with_heat"] = candidates["tech_score"] * 0.7 + candidates["heat"] * 0.3

    # final_score（低弹性行业差异化权重）
    is_low = candidates["industry"].isin(_LOW_ELASTIC_INDUSTRIES)
    candidates["final_score"] = np.where(
        is_low,
        _LOW_ELASTIC_TECH_W * candidates["tech_with_heat"]
        + _LOW_ELASTIC_FUND_W * candidates["fund_score"],
        _SCORE_W_TECH * candidates["tech_with_heat"]
        + _SCORE_W_FUND * candidates["fund_score"],
    )

    # ================================================================
    # Step 8: 排序取 Top N + 保存
    # ================================================================
    logger.info(f"[Scanner] Step 8/8: 排序取 Top {top_n}...")
    candidates = candidates.sort_values("final_score", ascending=False).head(top_n)

    result_codes = [c.split(".")[0] for c in candidates["ts_code"].tolist()]
    result_details = []
    for _, row in candidates.iterrows():
        code_6 = row["ts_code"].split(".")[0]
        fs = fund_scores.get(row["ts_code"], {})

        detail: dict = {
            "code": code_6,
            "name": code_name_map.get(row["ts_code"], ""),
            "final_score": round(float(row["final_score"]), 1),
            "tech_score": round(float(row["tech_score"]), 1),
            "fund_score": round(float(fs.get("fund_score", 0)), 1),
            "close": round(float(row["close"]), 2),
            "mom20": round(float(row.get("mom20", 0)), 1),
            "vol_ratio": round(float(row.get("vol_ratio_5_20", 0)), 2),
            "atr_pct": round(float(row.get("atr_ratio", 0)) * 100, 1),
            "industry": industry_map.get(row["ts_code"], ""),
            "heat": round(float(row.get("heat", 50)), 1),
        }
        if "pe_ttm" in row.index and pd.notna(row.get("pe_ttm")):
            detail["pe_ttm"] = round(float(row["pe_ttm"]), 1)
        if "pb" in row.index and pd.notna(row.get("pb")):
            detail["pb"] = round(float(row["pb"]), 2)
        if "turnover_rate" in row.index and pd.notna(row.get("turnover_rate")):
            detail["turnover_rate"] = round(float(row["turnover_rate"]), 2)
        if "winner_rate" in row.index and pd.notna(row.get("winner_rate")):
            detail["winner_rate"] = round(float(row["winner_rate"]), 1)
        # 概念热度（v18c）
        cpt_h = concept_heat.get(row["ts_code"])
        if cpt_h is not None:
            detail["concept_heat"] = round(float(cpt_h), 1)

        # 构造选股理由
        parts = [
            f"动量{row.get('mom20', 0):.1f}%",
            f"量比{row.get('vol_ratio_5_20', 0):.1f}x",
            f"ATR{row.get('atr_ratio', 0) * 100:.1f}%",
        ]
        if row.get("not_at_high", 0):
            parts.append("已回踩")
        if "winner_rate" in row.index and pd.notna(row.get("winner_rate")):
            parts.append(f"胜率{row['winner_rate']:.0f}%")
        detail["reason"] = "|".join(parts)

        result_details.append(detail)

    # 保存
    output = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": latest_date,
        "strategy": "v27b",
        "total_listed": total_listed,
        "valid_stocks": len(valid_codes),
        "after_tech_filter": len(tech_codes),
        "after_fund_filter": len(fund_passed),
        "after_filter": len(result_codes),
        "candidates": result_details,
        "codes": result_codes,
        "filters_used": {k: v for k, v in f.items() if not callable(v)},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info(f"[Scanner] ✅ v27b 扫描完成: {len(result_codes)} 只候选股, 耗时 {elapsed:.1f}s")
    logger.info(f"[Scanner] 结果已保存到: {output_path}")
    for d in result_details[:10]:
        logger.info(
            f"  {d['code']} {d['name']}: "
            f"综合={d['final_score']}  技术={d['tech_score']}  "
            f"基本面={d['fund_score']}  热度={d['heat']}  "
            f"动量={d['mom20']}%  {d['reason']}"
        )
    if len(result_details) > 10:
        logger.info(f"  ... 共 {len(result_details)} 只，仅显示前 10")

    return result_codes


def _save_empty_result(output_path: str, trade_date: str, total: int, valid: int, f: dict):
    """保存空结果文件"""
    output = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": trade_date,
        "strategy": "v27b",
        "total_listed": total,
        "valid_stocks": valid,
        "after_tech_filter": 0,
        "after_fund_filter": 0,
        "after_filter": 0,
        "candidates": [],
        "codes": [],
        "filters_used": {k: v for k, v in f.items() if not callable(v)},
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)


def _find_latest_scan_file() -> Optional[str]:
    """在 data/ 下找到最新的 scan_candidates_YYYYMMDD.json 文件。"""
    import glob

    pattern = os.path.join(DEFAULT_OUTPUT_DIR, "scan_candidates_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def load_scan_candidates(path: Optional[str] = None, latest: bool = False) -> List[str]:
    """读取扫描结果文件，返回候选股代码列表。

    Args:
        path: 指定文件路径。为 None 时按 latest 参数决定策略。
        latest: True 时查找 data/ 下最新的扫描文件；False 时查找当天文件。
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
        strategy = data.get("strategy", "unknown")
        logger.info(
            f"[Scanner] 读取扫描候选: {len(codes)} 只 "
            f"(策略: {strategy}, 扫描时间: {scan_date}, "
            f"交易日: {trade_date}, 文件: {path})"
        )
        return codes
    except Exception as e:
        logger.warning(f"[Scanner] 读取扫描结果失败: {e}")
        return []
