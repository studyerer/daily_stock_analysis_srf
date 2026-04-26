"""
完整交易系统 v18c（v18a基线 + 筹码胜率评分加成）
==================================================================================

基于v27b（放量封顶），新增单一改动：
  候选股如果筹码胜率（winner_rate）> 50%，tech_score额外加10分。
  
  筹码胜率 = 当前价格以上获利筹码占比。winner_rate > 50%意味着超过一半的
  持有者是盈利的——上方抛压小，回踩后的支撑更强。
  
  这与mom20不同：mom20衡量的是20天价格变化率（趋势方向），winner_rate衡量的
  是全部持有者的盈亏分布（筹码结构）。一只股票可能mom20=15%但winner_rate只有
  30%（从深坑中反弹，大量持有者仍被套），这类"解套反弹"比"干净的趋势上涨"
  （高mom20+高winner_rate）更容易遭遇抛压。
  
  数据来源：Tushare cyq_perf（5000积分，2018年起，盘后18-19点更新）。
  shift(1)避免前瞻偏差：14:30执行时只能看到T-1的胜率。

依赖：pip install pytdx tushare pandas numpy --break-system-packages
"""

import os, sys, time, argparse, pickle, gc
import numpy as np
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

TUSHARE_TOKEN = "948c885edff1169ef76e26ee6a540cebdc73c972b2595f9867825cf9"
TOTAL_CAPITAL = 100_000

MAX_POSITIONS    = 5
STOP_LOSS_PCT    = 0.12
HALT_SELL_DAYS   = 3

# ── 牛熊判断缓冲 ──────────────────────────
REGIME_CONFIRM_DAYS = 5          # 切换regime需连续5天确认，进一步过滤噪声

# ── Step1: 卖出参数 ──────────────────────────
TRAIL_STOP_PCT       = 0.10     # 从最高点回撤10%止盈
PARTIAL_PROFIT_1     = 0.20     # 收益>20%时卖出50%
PARTIAL_PROFIT_2     = 0.40     # 收益>40%时卖出剩余50%
TRAIL_BELOW_MA10_DAYS = 2       # 跌破MA10连续2天止盈（趋势止盈）
TRAIL_BELOW_MA20_DAYS = 3       # 跌破MA20连续3天止损

# BEAR/NEUTRAL 收紧止损（不强制清仓，但保护更紧）
BEAR_STOP_LOSS_PCT       = 0.05   # 熊市止损5%（vs 常规12%）
BEAR_TRAIL_STOP_PCT      = 0.05   # 熊市回撤止盈5%（vs 常规10%）
BEAR_TRAIL_MA_DAYS       = 1      # 熊市跌破MA5即卖（vs 常规MA10连续2天）

# ── Step2: 买入参数 ──────────────────────────
MIN_MOM20        = 5.0
MAX_MOM20        = 40.0
MOM_PREFERRED    = 25.0          # 5~25%全权，25~40%降权
VOLUME_RATIO     = 1.2
BREAKOUT_DAYS    = 20
PULLBACK_WINDOW  = 7             # v5k改动：回踩确认窗口从5天放宽到7天（覆盖A股牛市中80%的短期回调周期）
ATR_THRESHOLD    = 0.07          # ATR/close < 7%（放宽，允许成长股进入）

# ── Step3: 基本面参数 ──────────────────────────
MIN_ROE = 8.0
MIN_GROSS_MARGIN = 20.0
MAX_DEBT_RATIO = 70.0
PE_RANGE = (0, 80)
PB_RANGE = (0, 15)
MIN_REVENUE_GROWTH = 5.0
MIN_MARKET_CAP = 100

# 科技行业放宽参数
TECH_PE_RANGE = (0, 200)
TECH_MIN_ROE = 3.0
TECH_MIN_GROSS_MARGIN = 15.0
TECH_MIN_MARKET_CAP = 50

# 基本面评分权重（严格按指南：0.4 ROE + 0.3 营收 + 0.3 毛利率）
FUND_WEIGHT_ROE      = 0.40
FUND_WEIGHT_REVENUE  = 0.30
FUND_WEIGHT_MARGIN   = 0.30
# 可选增强因子作为额外加成（不稀释主权重）
FUND_BONUS_CASHFLOW  = 0.10   # 经营现金流加成上限10分
FUND_BONUS_PEG       = 0.05   # 净利润增速加成上限5分

# 最终评分权重（严格按指南：0.6 技术面 + 0.4 基本面）
# 行业热度并入技术面评分中（作为技术面的子因子）
SCORE_WEIGHT_TECH    = 0.60
SCORE_WEIGHT_FUND    = 0.40

# v20a：低弹性稳健行业——基本面天然高但趋势延续力弱，降低基本面权重
LOW_ELASTIC_INDUSTRIES = {"白酒", "啤酒", "乳制品", "环境保护", "路桥", "园区开发"}
LOW_ELASTIC_TECH_W   = 0.80   # 低弹性行业：技术面80%
LOW_ELASTIC_FUND_W   = 0.20   # 低弹性行业：基本面20%

# ── Step5: 仓位管理参数 ──────────────────────────
REGIME_MAX_POSITION = {
    "BULL":    1.00,     # 牛市满仓
    "NEUTRAL": 0.10,     # 中性1成仓（只持有存量，不新开仓）
    "BEAR":    0.00,     # 熊市彻底空仓（不新开仓）
}

# ── 交易成本 ──────────────────────────
STOCK_COMMISSION = 0.0003
STAMP_DUTY       = 0.0005
SLIPPAGE         = 0.001

DEFAULT_START = "2018-01-01"
DEFAULT_END   = "2026-03-30"
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "backtest_cache"

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ── 行业分类 ──────────────────────────
FINANCIAL_INDUSTRIES = {"银行", "保险", "证券", "多元金融", "银行I", "非银金融I"}

TECH_INDUSTRIES = {
    "半导体", "元器件", "光学光电子", "电子制造", "其他电子",
    "计算机应用", "计算机设备", "IT服务", "软件开发",
    "通信设备", "通信服务", "通信运营",
    "电气设备", "电源设备", "电机", "电力设备",
    "专用设备", "通用设备", "仪器仪表",
    "航天装备", "航空装备", "地面兵装", "船舶制造",
    "医疗器械", "化学制药", "生物制品", "中药",
    "汽车配件", "汽车整车",
}

# ── Step6: 默认红利种子池（动态筛选的兜底）──────
DIVIDEND_SEED_POOL = {
    "600519.SH": "贵州茅台", "000858.SZ": "五粮液",
    "600900.SH": "长江电力", "601985.SH": "中国核电",
    "601088.SH": "中国神华", "600188.SH": "兖矿能源",
    "600941.SH": "中国移动", "601728.SH": "中国电信",
    "600036.SH": "招商银行", "601398.SH": "工商银行",
    "601939.SH": "建设银行", "601288.SH": "农业银行",
    "601318.SH": "中国平安",
    "000333.SZ": "美的集团", "000651.SZ": "格力电器",
    "600690.SH": "海尔智家", "600887.SH": "伊利股份",
    "600585.SH": "海螺水泥", "600377.SH": "宁沪高速",
    "600795.SH": "国电电力", "600011.SH": "华能国际",
}


# ============================================================
# pytdx 30分钟K线下载 → 构造14:30伪日线
# ============================================================

# 通达信公共行情服务器列表（自动切换）
TDX_SERVERS = [
    ("119.147.212.81", 7709),
    ("114.80.63.12", 7709),
    ("218.75.126.9", 7709),
    ("124.74.236.94", 7721),
    ("221.231.141.60", 7709),
    ("59.173.18.140", 7709),
    ("112.74.214.43", 7727),
]

def _ts_code_to_tdx(ts_code):
    """600519.SH → (1, '600519')  000001.SZ → (0, '000001')"""
    code, market = ts_code.split(".")
    return (1 if market == "SH" else 0), code


def download_30min_pytdx(ts_codes, min_date="20170101"):
    """
    用pytdx下载30分钟K线数据。
    pytdx直连通达信服务器，TCP协议，每次返回800条，无频率限制。
    
    返回DataFrame: ts_code, trade_date, time_str, open, high, low, close, vol
    """
    from pytdx.hq import TdxHq_API

    print(f"[pytdx] 下载30分钟K线：{len(ts_codes)} 只")

    api = TdxHq_API()
    connected = False
    for host, port in TDX_SERVERS:
        try:
            api.connect(host, port)
            connected = True
            print(f"  已连接：{host}:{port}")
            break
        except Exception:
            continue

    if not connected:
        print("[pytdx] 无法连接任何通达信服务器")
        return pd.DataFrame()

    all_data = []
    total = len(ts_codes)
    t0 = time.time()

    for i, ts_code in enumerate(ts_codes):
        market, code = _ts_code_to_tdx(ts_code)
        stock_bars = []

        # pytdx每次最多800条，需循环获取全部历史
        offset = 0
        while True:
            try:
                bars = api.get_security_bars(2, market, code, offset, 800)  # 2=30min
            except Exception:
                break
            if not bars or len(bars) == 0:
                break
            stock_bars.extend(bars)
            offset += 800
            if len(bars) < 800:
                break  # 没有更多数据了

        if stock_bars:
            df = pd.DataFrame(stock_bars)
            # pytdx返回的字段：datetime, open, close, high, low, vol, amount, year, month, day, hour, minute
            df["ts_code"] = ts_code
            df["trade_date"] = df["datetime"].str[:10].str.replace("-", "")
            df["hour"] = df["datetime"].str[11:13].astype(int)
            df["minute"] = df["datetime"].str[14:16].astype(int)
            # 过滤：只保留min_date之后的数据
            df = df[df["trade_date"] >= min_date]
            if not df.empty:
                all_data.append(df[["ts_code", "trade_date", "hour", "minute",
                                    "open", "high", "low", "close", "vol"]])

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed / 60 if speed > 0 else 0
            print(f"    进度：{i+1}/{total}  "
                  f"已用{elapsed/60:.1f}分钟  预计剩余{eta:.1f}分钟")

    api.disconnect()

    if not all_data:
        print("[pytdx] 未获取到数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    for col in ["open", "high", "low", "close", "vol"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # 检查数据范围
    date_range = f"{result['trade_date'].min()} ~ {result['trade_date'].max()}"
    print(f"[pytdx] 完成：{len(result):,} 条，{result['ts_code'].nunique()} 只，范围 {date_range}")
    return result


def build_1430_daily_from_pytdx(min30_data):
    """
    从pytdx 30分钟K线构造14:30伪日线（纯pandas向量化，无Python循环）。
    
    pytdx 30分钟K线时间戳为bar起始时间：
      9:30, 10:00, 10:30, 11:00, 13:00, 13:30, 14:30, 14:30
    其中14:30那根bar覆盖14:30-14:30，其close即为14:30价格。
    """
    if min30_data.empty:
        return pd.DataFrame()

    print(f"  构造14:30伪日线（{len(min30_data):,} 条30分钟数据）...")
    data = min30_data.copy()

    # 过滤：只保留14:30之前的bar（hour<14 或 hour==14 & minute==0）
    mask = (data["hour"] < 14) | ((data["hour"] == 14) & (data["minute"] == 30))
    data = data[mask].copy()
    if data.empty:
        return pd.DataFrame()

    grp = data.groupby(["ts_code", "trade_date"])

    # 向量化聚合：high/low/vol
    agg_df = grp.agg(
        high=("high", "max"),
        low=("low", "min"),
        vol=("vol", "sum"),
    ).reset_index()

    # open = 每天第一根bar的open（按时间排序取第一条）
    data["time_key"] = data["hour"] * 100 + data["minute"]
    idx_first = data.groupby(["ts_code", "trade_date"])["time_key"].idxmin()
    open_df = data.loc[idx_first, ["ts_code", "trade_date", "open"]].reset_index(drop=True)

    # close = 14:30 bar的close（即14:30价格）；若无14:30 bar则取最后一根
    bar_1400 = data[(data["hour"] == 14) & (data["minute"] == 30)][["ts_code", "trade_date", "close"]]
    # 对于没有14:30 bar的交易日（半天市等），取当天最后一根bar的close
    idx_last = data.groupby(["ts_code", "trade_date"])["time_key"].idxmax()
    close_fallback = data.loc[idx_last, ["ts_code", "trade_date", "close"]].reset_index(drop=True)
    # 优先用14:30 bar，缺失的用fallback
    close_df = close_fallback.merge(
        bar_1400, on=["ts_code", "trade_date"], how="left", suffixes=("_fb", "_1400")
    )
    close_df["close"] = close_df["close_1400"].fillna(close_df["close_fb"])
    close_df = close_df[["ts_code", "trade_date", "close"]]

    # 合并所有字段
    result = agg_df.merge(open_df, on=["ts_code", "trade_date"])
    result = result.merge(close_df, on=["ts_code", "trade_date"])
    result["amount"] = 0

    result = result[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]]
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[14:30伪日线] {len(result):,} 条，{result['ts_code'].nunique()} 只，"
          f"范围 {result['trade_date'].min()} ~ {result['trade_date'].max()}")
    return result


def download_hs300_1430_pytdx(min_date="20170101"):
    """下载HS300 ETF的30分钟K线并提取14:30价格"""
    from pytdx.hq import TdxHq_API

    api = TdxHq_API()
    for host, port in TDX_SERVERS:
        try:
            api.connect(host, port)
            break
        except Exception:
            continue

    all_bars = []
    offset = 0
    while True:
        try:
            bars = api.get_security_bars(2, 1, "510300", offset, 800)  # SH market=1
        except Exception:
            break
        if not bars or len(bars) == 0:
            break
        all_bars.extend(bars)
        offset += 800
        if len(bars) < 800:
            break

    api.disconnect()

    if not all_bars:
        print("[pytdx] HS300 ETF数据为空")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["trade_date"] = df["datetime"].str[:10].str.replace("-", "")
    df["hour"] = df["datetime"].str[11:13].astype(int)
    df["minute"] = df["datetime"].str[14:16].astype(int)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["trade_date"] >= min_date]

    # 取14:30 bar的close作为14:30价格
    bar_1400 = df[(df["hour"] == 14) & (df["minute"] == 30)].copy()
    if bar_1400.empty:
        print("[pytdx] HS300 ETF无14:30 bar数据")
        return pd.DataFrame()

    result = pd.DataFrame({
        "ts_code": "510300.SH",
        "trade_date": bar_1400["trade_date"].values,
        "close": bar_1400["close"].values,
    }).reset_index(drop=True)

    print(f"[14:30 HS300] {len(result)} 条，范围 {result['trade_date'].min()} ~ {result['trade_date'].max()}")
    return result


# ============================================================
# Tushare数据下载（用于基本面/估值等非价格数据）
# ============================================================

def get_trade_calendar(start, end):
    cal = pro.trade_cal(exchange="SSE",
                        start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""))
    return sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())


def download_stock_basics():
    """全A股基本信息"""
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name,industry,list_date,market")
    df = df[df["ts_code"].str.match(r"^(00[0-2]|60)")]
    df = df[~df["name"].str.contains("ST|退", na=False)]
    return df.reset_index(drop=True)


def download_quarterly_snapshots(trade_dates, basics_codes):
    """下载每50交易日的 daily_basic 快照（PE/PB/市值/换手率/股息率）"""
    snap_dates = [trade_dates[i] for i in range(0, len(trade_dates), 50)]
    all_snaps = []
    for d in snap_dates:
        try:
            df = pro.daily_basic(trade_date=d,
                                 fields="ts_code,trade_date,pe_ttm,pb,total_mv,"
                                        "turnover_rate,dv_ratio")
            if df is not None and not df.empty:
                all_snaps.append(df)
        except Exception as e:
            print(f"    [警告] daily_basic {d}: {e}")
        time.sleep(0.15)
    if not all_snaps:
        return pd.DataFrame()
    result = pd.concat(all_snaps, ignore_index=True)
    print(f"    daily_basic 快照：{len(result):,} 条，{result['trade_date'].nunique()} 天")
    return result


def download_fina_indicators(codes):
    """下载财务指标（支持 point-in-time）"""
    all_data = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            df = pro.fina_indicator(ts_code=code,
                                    fields="ts_code,ann_date,end_date,"
                                           "roe_dt,grossprofit_margin,"
                                           "debt_to_assets,or_yoy,netprofit_yoy,ocfps")
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"    财报进度：{i+1}/{total}")
            time.sleep(0.5)
        time.sleep(0.06)
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    print(f"    财务数据：{len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result


def download_stock_daily(codes, start, end, label="个股"):
    """分批下载股票日线"""
    real_start = (pd.Timestamp(start) - pd.tseries.offsets.BDay(80)).strftime("%Y%m%d")
    real_end = end.replace("-", "")
    all_chunks = []
    batch = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            df = pro.daily(ts_code=code, start_date=real_start, end_date=real_end,
                           fields="ts_code,trade_date,open,high,low,close,vol,amount")
            if df is not None and not df.empty:
                batch.append(df)
        except Exception:
            pass
        if len(batch) >= 50:
            all_chunks.append(pd.concat(batch, ignore_index=True))
            batch = []
            gc.collect()
        if (i + 1) % 50 == 0:
            print(f"    {label}：{i+1}/{total}")
        time.sleep(0.06)
    if batch:
        all_chunks.append(pd.concat(batch, ignore_index=True))
    if not all_chunks:
        return pd.DataFrame()
    result = pd.concat(all_chunks, ignore_index=True)
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"    {label} 完成：{len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result



# ============================================================
# v18a新增：东财概念板块数据下载
# ============================================================

def download_dc_index_history(trade_dates):
    """
    下载东财概念板块每日涨跌数据（dc_index）。
    每个交易日一次调用，返回当天所有概念板块的涨跌幅。
    """
    print(f"[概念板块] 下载dc_index：{len(trade_dates)} 个交易日")
    all_data = []
    t0 = time.time()

    for i, date in enumerate(trade_dates):
        try:
            df = pro.dc_index(trade_date=date,
                              fields="ts_code,name,trade_date,pct_change,turnover_rate")
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(trade_dates) - i - 1) / speed / 60 if speed > 0 else 0
            print(f"    进度：{i+1}/{len(trade_dates)}  预计剩余{eta:.1f}分钟")
        time.sleep(0.35)  # 8000积分：200次/分

    if not all_data:
        print("[概念板块] dc_index无数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["pct_change"] = pd.to_numeric(result["pct_change"], errors="coerce")
    print(f"[概念板块] dc_index完成：{len(result):,} 条，"
          f"{result['trade_date'].nunique()} 天，{result['ts_code'].nunique()} 个概念")
    return result


def download_concept_stock_mapping(sample_date=None):
    """
    下载东财概念板块成分（dc_member），构建 stock→[concept_codes] 映射。
    使用单一日期的快照（概念成分变化较慢），一次性下载所有概念的成分。
    """
    if sample_date is None:
        sample_date = "20260101"  # 用一个较新的日期

    # 先获取所有概念代码
    print(f"[概念映射] 获取概念列表（{sample_date}）...")
    try:
        idx = pro.dc_index(trade_date=sample_date, fields="ts_code,name")
        if idx is None or idx.empty:
            # 如果指定日期没有数据，往前找
            for offset in range(1, 30):
                d = (pd.Timestamp(sample_date) - pd.DateOffset(days=offset)).strftime("%Y%m%d")
                try:
                    idx = pro.dc_index(trade_date=d, fields="ts_code,name")
                    if idx is not None and not idx.empty:
                        sample_date = d
                        break
                except Exception:
                    pass
                time.sleep(0.3)
    except Exception as e:
        print(f"[概念映射] 获取概念列表失败: {e}")
        return {}

    if idx is None or idx.empty:
        print("[概念映射] 无法获取概念列表")
        return {}

    concept_codes = idx["ts_code"].unique().tolist()
    concept_names = dict(zip(idx["ts_code"], idx["name"]))
    print(f"[概念映射] 共 {len(concept_codes)} 个概念板块，开始下载成分...")

    # 逐概念下载成分股
    stock_to_concepts = defaultdict(set)  # stock → {concept_code, ...}
    t0 = time.time()

    for i, ccode in enumerate(concept_codes):
        try:
            df = pro.dc_member(trade_date=sample_date, ts_code=ccode,
                               fields="ts_code,con_code")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    stock_to_concepts[row["con_code"]].add(ccode)
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(concept_codes) - i - 1) / speed / 60 if speed > 0 else 0
            print(f"    成分进度：{i+1}/{len(concept_codes)}  "
                  f"已映射{len(stock_to_concepts)}只股票  预计剩余{eta:.1f}分钟")
        time.sleep(0.35)

    # 转为普通dict
    stock_to_concepts = {k: list(v) for k, v in stock_to_concepts.items()}
    print(f"[概念映射] 完成：{len(stock_to_concepts)} 只股票映射到概念")
    return stock_to_concepts


def calc_concept_heat(date, dc_index_by_date, stock_concepts):
    """
    计算每只股票的概念热度。
    dc_index_by_date: {trade_date: {concept_code: pct_rank}} 预建索引
    返回 {ts_code: concept_heat_percentile}  范围0~100
    """
    if not dc_index_by_date or not stock_concepts:
        return {}

    concept_rank = dc_index_by_date.get(date, {})
    if not concept_rank:
        return {}

    result = {}
    for stock, concepts in stock_concepts.items():
        ranks = [concept_rank[c] for c in concepts if c in concept_rank]
        if ranks:
            result[stock] = max(ranks)

    return result


# ============================================================
# v18c新增：筹码成本和胜率数据下载
# ============================================================

def download_cyq_perf(codes, start, end):
    """
    下载个股筹码胜率数据（cyq_perf），按股票逐只查询。
    只下载winner_rate字段，节省内存。
    """
    print(f"[筹码胜率] 下载cyq_perf：{len(codes)} 只股票")
    all_data = []
    t0 = time.time()

    for i, code in enumerate(codes):
        try:
            df = pro.cyq_perf(ts_code=code, start_date=start, end_date=end,
                              fields="ts_code,trade_date,winner_rate")
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(codes) - i - 1) / speed / 60 if speed > 0 else 0
            print(f"    进度：{i+1}/{len(codes)}  预计剩余{eta:.1f}分钟")
        time.sleep(0.15)  # 8000积分：500次/分

    if not all_data:
        print("[筹码胜率] 无数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["winner_rate"] = pd.to_numeric(result["winner_rate"], errors="coerce")
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[筹码胜率] 完成：{len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result


def load_or_download(start, end, refresh=False):
    """
    v14e数据加载：
    1. 加载v5k缓存（基本面、估值、stock_daily等）
    2. 下载pytdx 30分钟K线（单独缓存，速度快）
    3. 构造14:30伪日线
    4. pytdx覆盖到的日期用14:30价格，更早的日期回退到v5k日线close
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"v11_{start.replace('-','')}_{end.replace('-','')}"
    v5k_cache = CACHE_DIR / f"{tag}.pkl"
    tdx_cache = CACHE_DIR / f"pytdx_30min_1430_{start.replace('-','')}_{end.replace('-','')}.pkl"

    # ── Step 1: 加载v5k缓存 ──
    if not v5k_cache.exists():
        print(f"[错误] 未找到v5k缓存：{v5k_cache}")
        print(f"  请先运行v5k回测生成缓存")
        sys.exit(1)

    print(f"[数据] 加载v5k缓存：{v5k_cache}")
    with open(v5k_cache, "rb") as f:
        v5k_data = pickle.load(f)

    all_dates = v5k_data["all_trade_dates"]
    bt_dates = v5k_data["bt_trade_dates"]
    fina_data = v5k_data["fina_data"]
    val_snaps = v5k_data["val_snaps"]
    basics = v5k_data["basics"]
    v5k_stock_daily = v5k_data["stock_daily"]
    v5k_hs300 = v5k_data["hs300_data"]

    need_codes = sorted(v5k_stock_daily["ts_code"].unique().tolist())
    print(f"  v5k缓存已加载：{len(need_codes)} 只\n")

    # ── Step 2: pytdx 30分钟K线（两阶段缓存）──
    raw_cache = CACHE_DIR / f"pytdx_raw30min_1430_{start.replace('-','')}_{end.replace('-','')}.pkl"
    
    if tdx_cache.exists() and not refresh:
        # 最终结果缓存存在，直接加载
        print(f"[pytdx] 结果缓存加载：{tdx_cache}")
        with open(tdx_cache, "rb") as f:
            tdx_data = pickle.load(f)
        daily_1430 = tdx_data["daily_1430"]
        hs300_1430 = tdx_data["hs300_1430"]
    else:
        real_start = (pd.Timestamp(start) - pd.tseries.offsets.BDay(80)).strftime("%Y%m%d")

        # 阶段1：下载原始30分钟数据（可复用已有缓存，避免重复下载）
        raw_cache_orig = CACHE_DIR / f"pytdx_raw30min_{start.replace('-','')}_{end.replace('-','')}.pkl"
        if raw_cache.exists() and not refresh:
            print(f"[pytdx] 原始数据缓存加载：{raw_cache}")
            with open(raw_cache, "rb") as f:
                raw_data = pickle.load(f)
            min30 = raw_data["min30"]
        elif raw_cache_orig.exists() and not refresh:
            print(f"[pytdx] 复用已有原始数据缓存：{raw_cache_orig}")
            with open(raw_cache_orig, "rb") as f:
                raw_data = pickle.load(f)
            min30 = raw_data["min30"]
        else:
            min30 = download_30min_pytdx(need_codes, min_date=real_start)
            raw_data = {"min30": min30}
            with open(raw_cache, "wb") as f:
                pickle.dump(raw_data, f)
            print(f"[pytdx] 原始数据已缓存 → {raw_cache}")

        # 阶段2：构造14:30伪日线（向量化，几秒完成）
        daily_1430 = build_1430_daily_from_pytdx(min30) if not min30.empty else pd.DataFrame()

        # HS300 ETF 14:30价格
        hs300_1430 = download_hs300_1430_pytdx(min_date=real_start)

        # 缓存最终结果
        tdx_data = {"daily_1430": daily_1430, "hs300_1430": hs300_1430}
        with open(tdx_cache, "wb") as f:
            pickle.dump(tdx_data, f)
        print(f"[pytdx] 结果已缓存 → {tdx_cache}\n")

    # ── Step 3: 合并——pytdx有的日期用14:30价格，没有的回退到v5k ──
    if daily_1430.empty:
        print("[警告] pytdx数据为空，回退到v5k日线（close=15:00收盘价）")
        stock_daily_final = v5k_stock_daily
        hs300_final = v5k_hs300
    else:
        tdx_date_min = daily_1430["trade_date"].min()
        tdx_date_max = daily_1430["trade_date"].max()
        print(f"[合并] pytdx 14:30数据范围：{tdx_date_min} ~ {tdx_date_max}")

        # 14:30伪日线添加name和industry
        name_map = dict(zip(basics["ts_code"], basics["name"]))
        ind_map = dict(zip(basics["ts_code"], basics["industry"]))
        daily_1430["name"] = daily_1430["ts_code"].map(name_map).fillna(daily_1430["ts_code"])
        daily_1430["industry"] = daily_1430["ts_code"].map(ind_map).fillna("")

        # v5k日线中早于pytdx范围的部分保留
        v5k_early = v5k_stock_daily[v5k_stock_daily["trade_date"] < tdx_date_min]
        # pytdx范围内用14:30数据
        stock_daily_final = pd.concat([v5k_early, daily_1430], ignore_index=True)
        stock_daily_final = stock_daily_final.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 确保name和industry列存在
        if "name" not in stock_daily_final.columns:
            stock_daily_final["name"] = stock_daily_final["ts_code"].map(name_map).fillna(stock_daily_final["ts_code"])
        if "industry" not in stock_daily_final.columns:
            stock_daily_final["industry"] = stock_daily_final["ts_code"].map(ind_map).fillna("")

        # HS300合并
        if not hs300_1430.empty:
            hs300_early = v5k_hs300[v5k_hs300["trade_date"] < hs300_1430["trade_date"].min()]
            hs300_final = pd.concat([hs300_early, hs300_1430], ignore_index=True)
            hs300_final = hs300_final.sort_values("trade_date").reset_index(drop=True)
        else:
            hs300_final = v5k_hs300

        n_1430 = len(daily_1430)
        n_v5k = len(v5k_early)
        print(f"  最终日线：{n_v5k:,} 条v5k + {n_1430:,} 条14:30 = {len(stock_daily_final):,} 条")

    # ── Step 4: 下载东财概念板块数据（v18a新增）──
    dc_cache = CACHE_DIR / f"dc_concept_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if dc_cache.exists() and not refresh:
        print(f"[概念板块] 缓存加载：{dc_cache}")
        with open(dc_cache, "rb") as f:
            dc_data = pickle.load(f)
        dc_index_hist = dc_data.get("dc_index", pd.DataFrame())
        stock_concepts = dc_data.get("stock_concepts", {})
    else:
        # dc_index数据从2020年开始
        dc_dates = [d for d in bt_dates if d >= "20200101"]
        dc_index_hist = download_dc_index_history(dc_dates) if dc_dates else pd.DataFrame()

        # 下载股票→概念映射（用回测区间末尾附近的日期）
        stock_concepts = download_concept_stock_mapping(sample_date=end.replace("-", ""))

        dc_data = {"dc_index": dc_index_hist, "stock_concepts": stock_concepts}
        with open(dc_cache, "wb") as f:
            pickle.dump(dc_data, f)
        print(f"[概念板块] 已缓存 → {dc_cache}")

    if not dc_index_hist.empty:
        print(f"[概念板块] dc_index: {dc_index_hist['trade_date'].nunique()} 天，"
              f"stock映射: {len(stock_concepts)} 只")

    # ── Step 5: 下载筹码胜率数据（v18c新增）──
    cyq_cache = CACHE_DIR / f"cyq_perf_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if cyq_cache.exists() and not refresh:
        print(f"[筹码胜率] 缓存加载：{cyq_cache}")
        with open(cyq_cache, "rb") as f:
            cyq_data = pickle.load(f)
    else:
        cyq_codes = stock_daily_final["ts_code"].unique().tolist()
        cyq_data = download_cyq_perf(cyq_codes, start.replace("-", ""), end.replace("-", ""))
        with open(cyq_cache, "wb") as f:
            pickle.dump(cyq_data, f)
        print(f"[筹码胜率] 已缓存 → {cyq_cache}")

    # 合并winner_rate到stock_daily
    if not cyq_data.empty:
        cyq_slim = cyq_data[["ts_code", "trade_date", "winner_rate"]].copy()
        stock_daily_final = stock_daily_final.merge(
            cyq_slim, on=["ts_code", "trade_date"], how="left"
        )
        n_wr = stock_daily_final["winner_rate"].notna().sum()
        print(f"[筹码胜率] 合并完成：{n_wr:,} 条有胜率数据")
    else:
        stock_daily_final["winner_rate"] = np.nan
        print("[筹码胜率] 无数据，winner_rate全部为NaN")

    data = {
        "all_trade_dates": all_dates,
        "bt_trade_dates": bt_dates,
        "hs300_data": hs300_final,
        "stock_daily": stock_daily_final,
        "fina_data": fina_data,
        "val_snaps": val_snaps,
        "basics": basics,
        "dc_index_hist": dc_index_hist,       # v18a
        "stock_concepts": stock_concepts,      # v18a
    }
    return data


# ============================================================
# Step3: 基本面评分（替代硬筛选）
# ============================================================

def _normalize_series(s, clip_low=None, clip_high=None):
    """将Series标准化到0~100"""
    s = s.copy()
    if clip_low is not None:
        s = s.clip(lower=clip_low)
    if clip_high is not None:
        s = s.clip(upper=clip_high)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100


def score_fundamentals(fina_data, val_snaps, basics, date):
    """
    返回 {ts_code: {"passed": bool, "fund_score": float, "is_tech": bool}}
    - passed: 是否通过基本面门槛（宽松）
    - fund_score: 0~100的评分（区分好公司和更好的公司）
    """
    result = {}
    if fina_data.empty:
        return result

    # 取point-in-time可用的最新财报
    available = fina_data[fina_data["ann_date"] <= date].copy()
    if available.empty:
        return result
    latest = available.sort_values("end_date").groupby("ts_code").last().reset_index()

    ind_map = dict(zip(basics["ts_code"], basics["industry"]))
    latest["industry"] = latest["ts_code"].map(ind_map).fillna("")
    latest["is_tech"] = latest["industry"].isin(TECH_INDUSTRIES)

    # ── 门槛筛选（通过/不通过）──
    std_mask = (
        (~latest["is_tech"]) &
        (latest["roe_dt"] >= MIN_ROE) &
        (latest["grossprofit_margin"] >= MIN_GROSS_MARGIN) &
        (latest["or_yoy"] >= MIN_REVENUE_GROWTH)
    )
    tech_mask = (
        (latest["is_tech"]) &
        (latest["roe_dt"] >= TECH_MIN_ROE) &
        (latest["grossprofit_margin"] >= TECH_MIN_GROSS_MARGIN) &
        (latest["or_yoy"] >= MIN_REVENUE_GROWTH)
    )
    pass_mask = std_mask | tech_mask

    # 负债率（金融豁免）
    non_fin = ~latest["industry"].isin(FINANCIAL_INDUSTRIES)
    debt_mask = (latest["debt_to_assets"] < MAX_DEBT_RATIO) | (~non_fin)
    pass_mask = pass_mask & debt_mask

    # 估值筛选
    val_codes = set()
    if not val_snaps.empty:
        recent_snaps = val_snaps[val_snaps["trade_date"] <= date]
        if not recent_snaps.empty:
            latest_snap_date = recent_snaps["trade_date"].max()
            snap = recent_snaps[recent_snaps["trade_date"] == latest_snap_date].copy()
            snap["industry"] = snap["ts_code"].map(ind_map).fillna("")
            snap["is_tech"] = snap["industry"].isin(TECH_INDUSTRIES)

            std_val = snap[~snap["is_tech"]]
            std_ok = std_val[
                (std_val["pe_ttm"] > PE_RANGE[0]) & (std_val["pe_ttm"] < PE_RANGE[1]) &
                (std_val["pb"] > PB_RANGE[0]) & (std_val["pb"] < PB_RANGE[1]) &
                (std_val["total_mv"] > MIN_MARKET_CAP * 10000)
            ]
            tech_val = snap[snap["is_tech"]]
            tech_ok = tech_val[
                (tech_val["pe_ttm"] > TECH_PE_RANGE[0]) & (tech_val["pe_ttm"] < TECH_PE_RANGE[1]) &
                (tech_val["pb"] > PB_RANGE[0]) & (tech_val["pb"] < PB_RANGE[1]) &
                (tech_val["total_mv"] > TECH_MIN_MARKET_CAP * 10000)
            ]
            val_codes = set(std_ok["ts_code"].tolist()) | set(tech_ok["ts_code"].tolist())

    # ── 基本面评分 ──
    # 对通过门槛的股票计算综合分
    passed_idx = latest[pass_mask].index
    if len(passed_idx) == 0:
        return result

    sub = latest.loc[passed_idx].copy()

    # ROE 评分（clip到0~50）
    roe_score = _normalize_series(sub["roe_dt"].fillna(0), clip_low=0, clip_high=50)
    # 营收增速评分（clip到-10~80）
    rev_score = _normalize_series(sub["or_yoy"].fillna(0), clip_low=-10, clip_high=80)
    # 毛利率评分（clip到10~80）
    gm_score = _normalize_series(sub["grossprofit_margin"].fillna(0), clip_low=10, clip_high=80)

    # 主评分 = 0.4*ROE + 0.3*营收 + 0.3*毛利率（严格按指南）
    fund_scores = (
        FUND_WEIGHT_ROE * roe_score +
        FUND_WEIGHT_REVENUE * rev_score +
        FUND_WEIGHT_MARGIN * gm_score
    )

    # 可选增强：经营现金流 + 净利润增速 作为额外加成（不稀释主权重）
    ocf_score = _normalize_series(sub["ocfps"].fillna(0), clip_low=-1, clip_high=10)
    np_score = _normalize_series(sub["netprofit_yoy"].fillna(0), clip_low=-20, clip_high=100)
    fund_scores = fund_scores + FUND_BONUS_CASHFLOW * ocf_score + FUND_BONUS_PEG * np_score

    for idx, row in sub.iterrows():
        code = row["ts_code"]
        passed = True
        if val_codes and code not in val_codes:
            passed = False
        result[code] = {
            "passed": passed,
            "fund_score": float(fund_scores.get(idx, 0)),
            "is_tech": bool(row["is_tech"]),
            "industry": row["industry"],
        }

    return result


# ============================================================
# Step4: 行业热度（多周期 + 排名）
# ============================================================

def calc_industry_heat(indicators, date):
    """
    多周期行业热度 + 百分位排名
    返回 {industry: heat_percentile}  范围0~100
    """
    industry_data = defaultdict(lambda: {"mom5": [], "mom20": [], "vol_ratio": []})

    for ts_code, idf in indicators.items():
        row = idf[idf["trade_date"] == date]
        if row.empty:
            continue
        row = row.iloc[0]
        mom5 = row.get("mom5", np.nan)
        mom20 = row.get("mom20", np.nan)
        vol_r = row.get("vol_ratio", np.nan)
        ind = row.get("industry", "")
        if ind and pd.notna(mom20):
            if pd.notna(mom5):
                industry_data[ind]["mom5"].append(mom5)
            industry_data[ind]["mom20"].append(mom20)
            if pd.notna(vol_r):
                industry_data[ind]["vol_ratio"].append(vol_r)

    # 计算原始热度分
    raw_heat = {}
    for ind, d in industry_data.items():
        if len(d["mom20"]) < 3:
            continue
        m5 = np.mean(d["mom5"]) if d["mom5"] else 0
        m20 = np.mean(d["mom20"])
        # 多周期动量：短期50% + 中期50%
        heat = 0.5 * m5 + 0.5 * m20
        # 资金维度加成：行业平均放量
        if d["vol_ratio"]:
            avg_vr = np.mean(d["vol_ratio"])
            if avg_vr > 1.2:
                heat *= 1.1  # 放量行业加10%
        raw_heat[ind] = heat

    if not raw_heat:
        return {}

    # 排名百分位化
    sorted_inds = sorted(raw_heat.items(), key=lambda x: x[1])
    n = len(sorted_inds)
    heat_pct = {}
    for rank, (ind, _) in enumerate(sorted_inds):
        heat_pct[ind] = (rank / max(n - 1, 1)) * 100

    return heat_pct


# ============================================================
# Step6: 动态红利股筛选
# ============================================================

def screen_dynamic_dividends(val_snaps, indicators, basics, date):
    """
    动态筛选红利防守股：高股息率 + 低波动 + 大市值
    返回 set of ts_codes
    """
    dividend_codes = set(DIVIDEND_SEED_POOL.keys())  # 兜底

    if val_snaps.empty:
        return dividend_codes

    recent_snaps = val_snaps[val_snaps["trade_date"] <= date]
    if recent_snaps.empty:
        return dividend_codes

    latest_snap_date = recent_snaps["trade_date"].max()
    snap = recent_snaps[recent_snaps["trade_date"] == latest_snap_date].copy()

    # 筛选条件：市值>200亿 + 股息率>2%
    snap = snap[
        (snap["total_mv"] > 200 * 10000) &
        (snap["dv_ratio"].fillna(0) > 2.0)
    ]

    if snap.empty:
        return dividend_codes

    # 计算波动率（从indicators中取）
    vol_dict = {}
    for code in snap["ts_code"].tolist():
        if code in indicators:
            idf = indicators[code]
            row = idf[idf["trade_date"] <= date].tail(1)
            if not row.empty and "volatility20" in row.columns:
                vol_dict[code] = float(row.iloc[0]["volatility20"])

    snap["volatility"] = snap["ts_code"].map(vol_dict)

    # 排序：高股息率 + 低波动 + 大市值（严格按指南三维度）
    # 波动率缺失的排在后面（fillna用大值）
    snap["vol_sort"] = snap["volatility"].fillna(999)
    snap = snap.sort_values(
        ["dv_ratio", "vol_sort", "total_mv"],
        ascending=[False, True, False]
    )

    # 取Top30
    dynamic_codes = set(snap.head(30)["ts_code"].tolist())

    # 与种子池合并
    dynamic_codes |= dividend_codes

    return dynamic_codes


# ============================================================
# 技术指标（增加 ATR / mom5 / MA10 / 波动率）
# ============================================================

def calc_indicators(df):
    df = df.sort_values("trade_date").copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    # 均线
    df["ma5"]  = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()

    # 新高
    df["high20"] = c.rolling(BREAKOUT_DAYS).max()

    # ── Step2: 回踩确认 - 过去5天内是否突破过20日新高 ──
    # 标记每天是否触及新高
    is_breakout = (c >= df["high20"].shift(1) * 0.99).astype(int)
    df["recent_breakout"] = is_breakout.rolling(PULLBACK_WINDOW).max()
    # 当前是否不是最高点（即回调了）
    df["not_at_high"] = (c < c.rolling(PULLBACK_WINDOW).max()).astype(int)

    # 成交量
    df["vol_ma5"] = df["vol"].rolling(5).mean()
    df["vol_ma20"] = df["vol"].rolling(20).mean()
    df["vol_ratio"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)

    # 动量
    df["mom5"] = (c / c.shift(5) - 1) * 100
    df["mom20"] = (c / c.shift(20) - 1) * 100

    # ── Step2: ATR波动过滤 ──
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / c  # ATR/close

    # 20日波动率（用于红利股筛选）
    df["volatility20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── Step1: MA5连续跌破天数（BEAR收紧用）──
    below_ma5 = (c < df["ma5"]).astype(int)
    streaks_5 = []; cnt = 0
    for v in below_ma5:
        cnt = cnt + 1 if v else 0
        streaks_5.append(cnt)
    df["below_ma5_streak"] = streaks_5

    # MA10连续跌破天数
    below_ma10 = (c < df["ma10"]).astype(int)
    streaks_10 = []; cnt = 0
    for v in below_ma10:
        cnt = cnt + 1 if v else 0
        streaks_10.append(cnt)
    df["below_ma10_streak"] = streaks_10

    # MA20连续跌破天数
    below_ma20 = (c < df["ma20"]).astype(int)
    streaks_20 = []; cnt = 0
    for v in below_ma20:
        cnt = cnt + 1 if v else 0
        streaks_20.append(cnt)
    df["below_ma20_streak"] = streaks_20

    # v18c新增：筹码胜率（shift(1)因为盘后18-19点更新，14:30只能看到T-1的）
    if "winner_rate" in df.columns:
        df["wr_prev"] = df["winner_rate"].shift(1)  # T-1的胜率
    else:
        df["wr_prev"] = np.nan

    return df


# ============================================================
# 市场环境判断（拓宽NEUTRAL缓冲区）
# ============================================================

def get_regime(hs300_data, date):
    hs = hs300_data[hs300_data["trade_date"] <= date].sort_values("trade_date")
    if len(hs) < 60:
        return "NEUTRAL"
    c = hs["close"].astype(float)
    cur = float(c.iloc[-1])
    ma20 = c.tail(20).mean()
    ma60 = c.tail(60).mean()

    # BULL：价格和MA20都在MA60上方（趋势确立）
    if cur > ma60 and ma20 > ma60:
        return "BULL"
    # NEUTRAL缓冲区：价格在MA60附近（上方或下方3%以内）
    # 这让NEUTRAL成为BULL和BEAR之间的真正过渡区
    if cur > ma60 * 0.97:
        return "NEUTRAL"
    # BEAR：价格明确跌破MA60的3%以下
    return "BEAR"


# ============================================================
# Step2: 买入信号（回踩确认 + 动量分段 + ATR过滤）
# ============================================================

def check_buy_v11(row):
    """
    v11买入条件（回踩确认版）：
    1. MA多头排列：close > MA20 > MA60
    2. 回踩确认：过去5天突破过20日新高（硬性条件）
       + 当前不在最高点（加分项，非硬过滤）
    3. 放量：5日均量/20日均量 >= 1.2
    4. 动量区间：5%~40%（25%以上降权）
    5. ATR过滤：ATR/close < 7%（排除过度波动，放宽以允许成长股）
    """
    close = row["close"]
    ma20 = row.get("ma20", np.nan)
    ma60 = row.get("ma60", np.nan)
    vol_ratio = row.get("vol_ratio", np.nan)
    mom = row.get("mom20", np.nan)
    recent_bk = row.get("recent_breakout", 0)
    not_at_high = row.get("not_at_high", 0)
    atr_r = row.get("atr_ratio", np.nan)

    if any(pd.isna(x) for x in [ma20, ma60, vol_ratio, mom, atr_r]):
        return False, "", 0

    # 1. MA多头
    if not (close > ma20 > ma60):
        return False, "", 0

    # 2. 回踩确认
    #    a) 近5天突破过20日新高（硬性条件）
    if recent_bk < 1:
        return False, "", 0
    #    b) 当前价格在MA20上方 → 已含在条件1

    # 3. 放量
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_RATIO:
        return False, "", 0

    # 4. 动量区间
    if mom < MIN_MOM20 or mom > MAX_MOM20:
        return False, "", 0

    # 5. ATR过滤
    if atr_r > ATR_THRESHOLD:
        return False, "", 0

    # ── 技术面评分 ──
    # 动量分段：5~25%满分区间，25~40%递减
    if mom <= MOM_PREFERRED:
        mom_score = (mom - MIN_MOM20) / (MOM_PREFERRED - MIN_MOM20) * 100
    else:
        mom_score = max(0, 100 - (mom - MOM_PREFERRED) / (MAX_MOM20 - MOM_PREFERRED) * 60)

    # 放量程度加分
    # v27b：放量评分封顶在2.5x，超过2.5x扣分
    if vol_ratio <= 2.5:
        vol_score = min(vol_ratio / VOLUME_RATIO * 50, 100)
    else:
        vol_score = max(0, 100 - (vol_ratio - 2.5) / 2.5 * 80)  # 超2.5x急剧扣分

    # 回踩质量加分：当前不在5日最高点（真回踩）+15分，在最高点不扣分
    pullback_bonus = 15 if not_at_high else 0

    # v18c：筹码胜率加成——T-1胜率>50%（多数持有者获利，抛压小）→ +10分
    wr = row.get("wr_prev", np.nan)
    wr_bonus = 10 if (pd.notna(wr) and wr > 50) else 0

    tech_score = mom_score * 0.5 + vol_score * 0.3 + pullback_bonus + wr_bonus

    wr_tag = f"|胜率{wr:.0f}%" if pd.notna(wr) else ""
    reason = f"回踩确认|放量{vol_ratio:.1f}x|动量{mom:.1f}%|ATR{atr_r*100:.1f}%{wr_tag}"
    return True, reason, tech_score


# ============================================================
# Step1: 卖出信号（回撤止盈 + 分批止盈 + MA10趋势止盈）
# ============================================================

def check_sell_v11(pos, row, market_ok, regime):
    """
    v11 review v3 卖出条件：
    - 不再用"大盘破MA60"作为个股卖出条件（regime统一管理市场风险）
    - BEAR/NEUTRAL时收紧止损参数，让还在涨的个股继续持有但保护更紧
    
    BULL模式：止损12% | 回撤止盈10% | MA10连续2天
    BEAR模式：止损5%  | 回撤止盈5%  | MA5跌破1天
    NEUTRAL模式：介于两者之间
    
    返回：(should_sell, reason, partial_ratio)
    """
    close = float(row["close"])
    cost = pos["cost"]
    peak = pos.get("peak_price", cost)
    pnl = (close - cost) / cost
    drawdown_from_peak = (close - peak) / peak if peak > 0 else 0

    partial_done_1 = pos.get("partial_1_done", False)
    partial_done_2 = pos.get("partial_2_done", False)

    # 根据regime选择止损参数
    if regime == "BEAR":
        eff_stop_loss = BEAR_STOP_LOSS_PCT
        eff_trail_stop = BEAR_TRAIL_STOP_PCT
        eff_trail_ma_key = "below_ma5_streak"
        eff_trail_ma_days = BEAR_TRAIL_MA_DAYS
        eff_trail_ma_label = "MA5"
    elif regime == "NEUTRAL":
        # NEUTRAL：中间值
        eff_stop_loss = (STOP_LOSS_PCT + BEAR_STOP_LOSS_PCT) / 2  # 8.5%
        eff_trail_stop = (TRAIL_STOP_PCT + BEAR_TRAIL_STOP_PCT) / 2  # 7.5%
        eff_trail_ma_key = "below_ma10_streak"
        eff_trail_ma_days = TRAIL_BELOW_MA10_DAYS  # 2天
        eff_trail_ma_label = "MA10"
    else:  # BULL
        eff_stop_loss = 0.15    # v5g改动：BULL硬止损从12%放宽到15%
        eff_trail_stop = 0.12   # v5c改动：BULL回撤止盈从10%放宽到12%
        eff_trail_ma_key = "below_ma10_streak"
        eff_trail_ma_days = 3   # v5j改动：BULL MA10趋势止盈从2天放宽到3天（减少牛市中假跌破）
        eff_trail_ma_label = "MA10"

    # 1. 硬止损（全卖）
    if pnl <= -eff_stop_loss:
        return True, f"止损{pnl*100:.1f}%(阈值{eff_stop_loss*100:.0f}%)", 1.0

    # 2. 回撤止盈（全卖，需先有5%以上盈利）
    if peak > cost * 1.05 and drawdown_from_peak <= -eff_trail_stop:
        return True, f"回撤止盈(峰值回落{drawdown_from_peak*100:.1f}%,阈值{eff_trail_stop*100:.0f}%)", 1.0

    # 3. 分批止盈（先①后②）
    if pnl >= PARTIAL_PROFIT_1 and not partial_done_1:
        return True, f"分批止盈①(+{pnl*100:.1f}%)", 0.5

    if pnl >= PARTIAL_PROFIT_2 and partial_done_1 and not partial_done_2:
        return True, f"分批止盈②(+{pnl*100:.1f}%)", 0.5

    # 4. 趋势止盈：跌破短期均线（有盈利时触发）
    if pnl > 0.03 and row.get(eff_trail_ma_key, 0) >= eff_trail_ma_days:
        return True, f"趋势止盈(破{eff_trail_ma_label}连续{row[eff_trail_ma_key]}天)", 1.0

    # 5. MA20趋势止损（任何regime都生效，最后防线）
    if row.get("below_ma20_streak", 0) >= TRAIL_BELOW_MA20_DAYS:
        return True, f"破MA20连续{row['below_ma20_streak']}天", 1.0

    return False, "", 0


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(data):
    all_dates  = data["all_trade_dates"]
    bt_dates   = data["bt_trade_dates"]
    hs300_data = data["hs300_data"]
    stock_daily = data["stock_daily"]
    fina_data  = data["fina_data"]
    val_snaps  = data["val_snaps"]
    basics     = data["basics"]
    dc_index_hist = data.get("dc_index_hist", pd.DataFrame())   # v18a
    stock_concepts = data.get("stock_concepts", {})              # v18a

    # v18a：预建dc_index按日期分组的索引（避免每次扫描全表）
    dc_index_by_date = {}
    if not dc_index_hist.empty:
        for date_key, grp in dc_index_hist.groupby("trade_date"):
            grp = grp.copy()
            grp["pct_rank"] = grp["pct_change"].rank(pct=True) * 100
            dc_index_by_date[date_key] = dict(zip(grp["ts_code"], grp["pct_rank"]))

    # 预计算技术指标
    print("[回测] 预计算技术指标...")
    indicators = {}
    all_codes = stock_daily["ts_code"].unique()
    for j, ts_code in enumerate(all_codes):
        sdf = stock_daily[stock_daily["ts_code"] == ts_code]
        indicators[ts_code] = calc_indicators(sdf)
        if (j + 1) % 200 == 0:
            print(f"  指标：{j+1}/{len(all_codes)}")
    print(f"  完成：{len(indicators)} 只")

    # 行业映射
    ind_map = dict(zip(basics["ts_code"], basics["industry"]))
    for ts_code in indicators:
        idf = indicators[ts_code]
        if "industry" not in idf.columns:
            idf["industry"] = ind_map.get(ts_code, "")

    capital = TOTAL_CAPITAL
    positions = {}     # {ts_code: {shares, cost, peak_price, ...}}
    nav_history = []
    trade_log = []
    days_in = 0; days_out = 0
    n_days = len(bt_dates)

    # 基本面评分缓存（每50天更新）
    fund_cache = {"date": "", "scores": {}}
    FUND_REFRESH_INTERVAL = 50

    # 红利股缓存
    dividend_cache = {"date": "", "codes": set(DIVIDEND_SEED_POOL.keys())}
    DIVIDEND_REFRESH = 50

    # 牛熊切换确认缓冲（避免whipsaw）
    confirmed_regime = "NEUTRAL"   # 当前已确认的regime
    pending_regime = "NEUTRAL"     # 待确认的新regime
    pending_count = 0              # 连续几天处于pending_regime

    print(f"\n[回测] {bt_dates[0]}~{bt_dates[-1]}，{n_days} 天")
    print(f"  卖出：止损{STOP_LOSS_PCT*100}% | 回撤止盈{TRAIL_STOP_PCT*100}% | "
          f"分批止盈{PARTIAL_PROFIT_1*100}%/{PARTIAL_PROFIT_2*100}%")
    print(f"  买入：回踩确认 | 动量{MIN_MOM20}~{MAX_MOM20}%(优选<{MOM_PREFERRED}%) | "
          f"ATR<{ATR_THRESHOLD*100}%")
    print(f"  仓位：BULL={REGIME_MAX_POSITION['BULL']*100}% | "
          f"NEUTRAL={REGIME_MAX_POSITION['NEUTRAL']*100}% | "
          f"BEAR={REGIME_MAX_POSITION['BEAR']*100}%")
    print(f"  牛熊切换确认：连续{REGIME_CONFIRM_DAYS}天\n")

    for i, date in enumerate(bt_dates):
        # ── 牛熊判断（带确认缓冲）──
        raw_regime = get_regime(hs300_data, date)
        if raw_regime == confirmed_regime:
            # 信号与当前一致，重置pending
            pending_regime = confirmed_regime
            pending_count = 0
        elif raw_regime == pending_regime:
            # 新信号持续，累计天数
            pending_count += 1
            if pending_count >= REGIME_CONFIRM_DAYS:
                confirmed_regime = pending_regime
                pending_count = 0
        else:
            # 信号又变了，重新开始计数
            pending_regime = raw_regime
            pending_count = 1

        regime = confirmed_regime
        market_ok = regime != "BEAR"

        # ── 更新基本面评分（每50天）──
        if i % FUND_REFRESH_INTERVAL == 0 or fund_cache["date"] == "":
            fund_scores = score_fundamentals(fina_data, val_snaps, basics, date)
            fund_cache = {"date": date, "scores": fund_scores}

        # ── 更新红利股（每50天）──
        if i % DIVIDEND_REFRESH == 0 or dividend_cache["date"] == "":
            dividend_codes = screen_dynamic_dividends(val_snaps, indicators, basics, date)
            dividend_cache = {"date": date, "codes": dividend_codes}
        else:
            dividend_codes = dividend_cache["codes"]

        # ── 更新持仓价格 & peak_price ──
        for code, pos in positions.items():
            idf = indicators.get(code)
            if idf is not None:
                row = idf[idf["trade_date"] == date]
                if not row.empty:
                    cur_price = float(row.iloc[0]["close"])
                    pos["current_price"] = cur_price
                    pos["last_update"] = date
                    # Step1: 跟踪持仓最高价
                    if cur_price > pos.get("peak_price", 0):
                        pos["peak_price"] = cur_price
                else:
                    last = pos.get("last_update", pos.get("buy_date", date))
                    li = all_dates.index(last) if last in all_dates else 0
                    ci = all_dates.index(date) if date in all_dates else 0
                    pos["halt_days"] = ci - li

        port_val = sum(p["shares"] * p["current_price"] for p in positions.values())
        total_val = capital + port_val
        nav = total_val / TOTAL_CAPITAL
        if positions:
            days_in += 1
        else:
            days_out += 1

        # ══════════════════════════════════════════
        # 卖出
        # ══════════════════════════════════════════
        for code, pos in list(positions.items()):
            # 停牌保护
            if pos.get("halt_days", 0) >= HALT_SELL_DAYS:
                sell_p = pos["current_price"] * (1 - SLIPPAGE)
                sell_amt = pos["shares"] * sell_p
                fee = sell_amt * (STOCK_COMMISSION + STAMP_DUTY)
                pnl = (sell_p - pos["cost"]) / pos["cost"]
                capital += sell_amt - fee
                hold = len([d for d in bt_dates if pos["buy_date"] <= d <= date])
                trade_log.append({"date": date, "action": f"卖出(停牌{pos['halt_days']}天)",
                    "code": code, "name": pos["name"], "price": round(sell_p, 3),
                    "shares": pos["shares"], "pnl_pct": round(pnl*100, 1),
                    "hold_days": hold, "fee": round(fee, 2)})
                del positions[code]
                continue

            idf = indicators.get(code)
            if idf is None:
                continue
            row = idf[idf["trade_date"] == date]
            if row.empty:
                continue
            row = row.iloc[0]

            # 不再强制清仓——让check_sell通过收紧的止损参数自然退出
            should_sell, reason, partial_ratio = check_sell_v11(pos, row, market_ok, regime)

            if should_sell:

                sell_shares = pos["shares"]
                if partial_ratio < 1.0:
                    # 分批止盈：卖出一半（取整到100股）
                    sell_shares = max(int(pos["shares"] * partial_ratio / 100) * 100, 100)
                    if sell_shares >= pos["shares"]:
                        sell_shares = pos["shares"]

                sell_p = pos["current_price"] * (1 - SLIPPAGE)
                sell_amt = sell_shares * sell_p
                fee = sell_amt * (STOCK_COMMISSION + STAMP_DUTY)
                pnl = (sell_p - pos["cost"]) / pos["cost"]
                capital += sell_amt - fee
                hold = len([d for d in bt_dates if pos["buy_date"] <= d <= date])

                trade_log.append({"date": date, "action": f"卖出({reason})",
                    "code": code, "name": pos["name"], "price": round(sell_p, 3),
                    "shares": sell_shares, "pnl_pct": round(pnl*100, 1),
                    "hold_days": hold, "fee": round(fee, 2)})

                if sell_shares >= pos["shares"]:
                    del positions[code]
                else:
                    pos["shares"] -= sell_shares
                    # 标记已完成第几次分批止盈
                    if "分批止盈①" in reason:
                        pos["partial_1_done"] = True
                    elif "分批止盈②" in reason:
                        pos["partial_2_done"] = True

        # ══════════════════════════════════════════
        # 买入（仅BULL开新仓；NEUTRAL/BEAR只持有存量，不新开仓）
        # ══════════════════════════════════════════
        # Step5: 动态仓位上限
        max_exposure = REGIME_MAX_POSITION.get(regime, 0.1)
        total_now = capital + sum(p["shares"] * p["current_price"] for p in positions.values())
        current_exposure = sum(p["shares"] * p["current_price"] for p in positions.values()) / total_now if total_now > 0 else 0

        if regime == "BULL" and current_exposure < max_exposure and len(positions) < MAX_POSITIONS:
            allowed_buy_amount = total_now * max_exposure - sum(
                p["shares"] * p["current_price"] for p in positions.values())
            allowed_buy_amount = max(allowed_buy_amount, 0)

            # 确定扫描范围（仅BULL进入此代码块）
            fund_scores = fund_cache["scores"]
            scan_codes = [c for c, s in fund_scores.items()
                          if s["passed"] and c in indicators and c not in positions]

            # 计算行业热度
            industry_heat = calc_industry_heat(indicators, date)

            # v18a：计算概念板块热度
            concept_heat_map = calc_concept_heat(date, dc_index_by_date, stock_concepts)

            candidates = []
            for ts_code in scan_codes:
                idf = indicators[ts_code]
                row = idf[idf["trade_date"] == date]
                if row.empty:
                    continue
                row = row.iloc[0]

                should_buy, reason, tech_score = check_buy_v11(row)
                if not should_buy:
                    continue

                # ── Step3: 综合评分（严格按指南：0.6技术面 + 0.4基本面）──
                fs = fund_scores.get(ts_code, {})
                fund_score = fs.get("fund_score", 0)

                ind = row.get("industry", "")
                ind_heat = industry_heat.get(ind, 50) if industry_heat else 50

                # v18a：融合行业热度和概念热度（各50%）
                # 无概念数据（2018-2019）时回退到纯行业热度，与v14e一致
                if concept_heat_map:
                    cpt_heat = concept_heat_map.get(ts_code, 50)
                    heat = 0.5 * ind_heat + 0.5 * cpt_heat
                else:
                    heat = ind_heat  # 回退到v14e

                # 行业热度并入技术面（技术面 = 原始tech_score*0.7 + 热度*0.3）
                tech_with_heat = tech_score * 0.7 + heat * 0.3

                # v20a：行业差异化权重
                # 低弹性行业（白酒等）基本面天然高但趋势弱→降低基本面权重
                if ind in LOW_ELASTIC_INDUSTRIES:
                    w_tech = LOW_ELASTIC_TECH_W
                    w_fund = LOW_ELASTIC_FUND_W
                else:
                    w_tech = SCORE_WEIGHT_TECH
                    w_fund = SCORE_WEIGHT_FUND

                final_score = w_tech * tech_with_heat + w_fund * fund_score

                candidates.append({
                    "ts_code": ts_code, "name": row.get("name", ts_code),
                    "close": float(row["close"]),
                    "mom20": float(row["mom20"]) if pd.notna(row.get("mom20")) else 0,
                    "industry": ind, "heat": round(heat, 1),
                    "fund_score": round(fund_score, 1),
                    "tech_score": round(tech_score, 1),
                    "final_score": round(final_score, 2),
                    "reason": reason,
                })

            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            slots = MAX_POSITIONS - len(positions)

            # Step5: 按评分分配仓位权重
            total_score = sum(c["final_score"] for c in candidates[:slots]) if candidates else 1
            if total_score <= 0:
                total_score = 1

            for c in candidates[:slots]:
                # 按评分比例分配（但不超过allowed_buy_amount）
                score_ratio = c["final_score"] / total_score
                per_pos = min(
                    allowed_buy_amount * score_ratio,
                    total_now * 0.25  # 单只上限25%
                )
                if per_pos < 5000:
                    continue

                price = c["close"] * (1 + SLIPPAGE)
                shares = int(per_pos / price / 100) * 100
                if shares < 100:
                    continue
                cost = shares * price
                fee = cost * STOCK_COMMISSION
                if cost + fee > capital:
                    shares = int(capital / (price * (1 + STOCK_COMMISSION)) / 100) * 100
                    if shares < 100:
                        continue
                    cost = shares * price
                    fee = cost * STOCK_COMMISSION
                capital -= cost + fee
                allowed_buy_amount -= cost

                positions[c["ts_code"]] = {
                    "shares": shares, "cost": price,
                    "current_price": price, "peak_price": price,
                    "name": c["name"],
                    "buy_date": date, "last_update": date, "halt_days": 0,
                    "partial_1_done": False, "partial_2_done": False,
                }
                trade_log.append({
                    "date": date, "action": "买入",
                    "code": c["ts_code"], "name": c["name"],
                    "price": round(price, 3), "shares": shares,
                    "pnl_pct": 0, "hold_days": 0, "fee": round(fee, 2),
                    "reason": (f"{c['reason']}|行业:{c['industry']}"
                               f"|热度:{c['heat']}|基本面:{c['fund_score']}"
                               f"|综合:{c['final_score']}"),
                })

        # ── 净值 ──
        port_val = sum(p["shares"] * p["current_price"] for p in positions.values())
        total_val = capital + port_val
        nav = total_val / TOTAL_CAPITAL
        nav_history.append({"date": date, "nav": round(nav, 4),
                            "capital": round(capital, 2),
                            "positions": len(positions), "regime": regime})

        if (i + 1) % 50 == 0:
            icon = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}[regime]
            names = ",".join(p["name"][:4] for p in list(positions.values())[:3])
            ps = f"{len(positions)}只({names})" if positions else "空仓"
            n_fund = len([s for s in fund_cache["scores"].values() if s["passed"]])
            exp_pct = current_exposure * 100
            raw_tag = f"(raw:{raw_regime})" if raw_regime != regime else ""
            print(f"  [{i+1}/{n_days}] {date}  净值:{nav:.4f}  {ps}  "
                  f"好公司:{n_fund}  仓位:{exp_pct:.0f}%/{max_exposure*100:.0f}%  "
                  f"资金:¥{capital:,.0f}  {icon}{regime}{raw_tag}")

    nav_df = pd.DataFrame(nav_history).set_index("date")
    trade_df = pd.DataFrame(trade_log)
    nav_df.attrs["days_in"] = days_in
    nav_df.attrs["days_out"] = days_out
    return nav_df, trade_df


# ============================================================
# 绩效报告
# ============================================================

def calc_perf(nav_df):
    nav = nav_df["nav"]; rets = nav.pct_change().dropna()
    n = len(nav); ny = n / 252
    tr = nav.iloc[-1] / nav.iloc[0] - 1
    ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    ex = rets - 0.02 / 252
    sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    cm = nav.cummax(); dd = (nav - cm) / cm
    mdd = dd.min(); dde = dd.idxmin(); dds = nav[:dde].idxmax()
    wr = (rets > 0).sum() / len(rets) if len(rets) > 0 else 0
    cal = ar / abs(mdd) if mdd != 0 else 0
    ns = nav.copy(); ns.index = pd.to_datetime(ns.index, format="%Y%m%d")
    mo = ns.resample("ME").last().pct_change().dropna()
    di = nav_df.attrs.get("days_in", 0); do = nav_df.attrs.get("days_out", 0)
    return {
        "回测区间": f"{nav_df.index[0]}~{nav_df.index[-1]}", "交易日": n,
        "总收益": f"{tr*100:+.2f}%", "年化": f"{ar*100:+.2f}%",
        "夏普": f"{sh:.3f}", "最大回撤": f"{mdd*100:.2f}%",
        "回撤区间": f"{dds}~{dde}", "Calmar": f"{cal:.3f}",
        "日胜率": f"{wr*100:.1f}%",
        "最佳月": f"{mo.max()*100:+.1f}%" if len(mo) > 0 else "-",
        "最差月": f"{mo.min()*100:+.1f}%" if len(mo) > 0 else "-",
        "持仓": f"{di}天({di/n*100:.0f}%)", "空仓": f"{do}天({do/n*100:.0f}%)",
        "期末净值": f"{nav.iloc[-1]:.4f}",
        "期末市值": f"¥{nav.iloc[-1]*TOTAL_CAPITAL:,.0f}",
    }


def print_report(perf, trade_df, nav_df):
    print("\n" + "=" * 70)
    print("📊 v27b 绩效报告（v20a + 放量>2.5x扣分）")
    print("=" * 70)
    for k, v in perf.items():
        print(f"  {k:<10}: {v}")
    print("=" * 70)

    if not trade_df.empty:
        buys = trade_df[trade_df["action"] == "买入"]
        sells = trade_df[trade_df["action"].str.startswith("卖出")]
        print(f"\n  交易：{len(trade_df)}次（买{len(buys)}，卖{len(sells)}）")

        if len(sells) > 0:
            print(f"  卖出均盈亏：{sells['pnl_pct'].mean():+.1f}%")
            w = sells[sells["pnl_pct"] > 0]; l = sells[sells["pnl_pct"] <= 0]
            if len(w):
                print(f"  盈利：{len(w)}次({len(w)/len(sells)*100:.0f}%)，均+{w['pnl_pct'].mean():.1f}%")
            if len(l):
                print(f"  亏损：{len(l)}次({len(l)/len(sells)*100:.0f}%)，均{l['pnl_pct'].mean():.1f}%")
            print(f"  均持有：{sells['hold_days'].mean():.0f}天")

            # 卖出原因统计
            print(f"\n  卖出原因分布：")
            reason_cats = {
                "止损": 0, "回撤止盈": 0, "分批止盈": 0,
                "趋势止盈(MA10)": 0, "破MA20": 0,
                "大盘破MA60": 0, "市场切换": 0, "停牌": 0, "其他": 0
            }
            for act in sells["action"]:
                if "止损" in act and "止盈" not in act:
                    reason_cats["止损"] += 1
                elif "回撤止盈" in act:
                    reason_cats["回撤止盈"] += 1
                elif "分批止盈" in act:
                    reason_cats["分批止盈"] += 1
                elif "MA10" in act or "趋势止盈" in act:
                    reason_cats["趋势止盈(MA10)"] += 1
                elif "MA20" in act:
                    reason_cats["破MA20"] += 1
                elif "MA60" in act:
                    reason_cats["大盘破MA60"] += 1
                elif "切换" in act:
                    reason_cats["市场切换"] += 1
                elif "停牌" in act:
                    reason_cats["停牌"] += 1
                else:
                    reason_cats["其他"] += 1
            for r, c in sorted(reason_cats.items(), key=lambda x: -x[1]):
                if c > 0:
                    print(f"    {r}：{c}次（{c/len(sells)*100:.0f}%）")

        print(f"\n  手续费：¥{trade_df['fee'].sum():,.0f}（{trade_df['fee'].sum()/TOTAL_CAPITAL*100:.2f}%）")

        if len(buys) > 0:
            print(f"\n  最常买入Top10：")
            for name, cnt in buys["name"].value_counts().head(10).items():
                print(f"    {name}：{cnt}次")
            if "reason" in buys.columns:
                industries = buys["reason"].str.extract(r"行业:(\w+)")[0].value_counts()
                print(f"\n  买入行业分布Top5：")
                for ind, cnt in industries.head(5).items():
                    print(f"    {ind}：{cnt}次")

    ns = nav_df["nav"].copy()
    ns.index = pd.to_datetime(ns.index, format="%Y%m%d")
    yr = ns.resample("YE").last()
    print(f"\n  年度：")
    prev = 1.0
    for d, v in yr.items():
        print(f"    {d.year}：{(v/prev-1)*100:+.1f}%（净值{v:.4f}）")
        prev = v
    for r in ["BULL", "NEUTRAL", "BEAR"]:
        c = (nav_df["regime"] == r).sum()
        print(f"  {r}: {c}天（{c/len(nav_df)*100:.0f}%）")
    print()


def save_results(nav_df, trade_df):
    d = CACHE_DIR / "results_v27b"
    d.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(d / "nav.csv")
    if not trade_df.empty:
        trade_df.to_csv(d / "trades.csv", index=False)
    print(f"  已保存 → {d}/")


def main():
    parser = argparse.ArgumentParser(description="v27b 放量回测")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print(f"  v27b（放量封顶）")
    print(f"{'='*75}")
    print(f"  区间：{args.start}~{args.end}")
    print(f"  ── 卖出（regime自适应）──")
    print(f"    BULL : 止损{STOP_LOSS_PCT*100}% | 回撤止盈{TRAIL_STOP_PCT*100}% | MA10连续{TRAIL_BELOW_MA10_DAYS}天")
    print(f"    BEAR : 止损{BEAR_STOP_LOSS_PCT*100}% | 回撤止盈{BEAR_TRAIL_STOP_PCT*100}% | MA5跌破{BEAR_TRAIL_MA_DAYS}天")
    print(f"    分批止盈:+{PARTIAL_PROFIT_1*100}%/+{PARTIAL_PROFIT_2*100}% | MA20连续{TRAIL_BELOW_MA20_DAYS}天(兜底)")
    print("    ❌ 不再使用大盘破MA60清仓 / 市场切换强制清仓")
    print(f"  ── 买入 ──")
    print(f"    回踩确认{PULLBACK_WINDOW}日(加分项) | 动量{MIN_MOM20}~{MAX_MOM20}%(优选<{MOM_PREFERRED}%)")
    print(f"    放量>{VOLUME_RATIO}x | ATR<{ATR_THRESHOLD*100:.0f}%")
    print(f"  ── 评分 ──")
    print(f"    最终 = 技术面(含行业热度){SCORE_WEIGHT_TECH*100:.0f}% + 基本面{SCORE_WEIGHT_FUND*100:.0f}%")
    print(f"    v20a: 低弹性行业({','.join(LOW_ELASTIC_INDUSTRIES)})改为技术{LOW_ELASTIC_TECH_W*100:.0f}%/基本面{LOW_ELASTIC_FUND_W*100:.0f}%")
    print(f"    基本面 = 0.4*ROE + 0.3*营收增速 + 0.3*毛利率 + 现金流/利润加成")
    print(f"  ── 仓位 ──")
    print(f"    BULL:{REGIME_MAX_POSITION['BULL']*100:.0f}% | "
          f"NEUTRAL:{REGIME_MAX_POSITION['NEUTRAL']*100:.0f}% | "
          f"BEAR:{REGIME_MAX_POSITION['BEAR']*100:.0f}%(不开新仓，存量收紧止损)")
    print(f"    NEUTRAL缓冲区：close > MA60*0.97")
    print(f"    牛熊切换需连续{REGIME_CONFIRM_DAYS}天确认")
    print(f"{'='*75}\n")

    data = load_or_download(args.start, args.end, refresh=args.refresh)
    nav_df, trade_df = run_backtest(data)
    perf = calc_perf(nav_df)
    print_report(perf, trade_df, nav_df)
    save_results(nav_df, trade_df)


if __name__ == "__main__":
    main()
