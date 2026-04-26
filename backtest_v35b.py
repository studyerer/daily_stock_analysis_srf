"""
v35b：v35a + 短期均线多头（MA5 > MA10）
==================================================================================
对照实验 Step 2：在 v35a 基础上新增一个条件——
  买入时要求 MA5 > MA10，确保短期趋势向上。

  目的：验证短期均线多头对胜率的提升效果。
  LLM 策略要求 MA5 > MA10 > MA20，这里先只加 MA5 > MA10。

累计变更（vs v27b）：
  条件6：bias_ma5 < 8%           (来自 v35a)
  条件7：MA5 > MA10              (v35b 新增)

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

REGIME_CONFIRM_DAYS = 5

# ── Step1: 卖出参数 ──
TRAIL_STOP_PCT       = 0.10
PARTIAL_PROFIT_1     = 0.20
PARTIAL_PROFIT_2     = 0.40
TRAIL_BELOW_MA10_DAYS = 2
TRAIL_BELOW_MA20_DAYS = 3

BEAR_STOP_LOSS_PCT       = 0.05
BEAR_TRAIL_STOP_PCT      = 0.05
BEAR_TRAIL_MA_DAYS       = 1

# ── Step2: 买入参数 ──
MIN_MOM20        = 5.0
MAX_MOM20        = 40.0
MOM_PREFERRED    = 25.0
VOLUME_RATIO     = 1.2
BREAKOUT_DAYS    = 20
PULLBACK_WINDOW  = 7
ATR_THRESHOLD    = 0.07
BIAS_MA5_MAX     = 8.0        # ★ v35a 新增

# ── Step3: 基本面参数 ──
MIN_ROE = 8.0
MIN_GROSS_MARGIN = 20.0
MAX_DEBT_RATIO = 70.0
PE_RANGE = (0, 80)
PB_RANGE = (0, 15)
MIN_REVENUE_GROWTH = 5.0
MIN_MARKET_CAP = 100
TECH_PE_RANGE = (0, 200)
TECH_MIN_ROE = 3.0
TECH_MIN_GROSS_MARGIN = 15.0
TECH_MIN_MARKET_CAP = 50

FUND_WEIGHT_ROE      = 0.40
FUND_WEIGHT_REVENUE  = 0.30
FUND_WEIGHT_MARGIN   = 0.30
FUND_BONUS_CASHFLOW  = 0.10
FUND_BONUS_PEG       = 0.05

SCORE_WEIGHT_TECH    = 0.60
SCORE_WEIGHT_FUND    = 0.40

LOW_ELASTIC_INDUSTRIES = {"白酒", "乳制品", "环境保护", "路桥", "仓储开发"}
LOW_ELASTIC_TECH_W   = 0.80
LOW_ELASTIC_FUND_W   = 0.20

# ── Step5: 仓位管理 ──
REGIME_MAX_POSITION = {
    "BULL":    1.00,
    "NEUTRAL": 0.10,
    "BEAR":    0.00,
}

STOCK_COMMISSION = 0.0003
STAMP_DUTY       = 0.0005
SLIPPAGE         = 0.001

DEFAULT_START = "2018-01-01"
DEFAULT_END   = "2026-03-30"
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "backtest_cache"

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

FINANCIAL_INDUSTRIES = {"银行", "保险", "证券", "多元金融", "银行I", "非银金融I"}
TECH_INDUSTRIES = {
    "半导体", "元器件", "光学光电子", "电子制造", "其他电子",
    "计算机应用", "计算机设备", "IT服务", "软件开发",
    "通信设备", "通信服务", "通信运营",
    "电气设备", "电源设备", "电源", "电力设备",
    "专用设备", "通用设备", "仪器仪表",
    "航天装备", "航空装备", "地面兵装", "船舶制造",
    "医疗器械", "化学制药", "生物制品", "中药",
    "汽车零件", "汽车整车",
}
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
# pytdx / Tushare 数据下载（与 v27b 完全相同，不修改）
# ============================================================

TDX_SERVERS = [
    ("119.147.212.81", 7709), ("114.80.63.12", 7709),
    ("218.75.126.9", 7709), ("124.74.236.94", 7721),
    ("221.231.141.60", 7709), ("59.173.18.140", 7709),
    ("112.74.214.43", 7727),
]

def _ts_code_to_tdx(ts_code):
    code, market = ts_code.split(".")
    return (1 if market == "SH" else 0), code

def download_30min_pytdx(ts_codes, min_date="20170101"):
    from pytdx.hq import TdxHq_API
    print(f"[pytdx] 下载30分钟K线：{len(ts_codes)} 只")
    api = TdxHq_API()
    connected = False
    for host, port in TDX_SERVERS:
        try:
            api.connect(host, port); connected = True
            print(f"  已连接：{host}:{port}"); break
        except Exception: continue
    if not connected:
        print("[pytdx] 无法连接任何通达信服务器"); return pd.DataFrame()
    all_data = []; total = len(ts_codes); t0 = time.time()
    for i, ts_code in enumerate(ts_codes):
        market, code = _ts_code_to_tdx(ts_code); stock_bars = []; offset = 0
        while True:
            try: bars = api.get_security_bars(2, market, code, offset, 800)
            except Exception: break
            if not bars or len(bars) == 0: break
            stock_bars.extend(bars); offset += 800
            if len(bars) < 800: break
        if stock_bars:
            df = pd.DataFrame(stock_bars); df["ts_code"] = ts_code
            df["trade_date"] = df["datetime"].str[:10].str.replace("-", "")
            df["hour"] = df["datetime"].str[11:13].astype(int)
            df["minute"] = df["datetime"].str[14:16].astype(int)
            df = df[df["trade_date"] >= min_date]
            if not df.empty:
                all_data.append(df[["ts_code","trade_date","hour","minute","open","high","low","close","vol"]])
        if (i+1) % 20 == 0:
            elapsed = time.time()-t0; speed = (i+1)/elapsed if elapsed>0 else 0
            eta = (total-i-1)/speed/60 if speed>0 else 0
            print(f"    进度：{i+1}/{total}  已用{elapsed/60:.1f}分钟  预计剩余{eta:.1f}分钟")
    api.disconnect()
    if not all_data: print("[pytdx] 未获取到数据"); return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    for col in ["open","high","low","close","vol"]: result[col] = pd.to_numeric(result[col], errors="coerce")
    print(f"[pytdx] 完成：{len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result

def build_1430_daily_from_pytdx(min30_data):
    if min30_data.empty: return pd.DataFrame()
    print(f"  构造14:30伪日线：{len(min30_data):,} 条30分钟数据...")
    data = min30_data.copy()
    mask = (data["hour"] < 14) | ((data["hour"] == 14) & (data["minute"] == 30))
    data = data[mask].copy()
    if data.empty: return pd.DataFrame()
    grp = data.groupby(["ts_code", "trade_date"])
    agg_df = grp.agg(high=("high","max"), low=("low","min"), vol=("vol","sum")).reset_index()
    data["time_key"] = data["hour"]*100 + data["minute"]
    idx_first = data.groupby(["ts_code","trade_date"])["time_key"].idxmin()
    open_df = data.loc[idx_first, ["ts_code","trade_date","open"]].reset_index(drop=True)
    bar_1400 = data[(data["hour"]==14)&(data["minute"]==30)][["ts_code","trade_date","close"]]
    idx_last = data.groupby(["ts_code","trade_date"])["time_key"].idxmax()
    close_fallback = data.loc[idx_last, ["ts_code","trade_date","close"]].reset_index(drop=True)
    close_df = close_fallback.merge(bar_1400, on=["ts_code","trade_date"], how="left", suffixes=("_fb","_1400"))
    close_df["close"] = close_df["close_1400"].fillna(close_df["close_fb"])
    close_df = close_df[["ts_code","trade_date","close"]]
    result = agg_df.merge(open_df, on=["ts_code","trade_date"])
    result = result.merge(close_df, on=["ts_code","trade_date"])
    result["amount"] = 0
    result = result[["ts_code","trade_date","open","high","low","close","vol","amount"]]
    result = result.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    print(f"[14:30伪日线] {len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result

def download_hs300_1430_pytdx(min_date="20170101"):
    from pytdx.hq import TdxHq_API
    api = TdxHq_API()
    for host, port in TDX_SERVERS:
        try: api.connect(host, port); break
        except Exception: continue
    all_bars = []; offset = 0
    while True:
        try: bars = api.get_security_bars(2, 1, "510300", offset, 800)
        except Exception: break
        if not bars or len(bars)==0: break
        all_bars.extend(bars); offset += 800
        if len(bars)<800: break
    api.disconnect()
    if not all_bars: return pd.DataFrame()
    df = pd.DataFrame(all_bars)
    df["trade_date"] = df["datetime"].str[:10].str.replace("-","")
    df["hour"] = df["datetime"].str[11:13].astype(int)
    df["minute"] = df["datetime"].str[14:16].astype(int)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["trade_date"] >= min_date]
    bar_1400 = df[(df["hour"]==14)&(df["minute"]==30)].copy()
    if bar_1400.empty: return pd.DataFrame()
    result = pd.DataFrame({"ts_code":"510300.SH","trade_date":bar_1400["trade_date"].values,"close":bar_1400["close"].values}).reset_index(drop=True)
    return result

def get_trade_calendar(start, end):
    cal = pro.trade_cal(exchange="SSE", start_date=start.replace("-",""), end_date=end.replace("-",""))
    return sorted(cal[cal["is_open"]==1]["cal_date"].tolist())

def download_stock_basics():
    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,list_date,market")
    df = df[df["ts_code"].str.match(r"^(00[0-2]|60)")]
    df = df[~df["name"].str.contains("ST|退", na=False)]
    return df.reset_index(drop=True)

def download_quarterly_snapshots(trade_dates, basics_codes):
    snap_dates = [trade_dates[i] for i in range(0, len(trade_dates), 50)]
    all_snaps = []
    for d in snap_dates:
        try:
            df = pro.daily_basic(trade_date=d, fields="ts_code,trade_date,pe_ttm,pb,total_mv,turnover_rate,dv_ratio")
            if df is not None and not df.empty: all_snaps.append(df)
        except Exception as e: print(f"    [警告] daily_basic {d}: {e}")
        time.sleep(0.15)
    if not all_snaps: return pd.DataFrame()
    result = pd.concat(all_snaps, ignore_index=True)
    print(f"    daily_basic 快照：{len(result):,} 条，{result['trade_date'].nunique()} 天")
    return result

def download_fina_indicators(codes):
    all_data = []; total = len(codes)
    for i, code in enumerate(codes):
        try:
            df = pro.fina_indicator(ts_code=code, fields="ts_code,ann_date,end_date,roe_dt,grossprofit_margin,debt_to_assets,or_yoy,netprofit_yoy,ocfps")
            if df is not None and not df.empty: all_data.append(df)
        except Exception: pass
        if (i+1)%100==0: print(f"    财报进度：{i+1}/{total}"); time.sleep(0.5)
        time.sleep(0.06)
    if not all_data: return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(["ts_code","end_date"]).reset_index(drop=True)
    print(f"    财务数据：{len(result):,} 条，{result['ts_code'].nunique()} 只")
    return result

def download_stock_daily(codes, start, end, label="个股"):
    real_start = (pd.Timestamp(start) - pd.tseries.offsets.BDay(80)).strftime("%Y%m%d")
    real_end = end.replace("-",""); all_chunks = []; batch = []; total = len(codes)
    for i, code in enumerate(codes):
        try:
            df = pro.daily(ts_code=code, start_date=real_start, end_date=real_end, fields="ts_code,trade_date,open,high,low,close,vol,amount")
            if df is not None and not df.empty: batch.append(df)
        except Exception: pass
        if len(batch)>=50: all_chunks.append(pd.concat(batch, ignore_index=True)); batch = []; gc.collect()
        if (i+1)%50==0: print(f"    {label}：{i+1}/{total}")
        time.sleep(0.06)
    if batch: all_chunks.append(pd.concat(batch, ignore_index=True))
    if not all_chunks: return pd.DataFrame()
    result = pd.concat(all_chunks, ignore_index=True)
    result = result.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    return result

def download_dc_index_history(trade_dates):
    print(f"[概念板块] 下载dc_index：{len(trade_dates)} 个交易日")
    all_data = []; t0 = time.time()
    for i, date in enumerate(trade_dates):
        try:
            df = pro.dc_index(trade_date=date, fields="ts_code,name,trade_date,pct_change,turnover_rate")
            if df is not None and not df.empty: all_data.append(df)
        except Exception: pass
        if (i+1)%100==0:
            elapsed=time.time()-t0; speed=(i+1)/elapsed if elapsed>0 else 0
            eta=(len(trade_dates)-i-1)/speed/60 if speed>0 else 0
            print(f"    进度：{i+1}/{len(trade_dates)}  预计剩余{eta:.1f}分钟")
        time.sleep(0.35)
    if not all_data: return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    result["pct_change"] = pd.to_numeric(result["pct_change"], errors="coerce")
    return result

def download_concept_stock_mapping(sample_date=None):
    if sample_date is None: sample_date = "20260101"
    print(f"[概念映射] 获取概念列表（{sample_date}）...")
    try:
        idx = pro.dc_index(trade_date=sample_date, fields="ts_code,name")
        if idx is None or idx.empty:
            for offset in range(1,30):
                d = (pd.Timestamp(sample_date)-pd.DateOffset(days=offset)).strftime("%Y%m%d")
                try:
                    idx = pro.dc_index(trade_date=d, fields="ts_code,name")
                    if idx is not None and not idx.empty: sample_date=d; break
                except Exception: pass
                time.sleep(0.3)
    except Exception as e: print(f"[概念映射] 失败: {e}"); return {}
    if idx is None or idx.empty: return {}
    concept_codes = idx["ts_code"].unique().tolist()
    stock_to_concepts = defaultdict(set); t0 = time.time()
    for i, ccode in enumerate(concept_codes):
        try:
            df = pro.dc_member(trade_date=sample_date, ts_code=ccode, fields="ts_code,con_code")
            if df is not None and not df.empty:
                for _, row in df.iterrows(): stock_to_concepts[row["con_code"]].add(ccode)
        except Exception: pass
        if (i+1)%50==0:
            elapsed=time.time()-t0; speed=(i+1)/elapsed if elapsed>0 else 0
            eta=(len(concept_codes)-i-1)/speed/60 if speed>0 else 0
            print(f"    成分进度：{i+1}/{len(concept_codes)}  已映射{len(stock_to_concepts)}只  预计剩余{eta:.1f}分钟")
        time.sleep(0.35)
    return {k: list(v) for k, v in stock_to_concepts.items()}

def calc_concept_heat(date, dc_index_by_date, stock_concepts):
    if not dc_index_by_date or not stock_concepts: return {}
    concept_rank = dc_index_by_date.get(date, {})
    if not concept_rank: return {}
    result = {}
    for stock, concepts in stock_concepts.items():
        ranks = [concept_rank[c] for c in concepts if c in concept_rank]
        if ranks: result[stock] = max(ranks)
    return result

def download_cyq_perf(codes, start, end):
    print(f"[筹码胜率] 下载cyq_perf：{len(codes)} 只股票")
    all_data = []; t0 = time.time()
    for i, code in enumerate(codes):
        try:
            df = pro.cyq_perf(ts_code=code, start_date=start, end_date=end, fields="ts_code,trade_date,winner_rate")
            if df is not None and not df.empty: all_data.append(df)
        except Exception: pass
        if (i+1)%100==0:
            elapsed=time.time()-t0; speed=(i+1)/elapsed if elapsed>0 else 0
            eta=(len(codes)-i-1)/speed/60 if speed>0 else 0
            print(f"    进度：{i+1}/{len(codes)}  预计剩余{eta:.1f}分钟")
        time.sleep(0.15)
    if not all_data: return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    result["winner_rate"] = pd.to_numeric(result["winner_rate"], errors="coerce")
    result = result.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    return result

def load_or_download(start, end, refresh=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"v11_{start.replace('-','')}_{end.replace('-','')}"
    v5k_cache = CACHE_DIR / f"{tag}.pkl"
    tdx_cache = CACHE_DIR / f"pytdx_30min_1430_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if not v5k_cache.exists(): print(f"[错误] 未找到v5k缓存：{v5k_cache}"); sys.exit(1)
    print(f"[数据] 加载v5k缓存：{v5k_cache}")
    with open(v5k_cache, "rb") as f: v5k_data = pickle.load(f)
    all_dates=v5k_data["all_trade_dates"]; bt_dates=v5k_data["bt_trade_dates"]
    fina_data=v5k_data["fina_data"]; val_snaps=v5k_data["val_snaps"]
    basics=v5k_data["basics"]; v5k_stock_daily=v5k_data["stock_daily"]; v5k_hs300=v5k_data["hs300_data"]
    need_codes = sorted(v5k_stock_daily["ts_code"].unique().tolist())
    print(f"  v5k缓存已加载：{len(need_codes)} 只\n")
    raw_cache = CACHE_DIR / f"pytdx_raw30min_1430_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if tdx_cache.exists() and not refresh:
        with open(tdx_cache,"rb") as f: tdx_data=pickle.load(f)
        daily_1430=tdx_data["daily_1430"]; hs300_1430=tdx_data["hs300_1430"]
    else:
        real_start = (pd.Timestamp(start)-pd.tseries.offsets.BDay(80)).strftime("%Y%m%d")
        raw_cache_orig = CACHE_DIR / f"pytdx_raw30min_{start.replace('-','')}_{end.replace('-','')}.pkl"
        if raw_cache.exists() and not refresh:
            with open(raw_cache,"rb") as f: min30=pickle.load(f)["min30"]
        elif raw_cache_orig.exists() and not refresh:
            with open(raw_cache_orig,"rb") as f: min30=pickle.load(f)["min30"]
        else:
            min30 = download_30min_pytdx(need_codes, min_date=real_start)
            with open(raw_cache,"wb") as f: pickle.dump({"min30":min30}, f)
        daily_1430 = build_1430_daily_from_pytdx(min30) if not min30.empty else pd.DataFrame()
        hs300_1430 = download_hs300_1430_pytdx(min_date=real_start)
        with open(tdx_cache,"wb") as f: pickle.dump({"daily_1430":daily_1430,"hs300_1430":hs300_1430}, f)
    if daily_1430.empty:
        stock_daily_final=v5k_stock_daily; hs300_final=v5k_hs300
    else:
        tdx_date_min=daily_1430["trade_date"].min()
        name_map=dict(zip(basics["ts_code"],basics["name"])); ind_map=dict(zip(basics["ts_code"],basics["industry"]))
        daily_1430["name"]=daily_1430["ts_code"].map(name_map).fillna(daily_1430["ts_code"])
        daily_1430["industry"]=daily_1430["ts_code"].map(ind_map).fillna("")
        v5k_early=v5k_stock_daily[v5k_stock_daily["trade_date"]<tdx_date_min]
        stock_daily_final=pd.concat([v5k_early,daily_1430],ignore_index=True).sort_values(["ts_code","trade_date"]).reset_index(drop=True)
        if "name" not in stock_daily_final.columns: stock_daily_final["name"]=stock_daily_final["ts_code"].map(name_map).fillna(stock_daily_final["ts_code"])
        if "industry" not in stock_daily_final.columns: stock_daily_final["industry"]=stock_daily_final["ts_code"].map(ind_map).fillna("")
        if not hs300_1430.empty:
            hs300_early=v5k_hs300[v5k_hs300["trade_date"]<hs300_1430["trade_date"].min()]
            hs300_final=pd.concat([hs300_early,hs300_1430],ignore_index=True).sort_values("trade_date").reset_index(drop=True)
        else: hs300_final=v5k_hs300
    dc_cache = CACHE_DIR / f"dc_concept_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if dc_cache.exists() and not refresh:
        with open(dc_cache,"rb") as f: dc_data=pickle.load(f)
        dc_index_hist=dc_data.get("dc_index",pd.DataFrame()); stock_concepts=dc_data.get("stock_concepts",{})
    else:
        dc_dates=[d for d in bt_dates if d>="20200101"]
        dc_index_hist=download_dc_index_history(dc_dates) if dc_dates else pd.DataFrame()
        stock_concepts=download_concept_stock_mapping(sample_date=end.replace("-",""))
        with open(dc_cache,"wb") as f: pickle.dump({"dc_index":dc_index_hist,"stock_concepts":stock_concepts}, f)
    cyq_cache = CACHE_DIR / f"cyq_perf_{start.replace('-','')}_{end.replace('-','')}.pkl"
    if cyq_cache.exists() and not refresh:
        with open(cyq_cache,"rb") as f: cyq_data=pickle.load(f)
    else:
        cyq_codes=stock_daily_final["ts_code"].unique().tolist()
        cyq_data=download_cyq_perf(cyq_codes, start.replace("-",""), end.replace("-",""))
        with open(cyq_cache,"wb") as f: pickle.dump(cyq_data, f)
    if not isinstance(cyq_data, pd.DataFrame): cyq_data = pd.DataFrame()
    if not cyq_data.empty:
        cyq_slim=cyq_data[["ts_code","trade_date","winner_rate"]].copy()
        stock_daily_final=stock_daily_final.merge(cyq_slim, on=["ts_code","trade_date"], how="left")
    else:
        stock_daily_final["winner_rate"]=np.nan
    return {"all_trade_dates":all_dates,"bt_trade_dates":bt_dates,"hs300_data":hs300_final,
            "stock_daily":stock_daily_final,"fina_data":fina_data,"val_snaps":val_snaps,
            "basics":basics,"dc_index_hist":dc_index_hist,"stock_concepts":stock_concepts}


# ============================================================
# 基本面 / 行业 / 红利（与 v27b 完全相同）
# ============================================================

def _normalize_series(s, clip_low=None, clip_high=None):
    s = s.copy()
    if clip_low is not None: s = s.clip(lower=clip_low)
    if clip_high is not None: s = s.clip(upper=clip_high)
    lo, hi = s.min(), s.max()
    if hi == lo: return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100

def score_fundamentals(fina_data, val_snaps, basics, date):
    result = {}
    if fina_data.empty: return result
    available = fina_data[fina_data["ann_date"] <= date].copy()
    if available.empty: return result
    latest = available.sort_values("end_date").groupby("ts_code").last().reset_index()
    ind_map = dict(zip(basics["ts_code"], basics["industry"]))
    latest["industry"] = latest["ts_code"].map(ind_map).fillna("")
    latest["is_tech"] = latest["industry"].isin(TECH_INDUSTRIES)
    std_mask = ((~latest["is_tech"]) & (latest["roe_dt"]>=MIN_ROE) & (latest["grossprofit_margin"]>=MIN_GROSS_MARGIN) & (latest["or_yoy"]>=MIN_REVENUE_GROWTH))
    tech_mask = ((latest["is_tech"]) & (latest["roe_dt"]>=TECH_MIN_ROE) & (latest["grossprofit_margin"]>=TECH_MIN_GROSS_MARGIN) & (latest["or_yoy"]>=MIN_REVENUE_GROWTH))
    pass_mask = std_mask | tech_mask
    non_fin = ~latest["industry"].isin(FINANCIAL_INDUSTRIES)
    debt_mask = (latest["debt_to_assets"]<MAX_DEBT_RATIO) | (~non_fin)
    pass_mask = pass_mask & debt_mask
    val_codes = set()
    if not val_snaps.empty:
        recent_snaps = val_snaps[val_snaps["trade_date"]<=date]
        if not recent_snaps.empty:
            snap = recent_snaps[recent_snaps["trade_date"]==recent_snaps["trade_date"].max()].copy()
            snap["industry"]=snap["ts_code"].map(ind_map).fillna(""); snap["is_tech"]=snap["industry"].isin(TECH_INDUSTRIES)
            std_ok=snap[~snap["is_tech"]]; std_ok=std_ok[(std_ok["pe_ttm"]>PE_RANGE[0])&(std_ok["pe_ttm"]<PE_RANGE[1])&(std_ok["pb"]>PB_RANGE[0])&(std_ok["pb"]<PB_RANGE[1])&(std_ok["total_mv"]>MIN_MARKET_CAP*10000)]
            tech_ok=snap[snap["is_tech"]]; tech_ok=tech_ok[(tech_ok["pe_ttm"]>TECH_PE_RANGE[0])&(tech_ok["pe_ttm"]<TECH_PE_RANGE[1])&(tech_ok["pb"]>PB_RANGE[0])&(tech_ok["pb"]<PB_RANGE[1])&(tech_ok["total_mv"]>TECH_MIN_MARKET_CAP*10000)]
            val_codes = set(std_ok["ts_code"].tolist()) | set(tech_ok["ts_code"].tolist())
    passed_idx = latest[pass_mask].index
    if len(passed_idx)==0: return result
    sub = latest.loc[passed_idx].copy()
    roe_score=_normalize_series(sub["roe_dt"].fillna(0),0,50)
    rev_score=_normalize_series(sub["or_yoy"].fillna(0),-10,80)
    gm_score=_normalize_series(sub["grossprofit_margin"].fillna(0),10,80)
    fund_scores = FUND_WEIGHT_ROE*roe_score + FUND_WEIGHT_REVENUE*rev_score + FUND_WEIGHT_MARGIN*gm_score
    ocf_score=_normalize_series(sub["ocfps"].fillna(0),-1,10)
    np_score=_normalize_series(sub["netprofit_yoy"].fillna(0),-20,100)
    fund_scores = fund_scores + FUND_BONUS_CASHFLOW*ocf_score + FUND_BONUS_PEG*np_score
    for idx, row in sub.iterrows():
        code=row["ts_code"]; passed=True
        if val_codes and code not in val_codes: passed=False
        result[code]={"passed":passed,"fund_score":float(fund_scores.get(idx,0)),"is_tech":bool(row["is_tech"]),"industry":row["industry"]}
    return result

def calc_industry_heat(indicators, date):
    industry_data = defaultdict(lambda: {"mom5":[],"mom20":[],"vol_ratio":[]})
    for ts_code, idf in indicators.items():
        row = idf[idf["trade_date"]==date]
        if row.empty: continue
        row=row.iloc[0]; mom5=row.get("mom5",np.nan); mom20=row.get("mom20",np.nan)
        vol_r=row.get("vol_ratio",np.nan); ind=row.get("industry","")
        if ind and pd.notna(mom20):
            if pd.notna(mom5): industry_data[ind]["mom5"].append(mom5)
            industry_data[ind]["mom20"].append(mom20)
            if pd.notna(vol_r): industry_data[ind]["vol_ratio"].append(vol_r)
    raw_heat = {}
    for ind, d in industry_data.items():
        if len(d["mom20"])<3: continue
        m5=np.mean(d["mom5"]) if d["mom5"] else 0; m20=np.mean(d["mom20"])
        heat=0.5*m5+0.5*m20
        if d["vol_ratio"]:
            if np.mean(d["vol_ratio"])>1.2: heat*=1.1
        raw_heat[ind]=heat
    if not raw_heat: return {}
    sorted_inds=sorted(raw_heat.items(), key=lambda x: x[1]); n=len(sorted_inds)
    return {ind:(rank/max(n-1,1))*100 for rank,(ind,_) in enumerate(sorted_inds)}

def screen_dynamic_dividends(val_snaps, indicators, basics, date):
    dividend_codes = set(DIVIDEND_SEED_POOL.keys())
    if val_snaps.empty: return dividend_codes
    recent_snaps = val_snaps[val_snaps["trade_date"]<=date]
    if recent_snaps.empty: return dividend_codes
    snap = recent_snaps[recent_snaps["trade_date"]==recent_snaps["trade_date"].max()].copy()
    snap = snap[(snap["total_mv"]>200*10000)&(snap["dv_ratio"].fillna(0)>2.0)]
    if snap.empty: return dividend_codes
    vol_dict={}
    for code in snap["ts_code"].tolist():
        if code in indicators:
            idf=indicators[code]; row=idf[idf["trade_date"]<=date].tail(1)
            if not row.empty and "volatility20" in row.columns: vol_dict[code]=float(row.iloc[0]["volatility20"])
    snap["volatility"]=snap["ts_code"].map(vol_dict)
    snap["vol_sort"]=snap["volatility"].fillna(999)
    snap=snap.sort_values(["dv_ratio","vol_sort","total_mv"],ascending=[False,True,False])
    return set(snap.head(30)["ts_code"].tolist()) | dividend_codes


# ============================================================
# 技术指标（新增 bias_ma5）
# ============================================================

def calc_indicators(df):
    df = df.sort_values("trade_date").copy()
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    df["ma5"]  = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["high20"] = c.rolling(BREAKOUT_DAYS).max()
    is_breakout = (c >= df["high20"].shift(1) * 0.99).astype(int)
    df["recent_breakout"] = is_breakout.rolling(PULLBACK_WINDOW).max()
    df["not_at_high"] = (c < c.rolling(PULLBACK_WINDOW).max()).astype(int)
    df["vol_ma5"] = df["vol"].rolling(5).mean()
    df["vol_ma20"] = df["vol"].rolling(20).mean()
    df["vol_ratio"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)
    df["mom5"] = (c / c.shift(5) - 1) * 100
    df["mom20"] = (c / c.shift(20) - 1) * 100
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / c
    df["volatility20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ★ v35a 新增：MA5 乖离率
    df["bias_ma5"] = (c / df["ma5"] - 1) * 100

    below_ma5 = (c < df["ma5"]).astype(int)
    streaks_5=[]; cnt=0
    for v in below_ma5: cnt=cnt+1 if v else 0; streaks_5.append(cnt)
    df["below_ma5_streak"]=streaks_5
    below_ma10 = (c < df["ma10"]).astype(int)
    streaks_10=[]; cnt=0
    for v in below_ma10: cnt=cnt+1 if v else 0; streaks_10.append(cnt)
    df["below_ma10_streak"]=streaks_10
    below_ma20 = (c < df["ma20"]).astype(int)
    streaks_20=[]; cnt=0
    for v in below_ma20: cnt=cnt+1 if v else 0; streaks_20.append(cnt)
    df["below_ma20_streak"]=streaks_20
    if "winner_rate" in df.columns: df["wr_prev"]=df["winner_rate"].shift(1)
    else: df["wr_prev"]=np.nan
    return df


# ============================================================
# 买入信号（★ 新增第6条件：乖离率过滤）
# ============================================================

def check_buy_v11(row):
    close = row["close"]
    ma20 = row.get("ma20", np.nan)
    ma60 = row.get("ma60", np.nan)
    vol_ratio = row.get("vol_ratio", np.nan)
    mom = row.get("mom20", np.nan)
    recent_bk = row.get("recent_breakout", 0)
    not_at_high = row.get("not_at_high", 0)
    atr_r = row.get("atr_ratio", np.nan)
    bias = row.get("bias_ma5", np.nan)       # ★ v35a

    if any(pd.isna(x) for x in [ma20, ma60, vol_ratio, mom, atr_r]):
        return False, "", 0

    # 1. MA多头
    if not (close > ma20 > ma60):
        return False, "", 0
    # 2. 回踩确认
    if recent_bk < 1:
        return False, "", 0
    # 3. 放量
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_RATIO:
        return False, "", 0
    # 4. 动量区间
    if mom < MIN_MOM20 or mom > MAX_MOM20:
        return False, "", 0
    # 5. ATR过滤
    if atr_r > ATR_THRESHOLD:
        return False, "", 0
    # 6. MA5 乖离率过滤 (v35a)
    if pd.notna(bias) and bias > BIAS_MA5_MAX:
        return False, "", 0
    # 7. ★ v35b 新增：短期均线多头 MA5 > MA10
    ma5 = row.get("ma5", np.nan)
    ma10 = row.get("ma10", np.nan)
    if pd.notna(ma5) and pd.notna(ma10) and ma5 <= ma10:
        return False, "", 0

    # ── 技术面评分 ──
    if mom <= MOM_PREFERRED:
        mom_score = (mom - MIN_MOM20) / (MOM_PREFERRED - MIN_MOM20) * 100
    else:
        mom_score = max(0, 100 - (mom - MOM_PREFERRED) / (MAX_MOM20 - MOM_PREFERRED) * 60)
    if vol_ratio <= 2.5:
        vol_score = min(vol_ratio / VOLUME_RATIO * 50, 100)
    else:
        vol_score = max(0, 100 - (vol_ratio - 2.5) / 2.5 * 80)
    pullback_bonus = 15 if not_at_high else 0
    wr = row.get("wr_prev", np.nan)
    wr_bonus = 10 if (pd.notna(wr) and wr > 50) else 0
    tech_score = mom_score * 0.5 + vol_score * 0.3 + pullback_bonus + wr_bonus
    wr_tag = f"|胜率{wr:.0f}%" if pd.notna(wr) else ""
    bias_tag = f"|乖离{bias:.1f}%" if pd.notna(bias) else ""
    ma_tag = "|MA5>10"
    reason = f"回踩确认|放量{vol_ratio:.1f}x|动量{mom:.1f}%|ATR{atr_r*100:.1f}%{bias_tag}{ma_tag}{wr_tag}"
    return True, reason, tech_score


# ============================================================
# 卖出信号（与 v27b 完全相同）
# ============================================================

def check_sell_v11(pos, row, market_ok, regime):
    close=float(row["close"]); cost=pos["cost"]; peak=pos.get("peak_price",cost)
    pnl=(close-cost)/cost; drawdown_from_peak=(close-peak)/peak if peak>0 else 0
    partial_done_1=pos.get("partial_1_done",False); partial_done_2=pos.get("partial_2_done",False)
    if regime=="BEAR":
        eff_stop_loss=BEAR_STOP_LOSS_PCT; eff_trail_stop=BEAR_TRAIL_STOP_PCT
        eff_trail_ma_key="below_ma5_streak"; eff_trail_ma_days=BEAR_TRAIL_MA_DAYS; eff_trail_ma_label="MA5"
    elif regime=="NEUTRAL":
        eff_stop_loss=(STOP_LOSS_PCT+BEAR_STOP_LOSS_PCT)/2; eff_trail_stop=(TRAIL_STOP_PCT+BEAR_TRAIL_STOP_PCT)/2
        eff_trail_ma_key="below_ma10_streak"; eff_trail_ma_days=TRAIL_BELOW_MA10_DAYS; eff_trail_ma_label="MA10"
    else:
        eff_stop_loss=0.15; eff_trail_stop=0.12
        eff_trail_ma_key="below_ma10_streak"; eff_trail_ma_days=3; eff_trail_ma_label="MA10"
    if pnl <= -eff_stop_loss: return True, f"止损{pnl*100:.1f}%", 1.0
    if peak>cost*1.05 and drawdown_from_peak<=-eff_trail_stop: return True, f"回撤止盈(峰值回落{drawdown_from_peak*100:.1f}%)", 1.0
    if pnl>=PARTIAL_PROFIT_1 and not partial_done_1: return True, f"分批止盈① (+{pnl*100:.1f}%)", 0.5
    if pnl>=PARTIAL_PROFIT_2 and partial_done_1 and not partial_done_2: return True, f"分批止盈② (+{pnl*100:.1f}%)", 0.5
    if pnl>0.03 and row.get(eff_trail_ma_key,0)>=eff_trail_ma_days: return True, f"趋势止盈(破{eff_trail_ma_label}连续{row[eff_trail_ma_key]}天)", 1.0
    if row.get("below_ma20_streak",0)>=TRAIL_BELOW_MA20_DAYS: return True, f"破MA20连续{row['below_ma20_streak']}天", 1.0
    return False, "", 0


# ============================================================
# 市场环境
# ============================================================

def get_regime(hs300_data, date):
    hs = hs300_data[hs300_data["trade_date"]<=date].sort_values("trade_date")
    if len(hs)<60: return "NEUTRAL"
    c = hs["close"].astype(float); cur=float(c.iloc[-1]); ma20=c.tail(20).mean(); ma60=c.tail(60).mean()
    if cur>ma60 and ma20>ma60: return "BULL"
    if cur>ma60*0.97: return "NEUTRAL"
    return "BEAR"


# ============================================================
# 回测引擎（与 v27b 完全相同，不修改）
# ============================================================

def run_backtest(data):
    all_dates=data["all_trade_dates"]; bt_dates=data["bt_trade_dates"]
    hs300_data=data["hs300_data"]; stock_daily=data["stock_daily"]
    fina_data=data["fina_data"]; val_snaps=data["val_snaps"]; basics=data["basics"]
    dc_index_hist=data.get("dc_index_hist",pd.DataFrame()); stock_concepts=data.get("stock_concepts",{})
    dc_index_by_date={}
    if not dc_index_hist.empty:
        for date_key, grp in dc_index_hist.groupby("trade_date"):
            grp=grp.copy(); grp["pct_rank"]=grp["pct_change"].rank(pct=True)*100
            dc_index_by_date[date_key]=dict(zip(grp["ts_code"],grp["pct_rank"]))
    print("[回测] 预计算技术指标...")
    indicators={}; all_codes=stock_daily["ts_code"].unique()
    for j, ts_code in enumerate(all_codes):
        sdf=stock_daily[stock_daily["ts_code"]==ts_code]; indicators[ts_code]=calc_indicators(sdf)
        if (j+1)%200==0: print(f"  指标：{j+1}/{len(all_codes)}")
    print(f"  完成：{len(indicators)} 只")
    ind_map=dict(zip(basics["ts_code"],basics["industry"]))
    for ts_code in indicators:
        idf=indicators[ts_code]
        if "industry" not in idf.columns: idf["industry"]=ind_map.get(ts_code,"")
    capital=TOTAL_CAPITAL; positions={}; nav_history=[]; trade_log=[]
    days_in=0; days_out=0; n_days=len(bt_dates)
    fund_cache={"date":"","scores":{}}; FUND_REFRESH_INTERVAL=50
    dividend_cache={"date":"","codes":set(DIVIDEND_SEED_POOL.keys())}; DIVIDEND_REFRESH=50
    confirmed_regime="NEUTRAL"; pending_regime="NEUTRAL"; pending_count=0
    print(f"\n[回测] {bt_dates[0]}~{bt_dates[-1]}，{n_days} 天")
    print(f"  卖出：止损{STOP_LOSS_PCT*100}% | 回撤止盈{TRAIL_STOP_PCT*100}% | 分批止盈{PARTIAL_PROFIT_1*100}%/{PARTIAL_PROFIT_2*100}%")
    print(f"  买入：回踩确认 | 动量{MIN_MOM20}~{MAX_MOM20}%(优选<{MOM_PREFERRED}%) | ATR<{ATR_THRESHOLD}% | 乖离<{BIAS_MA5_MAX}% | ★MA5>MA10")
    print(f"  仓位：BULL={REGIME_MAX_POSITION['BULL']*100}% | NEUTRAL={REGIME_MAX_POSITION['NEUTRAL']*100}% | BEAR={REGIME_MAX_POSITION['BEAR']*100}%")
    print(f"  牛熊切换确认：连续{REGIME_CONFIRM_DAYS}天\n")
    for i, date in enumerate(bt_dates):
        raw_regime=get_regime(hs300_data, date)
        if raw_regime==confirmed_regime: pending_regime=confirmed_regime; pending_count=0
        elif raw_regime==pending_regime:
            pending_count+=1
            if pending_count>=REGIME_CONFIRM_DAYS: confirmed_regime=pending_regime; pending_count=0
        else: pending_regime=raw_regime; pending_count=1
        regime=confirmed_regime; market_ok=regime!="BEAR"
        if i%FUND_REFRESH_INTERVAL==0 or fund_cache["date"]=="":
            fund_scores=score_fundamentals(fina_data,val_snaps,basics,date); fund_cache={"date":date,"scores":fund_scores}
        if i%DIVIDEND_REFRESH==0 or dividend_cache["date"]=="":
            dividend_codes=screen_dynamic_dividends(val_snaps,indicators,basics,date); dividend_cache={"date":date,"codes":dividend_codes}
        else: dividend_codes=dividend_cache["codes"]
        for code, pos in positions.items():
            idf=indicators.get(code)
            if idf is not None:
                row=idf[idf["trade_date"]==date]
                if not row.empty:
                    cur_price=float(row.iloc[0]["close"]); pos["current_price"]=cur_price; pos["last_update"]=date
                    if cur_price>pos.get("peak_price",0): pos["peak_price"]=cur_price
                else:
                    last=pos.get("last_update",pos.get("buy_date",date))
                    li=all_dates.index(last) if last in all_dates else 0; ci=all_dates.index(date) if date in all_dates else 0
                    pos["halt_days"]=ci-li
        port_val=sum(p["shares"]*p["current_price"] for p in positions.values())
        total_val=capital+port_val; nav=total_val/TOTAL_CAPITAL
        if positions: days_in+=1
        else: days_out+=1
        # 卖出
        for code, pos in list(positions.items()):
            if pos.get("halt_days",0)>=HALT_SELL_DAYS:
                sell_p=pos["current_price"]*(1-SLIPPAGE); sell_amt=pos["shares"]*sell_p
                fee=sell_amt*(STOCK_COMMISSION+STAMP_DUTY); pnl=(sell_p-pos["cost"])/pos["cost"]
                capital+=sell_amt-fee; hold=len([d for d in bt_dates if pos["buy_date"]<=d<=date])
                trade_log.append({"date":date,"action":f"卖出(停牌{pos['halt_days']}天)","code":code,"name":pos["name"],"price":round(sell_p,3),"shares":pos["shares"],"pnl_pct":round(pnl*100,1),"hold_days":hold,"fee":round(fee,2)})
                del positions[code]; continue
            idf=indicators.get(code)
            if idf is None: continue
            row=idf[idf["trade_date"]==date]
            if row.empty: continue
            row=row.iloc[0]
            should_sell,reason,partial_ratio=check_sell_v11(pos,row,market_ok,regime)
            if should_sell:
                sell_shares=pos["shares"]
                if partial_ratio<1.0:
                    sell_shares=max(int(pos["shares"]*partial_ratio/100)*100,100)
                    if sell_shares>=pos["shares"]: sell_shares=pos["shares"]
                sell_p=pos["current_price"]*(1-SLIPPAGE); sell_amt=sell_shares*sell_p
                fee=sell_amt*(STOCK_COMMISSION+STAMP_DUTY); pnl=(sell_p-pos["cost"])/pos["cost"]
                capital+=sell_amt-fee; hold=len([d for d in bt_dates if pos["buy_date"]<=d<=date])
                trade_log.append({"date":date,"action":f"卖出({reason})","code":code,"name":pos["name"],"price":round(sell_p,3),"shares":sell_shares,"pnl_pct":round(pnl*100,1),"hold_days":hold,"fee":round(fee,2)})
                if sell_shares>=pos["shares"]: del positions[code]
                else:
                    pos["shares"]-=sell_shares
                    if "分批止盈①" in reason: pos["partial_1_done"]=True
                    elif "分批止盈②" in reason: pos["partial_2_done"]=True
        # 买入
        max_exposure=REGIME_MAX_POSITION.get(regime,0.1)
        total_now=capital+sum(p["shares"]*p["current_price"] for p in positions.values())
        current_exposure=sum(p["shares"]*p["current_price"] for p in positions.values())/total_now if total_now>0 else 0
        if regime=="BULL" and current_exposure<max_exposure and len(positions)<MAX_POSITIONS:
            allowed_buy_amount=total_now*max_exposure-sum(p["shares"]*p["current_price"] for p in positions.values())
            allowed_buy_amount=max(allowed_buy_amount,0)
            fund_scores=fund_cache["scores"]
            scan_codes=[c for c,s in fund_scores.items() if s["passed"] and c in indicators and c not in positions]
            industry_heat=calc_industry_heat(indicators,date)
            concept_heat_map=calc_concept_heat(date,dc_index_by_date,stock_concepts)
            candidates=[]
            for ts_code in scan_codes:
                idf=indicators[ts_code]; row=idf[idf["trade_date"]==date]
                if row.empty: continue
                row=row.iloc[0]
                should_buy,reason,tech_score=check_buy_v11(row)
                if not should_buy: continue
                fs=fund_scores.get(ts_code,{}); fund_score=fs.get("fund_score",0)
                ind=row.get("industry",""); ind_heat=industry_heat.get(ind,50) if industry_heat else 50
                if concept_heat_map:
                    cpt_heat=concept_heat_map.get(ts_code,50); heat=0.5*ind_heat+0.5*cpt_heat
                else: heat=ind_heat
                tech_with_heat=tech_score*0.7+heat*0.3
                if ind in LOW_ELASTIC_INDUSTRIES: w_tech=LOW_ELASTIC_TECH_W; w_fund=LOW_ELASTIC_FUND_W
                else: w_tech=SCORE_WEIGHT_TECH; w_fund=SCORE_WEIGHT_FUND
                final_score=w_tech*tech_with_heat+w_fund*fund_score
                candidates.append({"ts_code":ts_code,"name":row.get("name",ts_code),"close":float(row["close"]),"mom20":float(row["mom20"]) if pd.notna(row.get("mom20")) else 0,"industry":ind,"heat":round(heat,1),"fund_score":round(fund_score,1),"tech_score":round(tech_score,1),"final_score":round(final_score,2),"reason":reason})
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            slots=MAX_POSITIONS-len(positions)
            total_score=sum(c["final_score"] for c in candidates[:slots]) if candidates else 1
            if total_score<=0: total_score=1
            for c in candidates[:slots]:
                score_ratio=c["final_score"]/total_score
                per_pos=min(allowed_buy_amount*score_ratio, total_now*0.25)
                if per_pos<5000: continue
                price=c["close"]*(1+SLIPPAGE); shares=int(per_pos/price/100)*100
                if shares<100: continue
                cost=shares*price; fee=cost*STOCK_COMMISSION
                if cost+fee>capital:
                    shares=int(capital/(price*(1+STOCK_COMMISSION))/100)*100
                    if shares<100: continue
                    cost=shares*price; fee=cost*STOCK_COMMISSION
                capital-=cost+fee; allowed_buy_amount-=cost
                positions[c["ts_code"]]={"shares":shares,"cost":price,"current_price":price,"peak_price":price,"name":c["name"],"buy_date":date,"last_update":date,"halt_days":0,"partial_1_done":False,"partial_2_done":False}
                trade_log.append({"date":date,"action":"买入","code":c["ts_code"],"name":c["name"],"price":round(price,3),"shares":shares,"pnl_pct":0,"hold_days":0,"fee":round(fee,2),"reason":f"{c['reason']}|行业:{c['industry']}|热度:{c['heat']}|基本面:{c['fund_score']}|综合:{c['final_score']}"})
        port_val=sum(p["shares"]*p["current_price"] for p in positions.values())
        total_val=capital+port_val; nav=total_val/TOTAL_CAPITAL
        nav_history.append({"date":date,"nav":round(nav,4),"capital":round(capital,2),"positions":len(positions),"regime":regime})
        if (i+1)%50==0:
            icon={"BULL":"🟢","NEUTRAL":"🟡","BEAR":"🔴"}[regime]
            names=",".join(p["name"][:4] for p in list(positions.values())[:3])
            ps=f"{len(positions)}只({names})" if positions else "空仓"
            n_fund=len([s for s in fund_cache["scores"].values() if s["passed"]])
            raw_tag=f"(raw:{raw_regime})" if raw_regime!=regime else ""
            print(f"  [{i+1}/{n_days}] {date}  净值:{nav:.4f}  {ps}  好公司:{n_fund}  仓位:{current_exposure*100:.0f}%/{max_exposure*100:.0f}%  资金:¥{capital:,.0f}  {icon}{regime}{raw_tag}")
    nav_df=pd.DataFrame(nav_history).set_index("date"); trade_df=pd.DataFrame(trade_log)
    nav_df.attrs["days_in"]=days_in; nav_df.attrs["days_out"]=days_out
    return nav_df, trade_df

def calc_perf(nav_df):
    nav=nav_df["nav"]; rets=nav.pct_change().dropna(); n=len(nav); ny=n/252
    tr=nav.iloc[-1]/nav.iloc[0]-1; ar=(1+tr)**(1/ny)-1 if ny>0 else 0
    ex=rets-0.02/252; sh=ex.mean()/ex.std()*np.sqrt(252) if ex.std()>0 else 0
    cm=nav.cummax(); dd=(nav-cm)/cm; mdd=dd.min(); dde=dd.idxmin(); dds=nav[:dde].idxmax()
    wr=(rets>0).sum()/len(rets) if len(rets)>0 else 0
    cal=ar/abs(mdd) if mdd!=0 else 0
    ns=nav.copy(); ns.index=pd.to_datetime(ns.index, format="%Y%m%d")
    mo=ns.resample("ME").last().pct_change().dropna()
    di=nav_df.attrs.get("days_in",0); do=nav_df.attrs.get("days_out",0)
    return {"回测区间":f"{nav_df.index[0]}~{nav_df.index[-1]}","交易日":n,
        "总收益":f"{tr*100:+.2f}%","年化":f"{ar*100:+.2f}%","夏普":f"{sh:.3f}",
        "最大回撤":f"{mdd*100:.2f}%","回撤区间":f"{dds}~{dde}","Calmar":f"{cal:.3f}",
        "日胜率":f"{wr*100:.1f}%",
        "最佳月":f"{mo.max()*100:+.1f}%" if len(mo)>0 else "-",
        "最差月":f"{mo.min()*100:+.1f}%" if len(mo)>0 else "-",
        "持仓":f"{di}天({di/n*100:.0f}%)","空仓":f"{do}天({do/n*100:.0f}%)",
        "期末净值":f"{nav.iloc[-1]:.4f}","期末市值":f"¥{nav.iloc[-1]*TOTAL_CAPITAL:,.0f}"}

def print_report(perf, trade_df, nav_df):
    print("\n"+"="*70)
    print("📊 v35b 绩效报告（v35a + MA5>MA10）")
    print("="*70)
    for k,v in perf.items(): print(f"  {k:<10}: {v}")
    print("="*70)
    if not trade_df.empty:
        buys=trade_df[trade_df["action"]=="买入"]; sells=trade_df[trade_df["action"].str.startswith("卖出")]
        print(f"\n  交易：{len(trade_df)}次（买{len(buys)}，卖{len(sells)}）")
        if len(sells)>0:
            print(f"  卖出均盈亏：{sells['pnl_pct'].mean():+.1f}%")
            w=sells[sells["pnl_pct"]>0]; l=sells[sells["pnl_pct"]<=0]
            if len(w): print(f"  盈利：{len(w)}次({len(w)/len(sells)*100:.0f}%)，均+{w['pnl_pct'].mean():.1f}%")
            if len(l): print(f"  亏损：{len(l)}次({len(l)/len(sells)*100:.0f}%)，均{l['pnl_pct'].mean():.1f}%")
            print(f"  均持有：{sells['hold_days'].mean():.0f}天")
            print(f"\n  卖出原因分布：")
            reason_cats={"止损":0,"回撤止盈":0,"分批止盈":0,"趋势止盈(MA10)":0,"破MA20":0,"大盘破MA60":0,"市场切换":0,"停牌":0,"其他":0}
            for act in sells["action"]:
                if "止损" in act and "止盈" not in act: reason_cats["止损"]+=1
                elif "回撤止盈" in act: reason_cats["回撤止盈"]+=1
                elif "分批止盈" in act: reason_cats["分批止盈"]+=1
                elif "MA10" in act or "趋势止盈" in act: reason_cats["趋势止盈(MA10)"]+=1
                elif "MA20" in act: reason_cats["破MA20"]+=1
                elif "MA60" in act: reason_cats["大盘破MA60"]+=1
                elif "切换" in act: reason_cats["市场切换"]+=1
                elif "停牌" in act: reason_cats["停牌"]+=1
                else: reason_cats["其他"]+=1
            for r,c in sorted(reason_cats.items(), key=lambda x:-x[1]):
                if c>0: print(f"    {r}：{c}次（{c/len(sells)*100:.0f}%）")
        print(f"\n  手续费：¥{trade_df['fee'].sum():,.0f}（{trade_df['fee'].sum()/TOTAL_CAPITAL*100:.2f}%）")
        if len(buys)>0:
            print(f"\n  最常买入Top10：")
            for name,cnt in buys["name"].value_counts().head(10).items(): print(f"    {name}：{cnt}次")
            if "reason" in buys.columns:
                industries=buys["reason"].str.extract(r"行业:(\w+)")[0].value_counts()
                print(f"\n  买入行业分布Top5：")
                for ind,cnt in industries.head(5).items(): print(f"    {ind}：{cnt}次")
    ns=nav_df["nav"].copy(); ns.index=pd.to_datetime(ns.index, format="%Y%m%d")
    yr=ns.resample("YE").last()
    print(f"\n  年度："); prev=1.0
    for d,v in yr.items(): print(f"    {d.year}：{(v/prev-1)*100:+.1f}%（净值{v:.4f}）"); prev=v
    for r in ["BULL","NEUTRAL","BEAR"]:
        c=(nav_df["regime"]==r).sum(); print(f"  {r}: {c}天（{c/len(nav_df)*100:.0f}%）")
    print()

def save_results(nav_df, trade_df):
    d = CACHE_DIR / "results_v35b"
    d.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(d/"nav.csv")
    if not trade_df.empty: trade_df.to_csv(d/"trades.csv", index=False)
    print(f"  已保存 → {d}/")

def main():
    parser = argparse.ArgumentParser(description="v35b 回测")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(f"\n{'='*75}")
    print(f"  v35b：v35a + 短期均线多头（MA5 > MA10）")
    print(f"{'='*75}")
    print(f"  区间：{args.start}~{args.end}")
    print(f"  ── 累计变更（vs v27b）──")
    print(f"    条件6：MA5乖离率 < {BIAS_MA5_MAX}%（来自v35a）")
    print(f"    ★ 条件7：MA5 > MA10（v35b新增）")
    print(f"  ── 买入（其余同v27b）──")
    print(f"    回踩确认{PULLBACK_WINDOW}日 | 动量{MIN_MOM20}~{MAX_MOM20}%(优选<{MOM_PREFERRED}%)")
    print(f"    放量>{VOLUME_RATIO}x | ATR<{ATR_THRESHOLD*100:.0f}%")
    print(f"{'='*75}\n")
    data = load_or_download(args.start, args.end, refresh=args.refresh)
    nav_df, trade_df = run_backtest(data)
    perf = calc_perf(nav_df)
    print_report(perf, trade_df, nav_df)
    save_results(nav_df, trade_df)

if __name__ == "__main__":
    main()
