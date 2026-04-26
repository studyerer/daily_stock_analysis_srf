#!/usr/bin/env python3
"""
每日交易信号生成器 v27b（14:30盘中执行版）
============================================
策略版本：v27b = v18c + v20a低弹性行业降权 + 放量>2.5x扣分

策略进化链：
  v18c（+概念热度+筹码胜率）→ v20a（+低弹性行业降权）→ v27b（+放量封顶）

相比v18c信号系统的两处改动：
  ① v20a：低弹性行业（白酒/啤酒/乳制品/环境保护/路桥/园区开发）
           基本面权重40%→20%，技术面权重60%→80%
  ② v27b：放量>2.5x时vol_score急剧扣分（极端放量=见顶信号）

回测绩效（v27b, 2018-2026）：
  夏普0.707 | 年化15.06% | 总收益+204% | 回撤-30.5% | 胜率54%

执行逻辑：
  14:30 AKShare获取全市场实时行情（准收盘价）
  + Tushare历史日线（截至昨天）
  → 拼接构造今日伪日线 → calc_indicators() → 生成信号 → 飞书推送
  → 你在14:30~15:00之间在券商APP下单

部署：crontab -e → 30 14 * * 1-5 cd /root/v27b_signal && python3 daily_signal.py
依赖：pip install tushare pandas numpy requests akshare --break-system-packages
"""

import os, sys, json, time, traceback
import numpy as np
import pandas as pd
import tushare as ts
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

TUSHARE_TOKEN = "948c885edff1169ef76e26ee6a540cebdc73c972b2595f9867825cf9"
FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/04d0ca06-e27a-4332-9c89-a577ef818cfa"

WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = WORK_DIR / "signal_state.json"
LOG_FILE = WORK_DIR / "signal.log"

# ── 策略参数（严格对齐 v27b 回测代码，一个数字都不改）──────────

MAX_POSITIONS    = 5
TOTAL_CAPITAL    = 100_000

# 卖出参数
STOP_LOSS_PCT        = 0.12
TRAIL_STOP_PCT       = 0.10
PARTIAL_PROFIT_1     = 0.20
PARTIAL_PROFIT_2     = 0.40
TRAIL_BELOW_MA10_DAYS = 2
TRAIL_BELOW_MA20_DAYS = 3
BEAR_STOP_LOSS_PCT   = 0.05
BEAR_TRAIL_STOP_PCT  = 0.05
BEAR_TRAIL_MA_DAYS   = 1

# ④ v5k: 回踩确认窗口7天
PULLBACK_WINDOW      = 7

# 买入参数
MIN_MOM20        = 5.0
MAX_MOM20        = 40.0
MOM_PREFERRED    = 25.0
VOLUME_RATIO     = 1.2
BREAKOUT_DAYS    = 20
ATR_THRESHOLD    = 0.07

# 基本面参数
MIN_ROE              = 8.0
MIN_GROSS_MARGIN     = 20.0
MAX_DEBT_RATIO       = 70.0
PE_RANGE             = (0, 80)
PB_RANGE             = (0, 15)
MIN_REVENUE_GROWTH   = 5.0
MIN_MARKET_CAP       = 100
TECH_PE_RANGE        = (0, 200)
TECH_MIN_ROE         = 3.0
TECH_MIN_GROSS_MARGIN = 15.0
TECH_MIN_MARKET_CAP  = 50

# 评分权重
FUND_WEIGHT_ROE      = 0.40
FUND_WEIGHT_REVENUE  = 0.30
FUND_WEIGHT_MARGIN   = 0.30
FUND_BONUS_CASHFLOW  = 0.10
FUND_BONUS_PEG       = 0.05
SCORE_WEIGHT_TECH    = 0.60
SCORE_WEIGHT_FUND    = 0.40

# v20a：低弹性行业基本面天然高但趋势弱→降低基本面权重
LOW_ELASTIC_INDUSTRIES = {"白酒", "啤酒", "乳制品", "环境保护", "路桥", "园区开发"}
LOW_ELASTIC_TECH_W   = 0.80   # 低弹性行业：技术面80%
LOW_ELASTIC_FUND_W   = 0.20   # 低弹性行业：基本面20%

# Regime
REGIME_CONFIRM_DAYS  = 5
REGIME_MAX_POSITION  = {"BULL": 1.00, "NEUTRAL": 0.10, "BEAR": 0.00}

# 行业分类
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

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

def log(msg):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════
# 状态管理
# ══════════════════════════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "confirmed_regime": "NEUTRAL",
        "pending_regime": "NEUTRAL",
        "pending_count": 0,
        "positions": {},
        "last_run_date": "",
        "fund_scores": {},
        "fund_date": "",
    }

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# 代码格式转换（Tushare ↔ AKShare）
# ══════════════════════════════════════════════════════════════

def ts_to_ak(ts_code):
    """600519.SH → 600519"""
    return ts_code.split(".")[0]

def ak_to_ts(ak_code):
    """600519 → 600519.SH / 000001 → 000001.SZ"""
    if ak_code.startswith("6"):
        return f"{ak_code}.SH"
    else:
        return f"{ak_code}.SZ"


# ══════════════════════════════════════════════════════════════
# 实时行情（三级数据源自动切换：AKShare → pytdx → 新浪）
# ══════════════════════════════════════════════════════════════

# pytdx通达信公共行情服务器列表
TDX_SERVERS = [
    ("119.147.212.81", 7709),
    ("114.80.63.12", 7709),
    ("218.75.126.9", 7709),
    ("124.74.236.94", 7721),
    ("221.231.141.60", 7709),
    ("59.173.18.140", 7709),
]


def _fetch_via_akshare(need_codes=None):
    """数据源1：AKShare（东方财富）— 一次获取全市场"""
    import akshare as ak
    log("  [源1] AKShare stock_zh_a_spot_em...")
    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        raise ValueError("AKShare返回空数据")

    result = pd.DataFrame()
    result["ts_code"] = df["代码"].apply(ak_to_ts)
    result["open"] = pd.to_numeric(df["今开"], errors="coerce")
    result["high"] = pd.to_numeric(df["最高"], errors="coerce")
    result["low"] = pd.to_numeric(df["最低"], errors="coerce")
    result["close"] = pd.to_numeric(df["最新价"], errors="coerce")
    result["vol"] = pd.to_numeric(df["成交量"], errors="coerce")  # 单位：手
    result["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
    result = result[result["close"] > 0].reset_index(drop=True)

    if need_codes:
        result = result[result["ts_code"].isin(need_codes)]
    log(f"  [源1] AKShare成功：{len(result)} 只")
    return result, "AKShare"


def _fetch_via_pytdx(need_codes):
    """数据源2：pytdx（通达信直连）— 按需获取指定股票"""
    from pytdx.hq import TdxHq_API
    log(f"  [源2] pytdx直连通达信，获取 {len(need_codes)} 只...")

    api = TdxHq_API()
    connected = False
    for host, port in TDX_SERVERS:
        try:
            api.connect(host, port)
            connected = True
            break
        except Exception:
            continue
    if not connected:
        raise ConnectionError("无法连接任何通达信服务器")

    all_rows = []
    # pytdx每次最多查80只，分批查询
    code_list = list(need_codes)
    for batch_start in range(0, len(code_list), 80):
        batch = code_list[batch_start:batch_start + 80]
        query = []
        for ts_code in batch:
            code_num = ts_code.split(".")[0]
            market = 1 if ts_code.endswith(".SH") else 0
            query.append((market, code_num))
        try:
            quotes = api.get_security_quotes(query)
            if quotes:
                for q in quotes:
                    ts_code = f"{q['code']}.{'SH' if q['market'] == 1 else 'SZ'}"
                    if q["price"] > 0:
                        all_rows.append({
                            "ts_code": ts_code,
                            "open": float(q["open"]),
                            "high": float(q["high"]),
                            "low": float(q["low"]),
                            "close": float(q["price"]),   # 现价
                            "vol": float(q["vol"]) / 100,  # pytdx vol单位是股，转手
                            "amount": float(q["amount"]),
                        })
        except Exception:
            pass

    api.disconnect()

    if not all_rows:
        raise ValueError("pytdx未获取到有效数据")

    result = pd.DataFrame(all_rows)
    log(f"  [源2] pytdx成功：{len(result)} 只")
    return result, "pytdx"


def _fetch_via_sina(need_codes):
    """数据源3：新浪财经HTTP接口 — 按需获取指定股票"""
    log(f"  [源3] 新浪财经HTTP接口，获取 {len(need_codes)} 只...")
    all_rows = []

    code_list = list(need_codes)
    # 新浪接口支持批量查询，每次最多约100只
    for batch_start in range(0, len(code_list), 100):
        batch = code_list[batch_start:batch_start + 100]
        # 转换为新浪代码格式：sh600519, sz000001
        sina_codes = []
        for ts_code in batch:
            code_num = ts_code.split(".")[0]
            prefix = "sh" if ts_code.endswith(".SH") else "sz"
            sina_codes.append(f"{prefix}{code_num}")

        url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
        try:
            resp = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"},
                                timeout=10)
            if resp.status_code != 200:
                continue
            for line in resp.text.strip().split("\n"):
                if "=" not in line or '""' in line:
                    continue
                # 解析：var hq_str_sh600519="贵州茅台,开盘价,...";
                var_part, data_part = line.split("=", 1)
                sina_code = var_part.split("_")[-1]
                fields = data_part.strip('";').split(",")
                if len(fields) < 32 or float(fields[3]) <= 0:
                    continue
                # 新浪字段：0名称,1今开,2昨收,3现价,4最高,5最低,...,8成交量(股),9成交额
                code_num = sina_code[2:]
                ts_code = f"{code_num}.{'SH' if sina_code.startswith('sh') else 'SZ'}"
                all_rows.append({
                    "ts_code": ts_code,
                    "open": float(fields[1]),
                    "high": float(fields[4]),
                    "low": float(fields[5]),
                    "close": float(fields[3]),
                    "vol": float(fields[8]) / 100,  # 新浪vol单位是股，转手
                    "amount": float(fields[9]),
                })
        except Exception:
            pass

    if not all_rows:
        raise ValueError("新浪接口未获取到有效数据")

    result = pd.DataFrame(all_rows)
    log(f"  [源3] 新浪成功：{len(result)} 只")
    return result, "新浪"


def fetch_realtime_stocks(need_codes=None):
    """
    三级数据源自动切换获取实时行情：AKShare → pytdx → 新浪。
    返回 (DataFrame, source_name)。
    """
    # 源1：AKShare（最优先，一次获取全市场）
    try:
        return _fetch_via_akshare(need_codes)
    except Exception as e:
        log(f"  [源1失败] AKShare: {e}")

    # 源2：pytdx（需要指定代码列表）
    if need_codes:
        try:
            return _fetch_via_pytdx(need_codes)
        except Exception as e:
            log(f"  [源2失败] pytdx: {e}")

        # 源3：新浪财经
        try:
            return _fetch_via_sina(need_codes)
        except Exception as e:
            log(f"  [源3失败] 新浪: {e}")

    log("  [错误] 所有数据源均失败！")
    return pd.DataFrame(), "无"


def fetch_realtime_hs300(last_close=None):
    """
    获取HS300 ETF (510300) 14:30实时价格，三级兜底。
    last_close: 昨日收盘价，用于校验实时价格合理性（偏差>20%丢弃，继续下一源）
    返回 (price, source_name)，失败返回 (None, "无")。
    """
    def _validate(price, source):
        """校验价格合理性，通过返回True"""
        if last_close is not None and last_close > 0:
            deviation = abs(price - last_close) / last_close
            if deviation > 0.20:
                log(f"  [{source}] HS300价格{price:.3f}与昨收{last_close:.3f}"
                    f"偏差{deviation*100:.1f}%，丢弃！")
                return False
        return True

    # 源1：AKShare ETF接口
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        if df is not None and not df.empty:
            row = df[df["代码"] == "510300"]
            if not row.empty:
                price = float(row.iloc[0]["最新价"])
                if price > 0 and _validate(price, "AKShare"):
                    return price, "AKShare"
    except Exception as e:
        log(f"  [HS300源1失败] AKShare: {e}")

    # 源2：pytdx
    try:
        from pytdx.hq import TdxHq_API
        api = TdxHq_API()
        for host, port in TDX_SERVERS:
            try:
                api.connect(host, port)
                quotes = api.get_security_quotes([(1, "510300")])
                api.disconnect()
                if quotes and quotes[0]["price"] > 0:
                    price = float(quotes[0]["price"])
                    if _validate(price, "pytdx"):
                        return price, "pytdx"
                    else:
                        break  # pytdx数据异常，跳过所有pytdx服务器，直接去新浪
            except Exception:
                continue
    except Exception as e:
        log(f"  [HS300源2失败] pytdx: {e}")

    # 源3：新浪
    try:
        url = "https://hq.sinajs.cn/list=sh510300"
        resp = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        if resp.status_code == 200:
            fields = resp.text.split('"')[1].split(",")
            if len(fields) > 3 and float(fields[3]) > 0:
                price = float(fields[3])
                if _validate(price, "新浪"):
                    return price, "新浪"
    except Exception as e:
        log(f"  [HS300源3失败] 新浪: {e}")

    return None, "无"


# ══════════════════════════════════════════════════════════════
# Tushare历史数据下载（截至昨天，与16:00版完全一致）
# ══════════════════════════════════════════════════════════════

def get_today_trade_date():
    """获取今天的交易日期（如果今天是交易日）"""
    now = datetime.now()
    end = now.strftime("%Y%m%d")
    start = (now - timedelta(days=10)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    trade_days = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    if end in trade_days:
        return end
    return None


def get_yesterday_trade_date():
    """获取最近一个已完成的交易日（用于下载Tushare历史数据）"""
    now = datetime.now()
    end = now.strftime("%Y%m%d")
    start = (now - timedelta(days=15)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    trade_days = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    # 今天14:30还没收盘，Tushare pro.daily()拿不到今天数据
    # 所以用今天之前最近的交易日
    past_days = [d for d in trade_days if d < end]
    return past_days[-1] if past_days else None


def download_hs300_history(lookback=80):
    """下载HS300 ETF历史日线（截至昨天）"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y%m%d")
    df = pro.fund_daily(ts_code="510300.SH", start_date=start, end_date=end,
                        fields="ts_code,trade_date,close")
    if df is not None and not df.empty:
        return df.sort_values("trade_date").reset_index(drop=True)
    return pd.DataFrame()


def download_stock_basics():
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name,industry,list_date,market")
    df = df[df["ts_code"].str.match(r"^(00[0-2]|60)")]
    df = df[~df["name"].str.contains("ST|退", na=False)]
    return df.reset_index(drop=True)


def download_daily_batch(codes, lookback=80):
    """批量下载历史日线（Tushare，截至昨天的数据）"""
    yesterday = get_yesterday_trade_date()
    if not yesterday:
        return pd.DataFrame()
    start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y%m%d")
    all_data = []
    for i, code in enumerate(codes):
        try:
            df = pro.daily(ts_code=code, start_date=start, end_date=yesterday,
                           fields="ts_code,trade_date,open,high,low,close,vol,amount")
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            log(f"  历史日线进度：{i+1}/{len(codes)}")
            time.sleep(0.5)
        time.sleep(0.06)
    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    return result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def download_fina_for_codes(codes):
    all_data = []
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
            log(f"  财报进度：{i+1}/{len(codes)}")
            time.sleep(0.5)
        time.sleep(0.06)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True).sort_values(["ts_code", "end_date"]).reset_index(drop=True)


def download_valuation_snap(date):
    try:
        df = pro.daily_basic(trade_date=date,
                             fields="ts_code,trade_date,pe_ttm,pb,total_mv,turnover_rate,dv_ratio")
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# 拼接历史日线 + 今日实时行情 → 完整日线序列
# ══════════════════════════════════════════════════════════════

def merge_history_and_realtime(history_daily, realtime_snap, today, ind_map):
    """
    将Tushare历史日线（截至昨天）和AKShare实时行情（今天14:30）合并。
    为每只股票构造今日的"伪日线"记录，拼接到历史数据后面。
    返回与原始stock_daily格式相同的DataFrame。
    """
    # 构造今日伪日线记录
    today_records = []
    for _, rt in realtime_snap.iterrows():
        code = rt["ts_code"]
        today_records.append({
            "ts_code": code,
            "trade_date": today,
            "open": rt["open"],
            "high": rt["high"],
            "low": rt["low"],
            "close": rt["close"],      # 14:30现价作为准收盘价
            "vol": rt["vol"],
            "amount": rt.get("amount", 0),
        })

    if not today_records:
        return history_daily

    today_df = pd.DataFrame(today_records)

    # 去掉历史数据中如果意外包含今天的记录（不应该有，但安全起见）
    if not history_daily.empty:
        history_daily = history_daily[history_daily["trade_date"] != today]

    # 拼接
    merged = pd.concat([history_daily, today_df], ignore_index=True)
    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return merged


# ══════════════════════════════════════════════════════════════
# 技术指标计算（与v27b回测完全一致）
# ══════════════════════════════════════════════════════════════

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

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / c

    df["volatility20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    for ma_name, ma_col in [("ma5", "ma5"), ("ma10", "ma10"), ("ma20", "ma20")]:
        below = (c < df[ma_col]).astype(int)
        streaks = []; cnt = 0
        for v in below:
            cnt = cnt + 1 if v else 0
            streaks.append(cnt)
        df[f"below_{ma_name}_streak"] = streaks

    return df


# ══════════════════════════════════════════════════════════════
# Regime判断（与v27b回测一致）
# ══════════════════════════════════════════════════════════════

def get_regime(hs300_data, date):
    hs = hs300_data[hs300_data["trade_date"] <= date].sort_values("trade_date")
    if len(hs) < 60:
        return "NEUTRAL"
    c = hs["close"].astype(float)
    cur = float(c.iloc[-1])
    ma20 = c.tail(20).mean()
    ma60 = c.tail(60).mean()
    if cur > ma60 and ma20 > ma60:
        return "BULL"
    if cur > ma60 * 0.97:
        return "NEUTRAL"
    return "BEAR"

def update_regime(state, hs300_data, date):
    """
    基于HS300历史数据回溯连续天数，不依赖脚本运行次数。
    往前扫描最近N个交易日的raw regime，数出连续相同的天数。
    """
    raw_regime = get_regime(hs300_data, date)
    confirmed = state["confirmed_regime"]

    if raw_regime == confirmed:
        # 信号与当前一致，无需切换
        state["pending_regime"] = confirmed
        state["pending_count"] = 0
    else:
        # 信号与当前不同，回溯历史数据数连续天数
        hs_dates = sorted(hs300_data["trade_date"].unique())
        hs_dates = [d for d in hs_dates if d <= date]

        consecutive = 0
        for d in reversed(hs_dates):
            d_regime = get_regime(hs300_data, d)
            if d_regime == raw_regime:
                consecutive += 1
            else:
                break
            if consecutive >= REGIME_CONFIRM_DAYS:
                break  # 已经够了，不用继续往前看

        state["pending_regime"] = raw_regime
        state["pending_count"] = consecutive
        log(f"  regime回溯：{raw_regime}已连续{consecutive}天（需{REGIME_CONFIRM_DAYS}天确认）")

        if consecutive >= REGIME_CONFIRM_DAYS:
            state["confirmed_regime"] = raw_regime
            state["pending_count"] = 0

    return state["confirmed_regime"], raw_regime


# ══════════════════════════════════════════════════════════════
# 买入信号（与v27b回测一致）
# ══════════════════════════════════════════════════════════════

def check_buy(row):
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
    if not (close > ma20 > ma60):
        return False, "", 0
    if recent_bk < 1:
        return False, "", 0
    if pd.isna(vol_ratio) or vol_ratio < VOLUME_RATIO:
        return False, "", 0
    if mom < MIN_MOM20 or mom > MAX_MOM20:
        return False, "", 0
    if atr_r > ATR_THRESHOLD:
        return False, "", 0

    if mom <= MOM_PREFERRED:
        mom_score = (mom - MIN_MOM20) / (MOM_PREFERRED - MIN_MOM20) * 100
    else:
        mom_score = max(0, 100 - (mom - MOM_PREFERRED) / (MAX_MOM20 - MOM_PREFERRED) * 60)
    # v27b：放量评分封顶在2.5x，超过2.5x扣分
    if vol_ratio <= 2.5:
        vol_score = min(vol_ratio / VOLUME_RATIO * 50, 100)
    else:
        vol_score = max(0, 100 - (vol_ratio - 2.5) / 2.5 * 80)  # 超2.5x急剧扣分
    pullback_bonus = 15 if not_at_high else 0
    tech_score = mom_score * 0.5 + vol_score * 0.3 + pullback_bonus

    reason = f"回踩确认|放量{vol_ratio:.1f}x|动量{mom:.1f}%|ATR{atr_r*100:.1f}%"
    return True, reason, tech_score


# ══════════════════════════════════════════════════════════════
# 卖出信号（严格对齐v27b回测）
# ══════════════════════════════════════════════════════════════

def check_sell(pos, row, regime):
    close = float(row["close"])
    cost = pos["cost"]
    peak = pos.get("peak_price", cost)
    pnl = (close - cost) / cost
    drawdown = (close - peak) / peak if peak > 0 else 0

    partial_done_1 = pos.get("partial_1_done", False)
    partial_done_2 = pos.get("partial_2_done", False)

    if regime == "BEAR":
        eff_stop_loss = BEAR_STOP_LOSS_PCT
        eff_trail_stop = BEAR_TRAIL_STOP_PCT
        eff_trail_ma_key = "below_ma5_streak"
        eff_trail_ma_days = BEAR_TRAIL_MA_DAYS
        eff_trail_ma_label = "MA5"
    elif regime == "NEUTRAL":
        eff_stop_loss = (STOP_LOSS_PCT + BEAR_STOP_LOSS_PCT) / 2
        eff_trail_stop = (TRAIL_STOP_PCT + BEAR_TRAIL_STOP_PCT) / 2
        eff_trail_ma_key = "below_ma10_streak"
        eff_trail_ma_days = TRAIL_BELOW_MA10_DAYS
        eff_trail_ma_label = "MA10"
    else:  # BULL
        eff_stop_loss = 0.15                         # ② v5g
        eff_trail_stop = 0.12                        # ① v5c
        eff_trail_ma_key = "below_ma10_streak"
        eff_trail_ma_days = 3                        # ③ v5j
        eff_trail_ma_label = "MA10"

    if pnl <= -eff_stop_loss:
        return True, f"止损{pnl*100:.1f}%(阈值{eff_stop_loss*100:.0f}%)", 1.0
    if peak > cost * 1.05 and drawdown <= -eff_trail_stop:
        return True, f"回撤止盈(峰值回落{drawdown*100:.1f}%,阈值{eff_trail_stop*100:.0f}%)", 1.0
    if pnl >= PARTIAL_PROFIT_1 and not partial_done_1:
        return True, f"分批止盈①(+{pnl*100:.1f}%)", 0.5
    if pnl >= PARTIAL_PROFIT_2 and partial_done_1 and not partial_done_2:
        return True, f"分批止盈②(+{pnl*100:.1f}%)", 0.5
    if pnl > 0.03 and row.get(eff_trail_ma_key, 0) >= eff_trail_ma_days:
        return True, f"趋势止盈(破{eff_trail_ma_label}连续{row[eff_trail_ma_key]}天)", 1.0
    if row.get("below_ma20_streak", 0) >= TRAIL_BELOW_MA20_DAYS:
        return True, f"破MA20连续{row['below_ma20_streak']}天", 1.0
    return False, "", 0


# ══════════════════════════════════════════════════════════════
# 基本面评分（与v27b回测一致）
# ══════════════════════════════════════════════════════════════

def _normalize_series(s, clip_low=None, clip_high=None):
    s = s.copy()
    if clip_low is not None: s = s.clip(lower=clip_low)
    if clip_high is not None: s = s.clip(upper=clip_high)
    lo, hi = s.min(), s.max()
    if hi == lo: return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100

def score_fundamentals(fina_data, val_snap, basics, date):
    result = {}
    if fina_data.empty:
        return result
    available = fina_data[fina_data["ann_date"] <= date].copy()
    if available.empty:
        return result
    latest = available.sort_values("end_date").groupby("ts_code").last().reset_index()
    ind_map = dict(zip(basics["ts_code"], basics["industry"]))
    latest["industry"] = latest["ts_code"].map(ind_map).fillna("")
    latest["is_tech"] = latest["industry"].isin(TECH_INDUSTRIES)

    std_mask = ((~latest["is_tech"]) & (latest["roe_dt"] >= MIN_ROE) &
                (latest["grossprofit_margin"] >= MIN_GROSS_MARGIN) &
                (latest["or_yoy"] >= MIN_REVENUE_GROWTH))
    tech_mask = ((latest["is_tech"]) & (latest["roe_dt"] >= TECH_MIN_ROE) &
                 (latest["grossprofit_margin"] >= TECH_MIN_GROSS_MARGIN) &
                 (latest["or_yoy"] >= MIN_REVENUE_GROWTH))
    pass_mask = std_mask | tech_mask
    non_fin = ~latest["industry"].isin(FINANCIAL_INDUSTRIES)
    debt_mask = (latest["debt_to_assets"] < MAX_DEBT_RATIO) | (~non_fin)
    pass_mask = pass_mask & debt_mask

    val_codes = set()
    if not val_snap.empty:
        snap = val_snap.copy()
        snap["industry"] = snap["ts_code"].map(ind_map).fillna("")
        snap["is_tech"] = snap["industry"].isin(TECH_INDUSTRIES)
        std_ok = snap[(~snap["is_tech"]) &
                      (snap["pe_ttm"] > PE_RANGE[0]) & (snap["pe_ttm"] < PE_RANGE[1]) &
                      (snap["pb"] > PB_RANGE[0]) & (snap["pb"] < PB_RANGE[1]) &
                      (snap["total_mv"] > MIN_MARKET_CAP * 10000)]
        tech_ok = snap[(snap["is_tech"]) &
                       (snap["pe_ttm"] > TECH_PE_RANGE[0]) & (snap["pe_ttm"] < TECH_PE_RANGE[1]) &
                       (snap["pb"] > PB_RANGE[0]) & (snap["pb"] < PB_RANGE[1]) &
                       (snap["total_mv"] > TECH_MIN_MARKET_CAP * 10000)]
        val_codes = set(std_ok["ts_code"].tolist()) | set(tech_ok["ts_code"].tolist())

    passed_idx = latest[pass_mask].index
    if len(passed_idx) == 0:
        return result
    sub = latest.loc[passed_idx].copy()
    roe_score = _normalize_series(sub["roe_dt"].fillna(0), clip_low=0, clip_high=50)
    rev_score = _normalize_series(sub["or_yoy"].fillna(0), clip_low=-10, clip_high=80)
    gm_score = _normalize_series(sub["grossprofit_margin"].fillna(0), clip_low=10, clip_high=80)
    fund_scores = FUND_WEIGHT_ROE * roe_score + FUND_WEIGHT_REVENUE * rev_score + FUND_WEIGHT_MARGIN * gm_score
    ocf_score = _normalize_series(sub["ocfps"].fillna(0), clip_low=-1, clip_high=10)
    np_score = _normalize_series(sub["netprofit_yoy"].fillna(0), clip_low=-20, clip_high=100)
    fund_scores = fund_scores + FUND_BONUS_CASHFLOW * ocf_score + FUND_BONUS_PEG * np_score

    for idx, row in sub.iterrows():
        code = row["ts_code"]
        passed = True
        if val_codes and code not in val_codes:
            passed = False
        result[code] = {"passed": passed, "fund_score": float(fund_scores.get(idx, 0)),
                        "is_tech": bool(row["is_tech"]), "industry": row["industry"]}
    return result


# ══════════════════════════════════════════════════════════════
# 行业热度（与v27b回测一致）
# ══════════════════════════════════════════════════════════════

def calc_industry_heat(indicators, date):
    industry_data = defaultdict(lambda: {"mom5": [], "mom20": [], "vol_ratio": []})
    for ts_code, idf in indicators.items():
        row = idf[idf["trade_date"] == date]
        if row.empty: continue
        row = row.iloc[0]
        mom5 = row.get("mom5", np.nan); mom20 = row.get("mom20", np.nan)
        vol_r = row.get("vol_ratio", np.nan); ind = row.get("industry", "")
        if ind and pd.notna(mom20):
            if pd.notna(mom5): industry_data[ind]["mom5"].append(mom5)
            industry_data[ind]["mom20"].append(mom20)
            if pd.notna(vol_r): industry_data[ind]["vol_ratio"].append(vol_r)
    raw_heat = {}
    for ind, d in industry_data.items():
        if len(d["mom20"]) < 3: continue
        m5 = np.mean(d["mom5"]) if d["mom5"] else 0
        m20 = np.mean(d["mom20"])
        heat = 0.5 * m5 + 0.5 * m20
        if d["vol_ratio"]:
            avg_vr = np.mean(d["vol_ratio"])
            if avg_vr > 1.2: heat *= 1.1
        raw_heat[ind] = heat
    if not raw_heat: return {}
    sorted_inds = sorted(raw_heat.items(), key=lambda x: x[1])
    n = len(sorted_inds)
    return {ind: (rank / max(n - 1, 1)) * 100 for rank, (ind, _) in enumerate(sorted_inds)}


# ══════════════════════════════════════════════════════════════
# v18c新增：东财概念板块热度
# ══════════════════════════════════════════════════════════════

CONCEPT_MAP_FILE = WORK_DIR / "concept_map.json"
CONCEPT_MAP_DATE_KEY = "concept_map_date"

def download_concept_mapping():
    """下载东财概念板块成分映射：stock → [concept_codes]"""
    yesterday = get_yesterday_trade_date()
    if not yesterday:
        return {}
    log("  下载东财概念板块映射...")

    # 获取所有概念代码
    try:
        idx = pro.dc_index(trade_date=yesterday, fields="ts_code,name")
        if idx is None or idx.empty:
            return {}
    except Exception as e:
        log(f"  [警告] dc_index获取概念列表失败: {e}")
        return {}

    concept_codes = idx["ts_code"].unique().tolist()
    log(f"  共 {len(concept_codes)} 个概念板块")

    stock_to_concepts = defaultdict(list)
    for i, ccode in enumerate(concept_codes):
        try:
            df = pro.dc_member(trade_date=yesterday, ts_code=ccode, fields="ts_code,con_code")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    stock_to_concepts[row["con_code"]].append(ccode)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            log(f"    概念映射进度：{i+1}/{len(concept_codes)}")
        time.sleep(0.35)

    result = dict(stock_to_concepts)
    log(f"  概念映射完成：{len(result)} 只股票")
    return result


def load_concept_mapping(state, today):
    """加载概念映射，超过20天或不存在时刷新"""
    map_date = state.get(CONCEPT_MAP_DATE_KEY, "")
    days_since = 999
    if map_date:
        try:
            days_since = (pd.Timestamp(today) - pd.Timestamp(map_date)).days
        except Exception:
            pass
    elif CONCEPT_MAP_FILE.exists():
        # state中没有日期（升级/reset后），用文件修改时间作为fallback
        try:
            import os
            mtime = os.path.getmtime(CONCEPT_MAP_FILE)
            file_date = datetime.fromtimestamp(mtime).strftime("%Y%m%d")
            days_since = (pd.Timestamp(today) - pd.Timestamp(file_date)).days
            log(f"  概念映射日期从文件修改时间恢复：{file_date}（{days_since}天前）")
        except Exception:
            pass

    if CONCEPT_MAP_FILE.exists() and days_since <= 20:
        with open(CONCEPT_MAP_FILE, "r") as f:
            mapping = json.load(f)
        log(f"  概念映射缓存（{map_date}，{days_since}天前）：{len(mapping)} 只")
        return mapping

    mapping = download_concept_mapping()
    if mapping:
        with open(CONCEPT_MAP_FILE, "w") as f:
            json.dump(mapping, f)
        state[CONCEPT_MAP_DATE_KEY] = today
    return mapping


def get_concept_heat_today(stock_concepts):
    """获取昨日概念板块涨跌排名，计算每只股票的概念热度百分位"""
    yesterday = get_yesterday_trade_date()
    if not yesterday or not stock_concepts:
        return {}

    try:
        dc = pro.dc_index(trade_date=yesterday, fields="ts_code,pct_change")
        if dc is None or dc.empty:
            return {}
    except Exception as e:
        log(f"  [警告] dc_index获取昨日概念行情失败: {e}")
        return {}

    dc["pct_change"] = pd.to_numeric(dc["pct_change"], errors="coerce")
    dc["pct_rank"] = dc["pct_change"].rank(pct=True) * 100
    concept_rank = dict(zip(dc["ts_code"], dc["pct_rank"]))

    result = {}
    for stock, concepts in stock_concepts.items():
        ranks = [concept_rank[c] for c in concepts if c in concept_rank]
        if ranks:
            result[stock] = max(ranks)
    return result


# ══════════════════════════════════════════════════════════════
# v18c新增：筹码胜率
# ══════════════════════════════════════════════════════════════

def get_winner_rates(codes):
    """获取昨日筹码胜率（cyq_perf），返回 {ts_code: winner_rate}"""
    yesterday = get_yesterday_trade_date()
    if not yesterday or not codes:
        return {}

    result = {}
    for i, code in enumerate(codes):
        try:
            df = pro.cyq_perf(ts_code=code, start_date=yesterday, end_date=yesterday,
                              fields="ts_code,winner_rate")
            if df is not None and not df.empty:
                wr = pd.to_numeric(df.iloc[0]["winner_rate"], errors="coerce")
                if pd.notna(wr):
                    result[code] = float(wr)
        except Exception:
            pass
        time.sleep(0.15)

    log(f"  筹码胜率获取：{len(result)}/{len(codes)} 只")
    return result


# ══════════════════════════════════════════════════════════════
# 飞书推送
# ══════════════════════════════════════════════════════════════

def send_feishu(title, content):
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": [{"tag": "markdown", "content": content}]
        }
    }
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        if resp.status_code == 200:
            log(f"飞书推送成功：{title}")
        else:
            log(f"飞书推送失败：{resp.status_code} {resp.text}")
    except Exception as e:
        log(f"飞书推送异常：{e}")


# ══════════════════════════════════════════════════════════════
# 主逻辑（14:30盘中执行版）
# ══════════════════════════════════════════════════════════════

def run(dry_run=False):
    log("=" * 60)
    log("v27b 14:30盘中信号生成器启动")
    log("策略：v27b = v20a + 放量>2.5x扣分")
    log("数据：Tushare历史日线(截至昨天) + AKShare实时行情(今天14:30)")
    log("=" * 60)

    # 执行时间校验
    now = datetime.now()
    if not (14 <= now.hour < 15) and not dry_run:
        log(f"[警告] 当前时间{now.strftime('%H:%M')}，非14:00-15:00交易窗口，实时数据可能不准确")
    if dry_run:
        log("[DRY-RUN] 测试模式，不推送飞书、不更新状态")

    # ── 0. 确定交易日 ──
    today = get_today_trade_date()
    is_trading_day = today is not None
    if not is_trading_day:
        today = get_yesterday_trade_date()
        if not today:
            log("无法获取最近交易日，退出")
            return
        log(f"非交易日，使用最近交易日（{today}）数据生成信号")
    else:
        log(f"交易日：{today}（14:30盘中执行）")

    state = load_state()
    if state["last_run_date"] == today and not dry_run:
        log(f"今日（{today}）已运行过，跳过（用--force强制运行）")
        return

    # ── 1. 获取HS300实时价格 + 历史数据 → 判断regime ──
    log("[1/6] 沪深300 regime判断...")
    hs300_hist = download_hs300_history(lookback=80)
    if hs300_hist.empty:
        log("HS300历史数据为空，退出")
        return

    # 获取HS300实时或历史价格
    if is_trading_day:
        last_close = float(hs300_hist.sort_values("trade_date")["close"].iloc[-1])
        hs300_realtime_price, hs300_source = fetch_realtime_hs300(last_close=last_close)

        if hs300_realtime_price is not None:
            # 构造今日HS300伪日线
            hs300_today = pd.DataFrame([{
                "ts_code": "510300.SH",
                "trade_date": today,
                "close": hs300_realtime_price,
            }])
            hs300_hist = hs300_hist[hs300_hist["trade_date"] != today]
            hs300 = pd.concat([hs300_hist, hs300_today], ignore_index=True).sort_values("trade_date")
            log(f"  HS300实时价：{hs300_realtime_price:.3f}（{hs300_source}）")
        else:
            log("  [警告] 未获取到HS300实时价格，使用昨日数据判断regime")
            hs300 = hs300_hist
    else:
        log("  非交易日，使用历史收盘数据判断regime")
        hs300 = hs300_hist

    regime, raw_regime = update_regime(state, hs300, today)
    icon = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}[regime]
    raw_tag = f"(raw:{raw_regime})" if raw_regime != regime else ""
    log(f"  当前regime：{icon}{regime}{raw_tag}")

    # ── 2. 下载股票基本信息和估值快照 ──
    log("[2/6] 股票基本信息和估值...")
    basics = download_stock_basics()
    ind_map = dict(zip(basics["ts_code"], basics["industry"]))
    name_map = dict(zip(basics["ts_code"], basics["name"]))
    log(f"  主板股票：{len(basics)} 只")

    # 估值快照用昨天的（今天盘中还拿不到）
    yesterday = get_yesterday_trade_date()
    val_snap = download_valuation_snap(yesterday) if yesterday else pd.DataFrame()
    log(f"  估值快照（{yesterday}）：{len(val_snap)} 条")

    # ── 3. 更新基本面评分（每20天刷新）──
    log("[3/6] 基本面评分...")
    fund_date = state.get("fund_date", "")
    days_since_fund = 999
    if fund_date:
        try:
            days_since_fund = (pd.Timestamp(today) - pd.Timestamp(fund_date)).days
        except Exception:
            pass

    if days_since_fund > 20 or not state.get("fund_scores"):
        log("  基本面评分过期或首次运行，重新计算...")
        pre_filtered = set()
        if not val_snap.empty:
            snap = val_snap.copy()
            snap["industry"] = snap["ts_code"].map(ind_map).fillna("")
            snap["is_tech"] = snap["industry"].isin(TECH_INDUSTRIES)
            std = snap[~snap["is_tech"]]
            std_ok = std[(std["pe_ttm"] > PE_RANGE[0]) & (std["pe_ttm"] < PE_RANGE[1]) &
                         (std["pb"] > PB_RANGE[0]) & (std["pb"] < PB_RANGE[1]) &
                         (std["total_mv"] > MIN_MARKET_CAP * 10000)]
            pre_filtered.update(std_ok["ts_code"].tolist())
            tech = snap[snap["is_tech"]]
            tech_ok = tech[(tech["pe_ttm"] > TECH_PE_RANGE[0]) & (tech["pe_ttm"] < TECH_PE_RANGE[1]) &
                           (tech["pb"] > PB_RANGE[0]) & (tech["pb"] < PB_RANGE[1]) &
                           (tech["total_mv"] > TECH_MIN_MARKET_CAP * 10000)]
            pre_filtered.update(tech_ok["ts_code"].tolist())
        pre_filtered = sorted(pre_filtered & set(basics["ts_code"].tolist()))
        log(f"  估值初筛：{len(pre_filtered)} 只，开始下载财务...")
        fina_data = download_fina_for_codes(pre_filtered)
        fund_scores = score_fundamentals(fina_data, val_snap, basics, today)
        state["fund_scores"] = {k: v for k, v in fund_scores.items() if v.get("passed")}
        state["fund_date"] = today
        log(f"  基本面通过：{len(state['fund_scores'])} 只")
    else:
        log(f"  使用缓存（{fund_date}，{days_since_fund}天前）：{len(state['fund_scores'])} 只好公司")

    fund_scores = state["fund_scores"]

    # ── 3b. 加载概念板块映射（v18c，每20天刷新）──
    log("[3b/6] 加载概念板块映射...")
    stock_concepts = load_concept_mapping(state, today)

    # ── 4. 下载历史日线（截至昨天）──
    log("[4/6] 下载历史日线（Tushare，截至昨天）...")
    need_codes = set(state["positions"].keys()) | set(fund_scores.keys())
    need_codes = sorted(need_codes & set(basics["ts_code"].tolist()))
    log(f"  需下载：{len(need_codes)} 只")

    history_daily = download_daily_batch(need_codes, lookback=80)
    if history_daily.empty:
        log("历史日线为空，退出")
        return
    log(f"  历史日线完成：{history_daily['ts_code'].nunique()} 只")

    # ── 5. 获取行情 → 计算技术指标 ──
    if is_trading_day:
        log("[5/6] 获取14:30实时行情并计算技术指标...")
        realtime_snap, rt_source = fetch_realtime_stocks(need_codes=set(need_codes))
        if realtime_snap.empty:
            log("[错误] 所有数据源均失败，无法生成信号，退出")
            return

        # 只保留需要的股票的实时数据
        realtime_snap = realtime_snap[realtime_snap["ts_code"].isin(need_codes)]
        log(f"  匹配到实时数据：{len(realtime_snap)} 只")

        # 拼接历史 + 今日实时
        merged_daily = merge_history_and_realtime(history_daily, realtime_snap, today, ind_map)
    else:
        log("[5/6] 非交易日，使用历史收盘数据计算技术指标...")
        rt_source = "历史收盘"
        # 历史日线已包含最近交易日数据，无需拼接
        merged_daily = history_daily.copy()
        # 构造空的 realtime_snap 供后续 today_open_map 使用
        last_day = history_daily[history_daily["trade_date"] == today]
        realtime_snap = last_day[["ts_code", "open", "high", "low", "close", "vol"]].copy()
        if "amount" in last_day.columns:
            realtime_snap["amount"] = last_day["amount"]

    merged_daily["industry"] = merged_daily["ts_code"].map(ind_map).fillna("")

    # 计算技术指标
    indicators = {}
    for ts_code in merged_daily["ts_code"].unique():
        sdf = merged_daily[merged_daily["ts_code"] == ts_code]
        indicators[ts_code] = calc_indicators(sdf)
    for ts_code in indicators:
        if "industry" not in indicators[ts_code].columns:
            indicators[ts_code]["industry"] = ind_map.get(ts_code, "")
    log(f"  技术指标完成：{len(indicators)} 只")

    # ── 6. 生成卖出和买入信号（逻辑与v27b回测一致）──
    log("[6/6] 生成信号...")

    # 辅助数据：用于飞书消息中展示参考价格
    # T日开盘价lookup（来自AKShare实时快照）
    today_open_map = dict(zip(realtime_snap["ts_code"], realtime_snap["open"]))

    def get_prev_day_prices(ts_code):
        """从indicators中获取T-1日的开盘价和收盘价"""
        idf = indicators.get(ts_code)
        if idf is None:
            return None, None
        hist = idf[idf["trade_date"] < today].sort_values("trade_date")
        if hist.empty:
            return None, None
        last = hist.iloc[-1]
        prev_open = float(last["open"]) if pd.notna(last.get("open")) else None
        prev_close = float(last["close"]) if pd.notna(last.get("close")) else None
        return prev_open, prev_close

    sell_signals = []
    buy_signals = []

    # ═══ 卖出信号扫描 ═══
    positions = state["positions"]
    for code, pos in list(positions.items()):
        if code not in indicators:
            continue
        idf = indicators[code]
        row = idf[idf["trade_date"] == today]
        if row.empty:
            continue
        row = row.iloc[0]

        cur_price = float(row["close"])
        pos["current_price"] = cur_price
        if cur_price > pos.get("peak_price", 0):
            pos["peak_price"] = cur_price

        should_sell, reason, partial_ratio = check_sell(pos, row, regime)
        if should_sell:
            pnl = (cur_price - pos["cost"]) / pos["cost"]
            sell_pct = int(partial_ratio * 100)
            prev_open, prev_close = get_prev_day_prices(code)
            sell_signals.append({
                "code": code,
                "name": pos.get("name", name_map.get(code, code)),
                "price": round(cur_price, 2),              # 14:30实时价
                "today_open": round(today_open_map.get(code, 0), 2),  # 今日开盘价
                "prev_open": round(prev_open, 2) if prev_open else None,   # T-1开盘价
                "prev_close": round(prev_close, 2) if prev_close else None, # T-1收盘价
                "cost": round(pos["cost"], 2),
                "pnl": round(pnl * 100, 1),
                "reason": reason,
                "sell_pct": sell_pct,
                "total_shares": pos.get("shares", 0),
                "sell_shares": pos.get("shares", 0) if partial_ratio >= 1.0
                    else max(int(pos.get("shares", 0) * partial_ratio / 100) * 100, 100),
            })
            if partial_ratio >= 1.0:
                del positions[code]
            else:
                if "分批止盈①" in reason:
                    pos["partial_1_done"] = True
                elif "分批止盈②" in reason:
                    pos["partial_2_done"] = True

    # ═══ 全市场扫描（始终执行，用于 Top10 候选推送）═══
    industry_heat = calc_industry_heat(indicators, today)

    # v18c：获取概念热度
    concept_heat_map = get_concept_heat_today(stock_concepts)
    scan_codes = [c for c in fund_scores if c in indicators and c not in positions]

    # 第一轮：先用check_buy过滤，不含winner_rate
    pre_candidates = []
    for ts_code in scan_codes:
        idf = indicators[ts_code]
        row = idf[idf["trade_date"] == today]
        if row.empty:
            continue
        row = row.iloc[0]

        should_buy, reason, tech_score = check_buy(row)
        if not should_buy:
            continue
        pre_candidates.append((ts_code, row, reason, tech_score))

    # 第二轮：只对通过check_buy的候选股获取筹码胜率（通常5-15只，而非200+只）
    if pre_candidates:
        buy_codes = [c[0] for c in pre_candidates]
        log(f"  获取候选股筹码胜率（{len(buy_codes)}只）...")
        winner_rates = get_winner_rates(buy_codes)
    else:
        winner_rates = {}

    candidates = []
    for ts_code, row, reason, tech_score in pre_candidates:
        # v18c：筹码胜率加成（+10分）
        wr = winner_rates.get(ts_code, None)
        wr_bonus = 10 if (wr is not None and wr > 50) else 0
        tech_score += wr_bonus

        fs = fund_scores.get(ts_code, {})
        fund_score = fs.get("fund_score", 0)
        ind = row.get("industry", "")
        ind_heat = industry_heat.get(ind, 50) if industry_heat else 50

        # v18c：融合概念热度（50%行业 + 50%概念）
        if concept_heat_map:
            cpt_heat = concept_heat_map.get(ts_code, 50)
            heat = 0.5 * ind_heat + 0.5 * cpt_heat
        else:
            heat = ind_heat

        tech_with_heat = tech_score * 0.7 + heat * 0.3

        # v20a：低弹性行业基本面权重降低
        if ind in LOW_ELASTIC_INDUSTRIES:
            w_tech = LOW_ELASTIC_TECH_W
            w_fund = LOW_ELASTIC_FUND_W
        else:
            w_tech = SCORE_WEIGHT_TECH
            w_fund = SCORE_WEIGHT_FUND

        final_score = w_tech * tech_with_heat + w_fund * fund_score

        prev_open, prev_close = get_prev_day_prices(ts_code)
        wr_tag = f" 胜率{wr:.0f}%" if wr is not None else ""
        candidates.append({
            "code": ts_code,
            "name": name_map.get(ts_code, ts_code),
            "close": round(float(row["close"]), 2),
            "today_open": round(today_open_map.get(ts_code, 0), 2),
            "prev_open": round(prev_open, 2) if prev_open else None,
            "prev_close": round(prev_close, 2) if prev_close else None,
            "mom20": round(float(row["mom20"]), 1) if pd.notna(row.get("mom20")) else 0,
            "industry": ind,
            "heat": round(heat, 1),
            "fund_score": round(fund_score, 1),
            "tech_score": round(tech_score, 1),
            "final_score": round(final_score, 2),
            "reason": reason + wr_tag,
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    watchlist = candidates[:10]  # 始终推送 Top10 候选

    # 买入信号：仅 BULL 模式且有空仓位时
    if regime == "BULL" and len(positions) < MAX_POSITIONS:
        slots = MAX_POSITIONS - len(positions)
        buy_signals = candidates[:slots]

    # ═══ 构建飞书消息并推送 ═══
    n_pos = len(positions)
    pos_names = ", ".join(f"{p.get('name', c)}" for c, p in list(positions.items())[:5])

    # 计算每只持仓的市值和仓位占比
    total_portfolio_value = TOTAL_CAPITAL  # 默认参考资金
    pos_values = {}
    for code, pos in positions.items():
        cur = pos.get("current_price", pos["cost"])
        shares = pos.get("shares", 0)
        mv = cur * shares
        pos_values[code] = mv
    total_mv = sum(pos_values.values())
    # 估算总资产 = 持仓市值 + 剩余现金（近似）
    # 现金无法精确知道（用户手动管理），用TOTAL_CAPITAL估算
    est_total = max(total_mv, TOTAL_CAPITAL)
    cash_est = max(est_total - total_mv, 0)
    pos_pct_total = total_mv / est_total * 100 if est_total > 0 else 0

    lines = []
    if is_trading_day:
        lines.append(f"**日期**：{today}  ⏰ **14:30盘中信号**")
    else:
        lines.append(f"**日期**：{today}  📅 **非交易日（基于历史收盘数据）**")
    lines.append(f"**Regime**：{icon} {regime}{raw_tag}")
    if n_pos:
        lines.append(f"**持仓**：{n_pos}只  总仓位≈{pos_pct_total:.0f}%（{pos_names}）")
    else:
        lines.append("**持仓**：空仓")
    if is_trading_day:
        lines.append(f"⚡ **请在15:00前完成交易操作**  |  数据源：{rt_source}")
    else:
        lines.append(f"📊 仅供参考（非交易日无法交易）  |  数据源：{rt_source}")
    lines.append("")

    if sell_signals:
        lines.append(f"🔴 **卖出信号（{len(sell_signals)}只）— 立即卖出！**")
        for s in sell_signals:
            pnl_icon = "📈" if s["pnl"] > 0 else "📉"
            # 计算该持仓的仓位占比
            s_mv = pos_values.get(s["code"], 0)
            s_pct = s_mv / est_total * 100 if est_total > 0 else 0
            lines.append(f"  {pnl_icon} **{s['name']}**（{s['code']}）"
                         f"  仓位{s_pct:.0f}%  盈亏{s['pnl']:+.1f}%")
            lines.append(f"    👉 **卖出{s['sell_shares']}股**"
                         f"（持有{s['total_shares']}股 × {s['sell_pct']}%）")
            lines.append(f"    成本{s['cost']}  |  "
                         f"昨开{s['prev_open'] or '-'}  昨收{s['prev_close'] or '-'}  |  "
                         f"今开{s['today_open']}  **14:30价{s['price']}**")
            lines.append(f"    原因：{s['reason']}")
        lines.append("")

    if buy_signals:
        lines.append(f"🟢 **买入信号（{len(buy_signals)}只）— 立即买入！**")
        # 计算每只建议买入金额和股数
        avail_slots = MAX_POSITIONS - n_pos + len([s for s in sell_signals if s["sell_pct"] == 100])
        per_pos_amount = est_total / MAX_POSITIONS  # 每个仓位的目标金额（等权）
        for i, b in enumerate(buy_signals):
            suggested_pct = round(100 / MAX_POSITIONS) if avail_slots > 0 else 0
            buy_price = b["close"]
            suggested_shares = int(per_pos_amount / buy_price / 100) * 100 if buy_price > 0 else 0
            suggested_amount = suggested_shares * buy_price
            lines.append(f"  {i+1}. **{b['name']}**（{b['code']}）"
                         f"  动量{b['mom20']}%")
            lines.append(f"    👉 **买入{suggested_shares}股 ≈ ¥{suggested_amount:,.0f}**"
                         f"（建议仓位≈{suggested_pct}%）")
            lines.append(f"    昨开{b['prev_open'] or '-'}  昨收{b['prev_close'] or '-'}  |  "
                         f"今开{b['today_open']}  **14:30价{b['close']}**")
            lines.append(f"    行业:{b['industry']}(热度{b['heat']})  综合{b['final_score']}分 | {b['reason']}")
        lines.append("")

    if not sell_signals and not buy_signals:
        lines.append("📊 **今日无交易信号**")
        if regime != "BULL":
            lines.append(f"  （{regime}模式，不新开仓）")
        lines.append("")

    # 始终推送 Top10 候选（不受 Regime 限制）
    if watchlist:
        lines.append(f"📋 **今日 Top10 候选**（综合评分排名，仅供观察）")
        for i, w in enumerate(watchlist):
            lines.append(f"  {i+1}. **{w['name']}**（{w['code']}）"
                         f"  综合{w['final_score']}  动量{w['mom20']}%"
                         f"  {w['industry']}(热度{w['heat']})  | {w['reason']}")
        lines.append("")

    if positions:
        lines.append("📋 **当前持仓**")
        for code, pos in positions.items():
            cur = pos.get("current_price", pos["cost"])
            shares = pos.get("shares", 0)
            mv = cur * shares
            pct = mv / est_total * 100 if est_total > 0 else 0
            pnl = (cur - pos["cost"]) / pos["cost"] * 100
            peak = pos.get("peak_price", cur)
            dd = (cur - peak) / peak * 100 if peak > 0 else 0
            p1 = "✅" if pos.get("partial_1_done") else "—"
            p2 = "✅" if pos.get("partial_2_done") else "—"
            lines.append(f"  {pos.get('name', code)}: ¥{cur:.2f}×{shares}股"
                         f"  **仓位{pct:.0f}%**  盈亏{pnl:+.1f}%"
                         f"  峰值回落{dd:.1f}%  分批①{p1}②{p2}")
        lines.append(f"  💰 现金≈¥{cash_est:,.0f}（{100-pos_pct_total:.0f}%）")

    lines.append("")
    lines.append("---")
    lines.append(f"v27b-14:30 | 夏普0.707 | 数据源:{rt_source}")

    content = "\n".join(lines)
    title = f"⏰ v27b盘中信号 {today} {icon}{regime}"

    send_feishu(title, content) if not dry_run else log("[DRY-RUN] 跳过飞书推送")
    log("\n" + content)

    if buy_signals:
        log("\n[提示] 买入后请运行以下命令更新持仓：")
        for b in buy_signals:
            log(f"  python3 daily_signal.py --add {b['code']} --cost <实际买入价> --shares <股数>")

    if not dry_run:
        state["last_run_date"] = today
        save_state(state)
        log(f"\n状态已保存到 {STATE_FILE}")
    else:
        log("\n[DRY-RUN] 跳过状态保存")
    log("运行完毕\n")


# ══════════════════════════════════════════════════════════════
# 持仓管理命令行工具（与16:00版完全一致）
# ══════════════════════════════════════════════════════════════

def cmd_add_position(code, cost, shares, name=""):
    state = load_state()
    if not name:
        try:
            basics = download_stock_basics()
            nm = dict(zip(basics["ts_code"], basics["name"]))
            name = nm.get(code, code)
        except Exception:
            name = code
    state["positions"][code] = {
        "name": name, "cost": float(cost), "shares": int(shares),
        "peak_price": float(cost),
        "buy_date": datetime.now().strftime("%Y%m%d"),
        "partial_1_done": False, "partial_2_done": False,
    }
    save_state(state)
    log(f"已添加持仓：{name}({code}) 成本{cost} {shares}股")

def cmd_remove_position(code):
    state = load_state()
    if code in state["positions"]:
        name = state["positions"][code].get("name", code)
        del state["positions"][code]
        save_state(state)
        log(f"已删除持仓：{name}({code})")
    else:
        log(f"持仓中未找到：{code}")

def cmd_list_positions():
    state = load_state()
    positions = state["positions"]
    if not positions:
        print("当前无持仓")
        return
    print(f"当前持仓（{len(positions)}只）：")
    print(f"  Regime：{state['confirmed_regime']}（pending:{state['pending_regime']} x{state['pending_count']}天）")
    for code, pos in positions.items():
        p1 = "✅" if pos.get("partial_1_done") else "—"
        p2 = "✅" if pos.get("partial_2_done") else "—"
        print(f"  {pos.get('name', code)}({code}): 成本{pos['cost']} "
              f"{pos.get('shares', '?')}股 峰值{pos.get('peak_price', '?')} "
              f"买入{pos.get('buy_date', '?')} 分批①{p1}②{p2}")

def cmd_reset_state():
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    log("状态已重置")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v27b 14:30盘中信号生成器")
    parser.add_argument("--add", type=str, help="添加持仓（股票代码，如600519.SH）")
    parser.add_argument("--cost", type=float, help="买入成本价")
    parser.add_argument("--shares", type=int, help="买入股数")
    parser.add_argument("--name", type=str, default="", help="股票名称（可选）")
    parser.add_argument("--remove", type=str, help="删除持仓（股票代码）")
    parser.add_argument("--list", action="store_true", help="查看当前持仓")
    parser.add_argument("--reset", action="store_true", help="重置状态")
    parser.add_argument("--force", action="store_true", help="强制运行（忽略今日已运行检查）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="测试模式（不推送飞书）")
    args = parser.parse_args()

    if args.add:
        if not args.cost or not args.shares:
            print("添加持仓需要 --cost 和 --shares 参数")
            print("示例：python3 daily_signal.py --add 600519.SH --cost 1800.5 --shares 100")
            sys.exit(1)
        cmd_add_position(args.add, args.cost, args.shares, args.name)
    elif args.remove:
        cmd_remove_position(args.remove)
    elif args.list:
        cmd_list_positions()
    elif args.reset:
        cmd_reset_state()
    else:
        if args.force:
            state = load_state()
            state["last_run_date"] = ""
            save_state(state)
        run(dry_run=args.dry_run)
