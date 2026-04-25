# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（动态调整）：
【配置了 TUSHARE_TOKEN 时】
1. TushareFetcher (Priority 0) - 🔥 最高优先级（动态提升）
2. EfinanceFetcher (Priority 0) - 同优先级
3. AkshareFetcher (Priority 1) - 来自 akshare 库
4. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
5. BaostockFetcher (Priority 3) - 来自 baostock 库
6. YfinanceFetcher (Priority 4) - 来自 yfinance 库

【未配置 TUSHARE_TOKEN 时】
1. EfinanceFetcher (Priority 0) - 最高优先级，来自 efinance 库
2. AkshareFetcher (Priority 1) - 来自 akshare 库
3. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
4. TushareFetcher (Priority 2) - 来自 tushare 库（不可用）
5. BaostockFetcher (Priority 3) - 来自 baostock 库
6. YfinanceFetcher (Priority 4) - 来自 yfinance 库

提示：优先级数字越小越优先，同优先级按初始化顺序排列
"""

import threading
from typing import Optional

from .base import BaseFetcher, DataFetcherManager
from .efinance_fetcher import EfinanceFetcher
from .akshare_fetcher import AkshareFetcher, is_hk_stock_code
from .tushare_fetcher import TushareFetcher
from .pytdx_fetcher import PytdxFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher
from .us_index_mapping import is_us_index_code, is_us_stock_code, get_us_index_yf_symbol, US_INDEX_MAPPING


# ========================================
# 进程内 DataFetcherManager 单例
# ========================================
# 历史问题：每个 Agent tool call 都新建一次 DataFetcherManager，
# 造成 6 个 Fetcher 反复重建、内置缓存反复作废、流控计数器归零，
# 在 Tushare 80次/分钟配额下极易触发限流。
# 单例可让所有调用方共享同一份 fetcher 列表与内存缓存。

_default_manager: Optional[DataFetcherManager] = None
_default_manager_lock = threading.Lock()


def get_default_manager() -> DataFetcherManager:
    """
    返回进程内共享的 DataFetcherManager 单例。

    线程安全：使用 double-checked locking。

    用法：
        from data_provider import get_default_manager
        manager = get_default_manager()
        df, source = manager.get_daily_data("605117", days=60)
    """
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = DataFetcherManager()
    return _default_manager


def reset_default_manager() -> None:
    """重置单例（仅供测试使用）。"""
    global _default_manager
    with _default_manager_lock:
        _default_manager = None


__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'EfinanceFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'PytdxFetcher',
    'BaostockFetcher',
    'YfinanceFetcher',
    'is_us_index_code',
    'is_us_stock_code',
    'is_hk_stock_code',
    'get_us_index_yf_symbol',
    'US_INDEX_MAPPING',
    'get_default_manager',
    'reset_default_manager',
]
