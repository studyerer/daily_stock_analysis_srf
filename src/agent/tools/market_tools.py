# -*- coding: utf-8 -*-
"""
Market tools — wraps DataFetcherManager market-level methods as agent tools.

Tools:
- get_market_indices: major market index data
- get_sector_rankings: sector performance rankings
"""

import logging
import threading
import time

from src.agent.tools.registry import ToolParameter, ToolDefinition

logger = logging.getLogger(__name__)

# ── 板块排行当日缓存（同一交易日只调一次 API） ──────────────
_sector_cache_lock = threading.Lock()
_sector_cache: dict = {
    "timestamp": 0.0,
    "data": None,
    "top_n": 0,
}
_SECTOR_CACHE_TTL = 3600  # 1 小时过期


def _get_fetcher_manager():
    """Lazy import to avoid circular deps. Returns the shared singleton manager."""
    from data_provider import get_default_manager
    return get_default_manager()


# ============================================================
# get_market_indices
# ============================================================

def _handle_get_market_indices(region: str = "cn") -> dict:
    """Get major market indices."""
    manager = _get_fetcher_manager()
    indices = manager.get_main_indices(region=region)

    if not indices:
        return {"error": f"No market index data available for region '{region}'"}

    return {
        "region": region,
        "indices_count": len(indices),
        "indices": indices,
    }


get_market_indices_tool = ToolDefinition(
    name="get_market_indices",
    description="Get major market indices (e.g., Shanghai Composite, Shenzhen Component, "
                "CSI 300 for China; S&P 500, Nasdaq, Dow for US). Provides market overview.",
    parameters=[
        ToolParameter(
            name="region",
            type="string",
            description="Market region: 'cn' for China A-shares, 'us' for US stocks (default: 'cn')",
            required=False,
            default="cn",
            enum=["cn", "us"],
        ),
    ],
    handler=_handle_get_market_indices,
    category="market",
)


# ============================================================
# get_sector_rankings
# ============================================================

def _handle_get_sector_rankings(top_n: int = 10) -> dict:
    """Get sector performance rankings (with intra-run cache)."""
    global _sector_cache

    now = time.time()
    with _sector_cache_lock:
        cached = _sector_cache
        if cached["data"] is not None and cached["top_n"] >= top_n and (now - cached["timestamp"]) < _SECTOR_CACHE_TTL:
            logger.debug("板块排行命中缓存 (age=%.0fs)", now - cached["timestamp"])
            result = cached["data"]
            # 如果缓存的 top_n 比请求的大，截取前 top_n
            if isinstance(result, tuple) and len(result) == 2:
                top_sectors, bottom_sectors = result
                return {
                    "top_sectors": top_sectors[:top_n],
                    "bottom_sectors": bottom_sectors[:top_n],
                }
            elif isinstance(result, list):
                return {"sectors": result}
            else:
                return {"data": str(result)}

    # 缓存未命中，调用 API
    manager = _get_fetcher_manager()
    result = manager.get_sector_rankings(n=top_n)

    if result is None:
        return {"error": "No sector ranking data available"}

    # 写入缓存
    with _sector_cache_lock:
        _sector_cache = {
            "timestamp": time.time(),
            "data": result,
            "top_n": top_n,
        }

    # get_sector_rankings returns Tuple[List[Dict], List[Dict]]
    # (top_sectors, bottom_sectors)
    if isinstance(result, tuple) and len(result) == 2:
        top_sectors, bottom_sectors = result
        return {
            "top_sectors": top_sectors,
            "bottom_sectors": bottom_sectors,
        }
    elif isinstance(result, list):
        return {"sectors": result}
    else:
        return {"data": str(result)}


get_sector_rankings_tool = ToolDefinition(
    name="get_sector_rankings",
    description="Get sector/industry performance rankings. Returns top N and bottom N "
                "sectors by daily change percentage. Useful for sector rotation analysis.",
    parameters=[
        ToolParameter(
            name="top_n",
            type="integer",
            description="Number of top/bottom sectors to return (default: 10)",
            required=False,
            default=10,
        ),
    ],
    handler=_handle_get_sector_rankings,
    category="market",
)


ALL_MARKET_TOOLS = [
    get_market_indices_tool,
    get_sector_rankings_tool,
]
