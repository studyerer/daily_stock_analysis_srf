# -*- coding: utf-8 -*-
"""
Search tools — wraps SearchService methods as agent-callable tools.

Tools:
- search_stock_news: search latest stock news
- search_comprehensive_intel: multi-dimensional intelligence search
"""

import logging
import threading
from typing import Optional, Any

from src.agent.tools.registry import ToolParameter, ToolDefinition

logger = logging.getLogger(__name__)


# ============================================================
# 新闻搜索结果捕获（避免 Pipeline 在 Agent 之后重复搜一次）
# ============================================================
# 思路：Agent 调用 search_stock_news 时把 SearchResponse 缓存到这里，
# Pipeline 在 Agent 完成后读取并保存到 DB；找不到时再退化到 Pipeline 自行搜索。
_news_captures: dict = {}
_news_captures_lock = threading.Lock()


def capture_news_response(stock_code: str, response: Any) -> None:
    """供 search_stock_news tool 调用：记录最近一次成功的 SearchResponse。"""
    if not stock_code or response is None:
        return
    with _news_captures_lock:
        _news_captures[stock_code] = response


def get_captured_news_response(stock_code: str) -> Optional[Any]:
    """供 Pipeline 调用：读取并清空指定股票的捕获结果。"""
    with _news_captures_lock:
        return _news_captures.pop(stock_code, None)


def clear_captured_news_response(stock_code: str) -> None:
    """供 Pipeline 在 Agent 运行前清理可能的旧结果（防止跨股票串味）。"""
    with _news_captures_lock:
        _news_captures.pop(stock_code, None)


def _get_search_service():
    """Lazy-init SearchService with config keys."""
    from src.search_service import SearchService
    from src.config import get_config
    config = get_config()
    return SearchService(
        bocha_keys=config.bocha_api_keys,
        tavily_keys=config.tavily_api_keys,
        brave_keys=config.brave_api_keys,
        serpapi_keys=config.serpapi_keys,
        news_max_age_days=config.news_max_age_days,
    )


def _handle_search_stock_news(stock_code: str, stock_name: str) -> dict:
    """Search latest news for a stock."""
    service = _get_search_service()

    if not service.is_available:
        return {"error": "No search engine available (no API keys configured)"}

    response = service.search_stock_news(stock_code, stock_name, max_results=5)

    if not response.success:
        return {
            "query": response.query,
            "success": False,
            "error": response.error_message,
        }

    # 捕获原始响应对象，让 Pipeline 不再重复搜索一次
    if response.results:
        capture_news_response(stock_code, response)

    return {
        "query": response.query,
        "provider": response.provider,
        "success": True,
        "results_count": len(response.results),
        "results": [
            {
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
                "source": r.source,
                "published_date": r.published_date,
            }
            for r in response.results
        ],
    }


search_stock_news_tool = ToolDefinition(
    name="search_stock_news",
    description="Search for the latest news articles about a specific stock. "
                "Requires both stock_code and stock_name for accurate search. "
                "Returns news titles, snippets, sources, and URLs.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
        ToolParameter(
            name="stock_name",
            type="string",
            description="Stock name in Chinese, e.g., '贵州茅台'",
        ),
    ],
    handler=_handle_search_stock_news,
    category="search",
)


# ============================================================
# search_comprehensive_intel
# ============================================================

def _handle_search_comprehensive_intel(stock_code: str, stock_name: str) -> dict:
    """Multi-dimensional intelligence search."""
    service = _get_search_service()

    if not service.is_available:
        return {"error": "No search engine available (no API keys configured)"}

    intel_results = service.search_comprehensive_intel(
        stock_code=stock_code,
        stock_name=stock_name,
        max_searches=5,
    )

    if not intel_results:
        return {"error": "Comprehensive intel search returned no results"}

    # Format into readable report
    report = service.format_intel_report(intel_results, stock_name)

    # Also return structured data
    dimensions = {}
    for dim_name, response in intel_results.items():
        if response and response.success:
            dimensions[dim_name] = {
                "query": response.query,
                "results_count": len(response.results),
                "results": [
                    {
                        "title": r.title,
                        "snippet": r.snippet,
                        "source": r.source,
                    }
                    for r in response.results[:3]  # limit to 3 per dimension to save tokens
                ],
            }

    return {
        "report": report,
        "dimensions": dimensions,
    }


search_comprehensive_intel_tool = ToolDefinition(
    name="search_comprehensive_intel",
    description="Multi-dimensional intelligence search: latest news, market analysis, "
                "risk checking, earnings outlook, and industry trends for a stock. "
                "Returns a formatted report and structured results.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
        ToolParameter(
            name="stock_name",
            type="string",
            description="Stock name in Chinese, e.g., '贵州茅台'",
        ),
    ],
    handler=_handle_search_comprehensive_intel,
    category="search",
)


ALL_SEARCH_TOOLS = [
    search_stock_news_tool,
    search_comprehensive_intel_tool,
]
