"""Tavily web search fallback for the RAG pipeline.

Returns LangChain Document objects so results slot directly into
context_to_show alongside the normal VDB-retrieved documents.
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from langchain_core.documents import Document

if TYPE_CHECKING:
    from nvidia_rag.utils.configuration import TavilyConfig

logger = logging.getLogger(__name__)


async def search_tavily(query: str, config: "TavilyConfig") -> list[Document]:
    """Search the web via Tavily and return results as LangChain Documents.

    The API key is injected into the process environment before calling
    TavilySearchResults (which reads TAVILY_API_KEY from env).  Results are
    filtered by relevance score and converted to Documents with source
    metadata so they can be cited normally by the LLM chain.

    Returns an empty list on timeout or any other error — never raises.
    """
    from langchain_community.tools import TavilySearchResults  # noqa: PLC0415

    # Inject API key into env — TavilySearchResults reads it from there
    if config.api_key:
        os.environ["TAVILY_API_KEY"] = config.api_key.get_secret_value()

    # Parse comma-separated domain allowlist (empty string → unrestricted)
    include_domains: list[str] = [
        d.strip() for d in config.include_domains.split(",") if d.strip()
    ] if config.include_domains else []

    all_results: list[dict] = []

    try:
        if include_domains:
            # Query each chunk of ≤5 domains separately (Tavily limit)
            for i in range(0, len(include_domains), 5):
                chunk = include_domains[i : i + 5]
                tool = TavilySearchResults(
                    max_results=config.max_results,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False,
                    include_domains=chunk,
                )
                try:
                    async with asyncio.timeout(30):
                        all_results.extend(await tool.ainvoke({"query": query}))
                except asyncio.TimeoutError:
                    logger.warning("Tavily timeout for domains %s", chunk)
        else:
            # Two rounds with domain exclusion for result diversity
            seen_domains: list[str] = []
            for round_num in range(2):
                tool = TavilySearchResults(
                    max_results=config.max_results,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False,
                    exclude_domains=seen_domains,
                )
                try:
                    async with asyncio.timeout(30):
                        round_results = await tool.ainvoke({"query": query})
                        all_results.extend(round_results)
                        for r in round_results:
                            try:
                                netloc = urlparse(r.get("url", "")).netloc
                                if netloc:
                                    seen_domains.append(netloc)
                            except Exception:
                                pass
                except asyncio.TimeoutError:
                    logger.warning("Tavily timeout on round %d", round_num + 1)

        # Filter by relevance score and convert to Documents
        docs: list[Document] = []
        for r in all_results:
            content = r.get("content", "")
            url = r.get("url", "")
            raw_score = r.get("score")
            try:
                score_float = float(raw_score) if raw_score is not None else 1.0
            except (TypeError, ValueError):
                score_float = 0.0

            if score_float >= config.score_threshold and content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "source_uri": url,
                            "source_system": "tavily_web_search",
                            "score": score_float,
                        },
                    )
                )

        logger.info(
            "Tavily search for '%s...' returned %d documents (threshold=%.2f)",
            query[:80],
            len(docs),
            config.score_threshold,
        )
        return docs

    except Exception as exc:
        logger.warning("Tavily search failed for query '%s...': %s", query[:80], exc)
        return []
