"""Tavily web search fallback for the RAG pipeline.

Returns LangChain Document objects so results slot directly into
context_to_show alongside the normal VDB-retrieved documents.
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import certifi
import httpx
from langchain_core.documents import Document

if TYPE_CHECKING:
    from nvidia_rag.utils.configuration import TavilyConfig

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


async def _call_tavily(
    api_key: str,
    query: str,
    max_results: int,
    *,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Single Tavily API call returning raw result dicts.

    Uses certifi's CA bundle explicitly via ``httpx.AsyncClient(verify=...)``
    to bypass the ``SSL_CERT_FILE`` environment variable, which is set to the
    ECK-internal CA cert for Elasticsearch TLS and would otherwise cause
    certificate-verification failures for public HTTPS endpoints.
    """
    payload: dict = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    async with httpx.AsyncClient(
        verify=certifi.where(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=timeout,
    ) as client:
        resp = await client.post(_TAVILY_URL, json=payload)
        resp.raise_for_status()
        return resp.json().get("results", [])


async def search_tavily(query: str, config: "TavilyConfig") -> list[Document]:
    """Search the web via Tavily and return results as LangChain Documents.

    Calls the Tavily REST API directly with httpx (certifi CA bundle) so that
    the ECK SSL_CERT_FILE override does not interfere.  Results are filtered by
    relevance score and converted to Documents with source metadata so they can
    be cited normally by the LLM chain.

    Returns an empty list on timeout or any other error — never raises.
    """
    if not config.api_key:
        logger.warning("Tavily search skipped: no API key configured")
        return []

    api_key = config.api_key.get_secret_value()

    # Parse comma-separated domain allowlist (empty string → unrestricted)
    include_domains: list[str] = (
        [d.strip() for d in config.include_domains.split(",") if d.strip()]
        if config.include_domains
        else []
    )

    all_results: list[dict] = []

    try:
        if include_domains:
            # Query each chunk of ≤5 domains separately (Tavily limit)
            for i in range(0, len(include_domains), 5):
                chunk = include_domains[i : i + 5]
                try:
                    async with asyncio.timeout(35):
                        all_results.extend(
                            await _call_tavily(
                                api_key, query, config.max_results,
                                include_domains=chunk,
                            )
                        )
                except asyncio.TimeoutError:
                    logger.warning("Tavily timeout for domains %s", chunk)
        else:
            # Two rounds with domain exclusion for result diversity
            seen_domains: list[str] = []
            for round_num in range(2):
                try:
                    async with asyncio.timeout(35):
                        results = await _call_tavily(
                            api_key, query, config.max_results,
                            exclude_domains=seen_domains or None,
                        )
                        all_results.extend(results)
                        for r in results:
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
            if not isinstance(r, dict):
                continue
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
