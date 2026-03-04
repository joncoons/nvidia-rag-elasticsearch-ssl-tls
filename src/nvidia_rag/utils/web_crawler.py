# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-phase BFS web crawler for the NVIDIA RAG ingestor server.

Phase 1 — Crawl: BFS-traverse the domain up to max_pages, fetching HTML
pages and downloading linked binary files.  All content is saved to temp
files; nothing is ingested yet.

Phase 2 — Batch ingest: Pass all collected temp files to a single
upload_documents() call so the nv-ingest batching machinery can process
them in parallel (concurrent_batches × files_per_batch).

This is significantly faster than the previous per-page serial approach for
large crawls because fetch latency and ingest latency are fully overlapped.

Supported linked-file types (requires extract_linked_files=True):
  Documents : PDF, DOCX, XLSX, PPTX, DOC, XLS
  Text/MD   : .md, .txt
  Images    : PNG, JPG/JPEG, BMP, TIFF
  Audio     : WAV, MP3
  XML       : RSS 2.0, Atom 1.0, Sitemap, generic — pre-processed to Markdown
  Video     : (none — mp4/avi/mkv/mov lack an nv-ingest extractor)

Usage::

    from nvidia_rag.utils.web_crawler import SimpleWebCrawler

    crawler = SimpleWebCrawler(
        start_url="https://docs.nvidia.com/cuda/",
        max_pages=50,
        extract_linked_files=True,
    )
    result = await crawler.crawl(ingestor, collection_name="nvidia-docs")
"""

import asyncio
import logging
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import requests

from nvidia_rag.utils.xml_preprocessor import xml_to_markdown

if TYPE_CHECKING:
    from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor

logger = logging.getLogger(__name__)

# File extensions considered binary / document files (not crawled as HTML).
# These are downloaded and ingested when extract_linked_files=True.
# Only extensions supported by nv-ingest are included.
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Documents
        ".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls",
        # Markdown / plain text
        ".md", ".txt",
        # Images
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
        # Audio
        ".wav", ".mp3",
        # XML — pre-processed to Markdown before ingestion via xml_preprocessor
        ".xml",
        # Video excluded: avi/mkv/mov/mp4 have no nv-ingest extractor registered
    }
)


def _is_binary_url(url: str) -> bool:
    """Return True if *url* points to a binary document file."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _BINARY_EXTENSIONS)


def _same_domain(url: str, netloc: str) -> bool:
    """Return True if *url* belongs to *netloc* (exact or subdomain)."""
    parsed = urlparse(url)
    if not parsed.netloc:
        return True  # relative URL — keep
    return parsed.netloc == netloc or parsed.netloc.endswith("." + netloc)


class SimpleWebCrawler:
    """
    Two-phase BFS web crawler: crawl entire domain first, then batch-ingest.

    Phase 1 collects all HTML pages and linked binary files into temp files.
    Phase 2 passes them all to a single upload_documents() call so nv-ingest
    can process them in parallel batches rather than one at a time.

    Parameters
    ----------
    start_url : str
        URL to begin crawling from.
    max_pages : int
        Maximum number of HTML pages to crawl (binary file downloads are not
        counted against this limit).
    extract_linked_files : bool
        When True, ``<a href>`` links pointing to supported binary files
        (documents, images, audio, XML, markdown) are downloaded and
        ingested in addition to HTML pages.  XML files are automatically
        pre-processed to Markdown via ``xml_preprocessor.xml_to_markdown``.
    use_nemoretriever_parse : bool
        Forwarded to ``upload_documents()`` for the batch ingest.
    force_nemoretriever_parse : bool
        Forwarded to ``upload_documents()``; implies ``use_nemoretriever_parse``.
    request_timeout : int
        HTTP request timeout in seconds for each fetch (default 30).
    user_agent : str
        User-Agent header sent with every request.
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = 50,
        extract_linked_files: bool = False,
        use_nemoretriever_parse: bool = False,
        force_nemoretriever_parse: bool = False,
        request_timeout: int = 30,
        user_agent: str = "NVIDIA-RAG-Crawler/1.0",
    ) -> None:
        self.start_url = start_url.rstrip("/")
        self.max_pages = max_pages
        self.extract_linked_files = extract_linked_files
        self.use_nemoretriever_parse = use_nemoretriever_parse
        self.force_nemoretriever_parse = force_nemoretriever_parse
        self.request_timeout = request_timeout

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        # Disable environment variable passthrough so that REQUESTS_CA_BUNDLE
        # (which points to the ECK-internal CA) does not override standard CA
        # validation for outbound public HTTPS requests.  With trust_env=False
        # requests uses the certifi bundle it ships with (verify=True default).
        self._session.trust_env = False
        self._netloc = urlparse(start_url).netloc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(
        self,
        ingestor: "NvidiaRAGIngestor",
        collection_name: str,
        vdb_auth_token: str = "",
    ) -> dict[str, Any]:
        """
        BFS-crawl the site and batch-upload all discovered content.

        Parameters
        ----------
        ingestor : NvidiaRAGIngestor
            Ingestor instance used to upload files.
        collection_name : str
            Target vector-store collection.
        vdb_auth_token : str
            Optional bearer token forwarded to the vector store.

        Returns
        -------
        dict
            Summary dict compatible with ``UploadDocumentResponse``.
        """
        # Capture the running event loop BEFORE entering the executor thread.
        # The thread will use run_coroutine_threadsafe to submit coroutines
        # back to this loop — the correct pattern for calling async code from
        # a worker thread while an event loop is already running.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._crawl_sync,
            ingestor,
            collection_name,
            vdb_auth_token,
            loop,
        )

    # ------------------------------------------------------------------
    # Internal sync implementation (runs in a thread-pool executor)
    # ------------------------------------------------------------------

    def _crawl_sync(
        self,
        ingestor: "NvidiaRAGIngestor",
        collection_name: str,
        vdb_auth_token: str,
        loop: asyncio.AbstractEventLoop,
    ) -> dict[str, Any]:
        """
        Two-phase crawl:
          Phase 1 — BFS fetch, save all pages/files to temp files.
          Phase 2 — Single batch upload_documents() call for all collected files.
        """
        # ── Phase 1: BFS crawl ──────────────────────────────────────────────
        visited_html: set[str] = set()
        visited_files: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])

        # (tmp_path, metadata_entry) — one entry per file to ingest
        collected: list[tuple[str, dict]] = []
        temp_files: list[str] = []
        errors: list[dict] = []
        pages_crawled = 0

        logger.info(
            "Phase 1: BFS crawl starting at %s (max_pages=%d)", self.start_url, self.max_pages
        )

        try:
            while queue and pages_crawled < self.max_pages:
                url, depth = queue.popleft()
                if url in visited_html:
                    continue
                visited_html.add(url)

                logger.info("Crawling [depth=%d] %s", depth, url)
                html_content, page_title, meta_desc, section_h1, linked_urls = (
                    self._fetch_html(url)
                )
                if html_content is None:
                    errors.append({"url": url, "error": "fetch failed"})
                    continue

                tmp_path = self._save_temp(html_content.encode("utf-8"), suffix=".html")
                temp_files.append(tmp_path)
                collected.append((
                    tmp_path,
                    {
                        "filename": os.path.basename(tmp_path),
                        "metadata": {
                            "source_url": url,
                            "page_title": page_title,
                            "crawl_depth": depth,
                            "section_h1": section_h1,
                            "meta_description": meta_desc,
                            "source_system": "web_crawl",
                        },
                    },
                ))
                pages_crawled += 1
                logger.info("Collected page %d/%d: %s", pages_crawled, self.max_pages, url)

                # Enqueue discovered links
                for href in linked_urls:
                    abs_href = urljoin(url, href)
                    if self.extract_linked_files and _is_binary_url(abs_href):
                        if abs_href not in visited_files:
                            visited_files.add(abs_href)
                            entry = self._collect_binary_file(
                                abs_href, depth + 1, temp_files, errors
                            )
                            if entry is not None:
                                collected.append(entry)
                    elif (
                        not _is_binary_url(abs_href)
                        and _same_domain(abs_href, self._netloc)
                        and abs_href not in visited_html
                        and pages_crawled < self.max_pages
                    ):
                        queue.append((abs_href, depth + 1))

            files_collected = len(collected) - pages_crawled  # binary files only
            logger.info(
                "Phase 1 complete: %d HTML pages + %d binary files collected",
                pages_crawled, files_collected,
            )

            if not collected:
                return {
                    "message": "Crawl complete: no content collected.",
                    "pages_crawled": 0,
                    "files_ingested": 0,
                    "errors": errors,
                }

            # ── Phase 2: Batch ingest ────────────────────────────────────────
            logger.info(
                "Phase 2: batch-ingesting %d files into collection '%s'",
                len(collected), collection_name,
            )

            filepaths = [p for p, _ in collected]
            custom_metadata = [m for _, m in collected]

            # Allow 30 s per file, floor of 10 minutes
            ingest_timeout = max(600, len(filepaths) * 30)

            try:
                future = asyncio.run_coroutine_threadsafe(
                    ingestor.upload_documents(
                        filepaths=filepaths,
                        collection_name=collection_name,
                        vdb_auth_token=vdb_auth_token,
                        blocking=True,
                        custom_metadata=custom_metadata,
                        use_nemoretriever_parse=self.use_nemoretriever_parse,
                        force_nemoretriever_parse=self.force_nemoretriever_parse,
                        source_system="web_crawl",
                    ),
                    loop,
                )
                result = future.result(timeout=ingest_timeout)
                # Surface any per-document failures returned by the ingestor
                failed_docs = result.get("failed_documents", []) if isinstance(result, dict) else []
                for f in failed_docs:
                    errors.append({
                        "url": f.get("document_name", "unknown"),
                        "error": f.get("error", "ingest failed"),
                    })
                logger.info("Phase 2 complete: %d files ingested", len(filepaths))
            except Exception as exc:
                logger.error("Batch ingest failed: %s", exc)
                errors.append({"url": "batch_ingest", "error": str(exc)})

        finally:
            for tp in temp_files:
                try:
                    os.unlink(tp)
                except OSError:
                    pass

        return {
            "message": (
                f"Crawl complete: {pages_crawled} HTML pages and "
                f"{files_collected} linked files ingested."
            ),
            "pages_crawled": pages_crawled,
            "files_ingested": files_collected,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_html(
        self, url: str
    ) -> tuple[str | None, str, str, str, list[str]]:
        """
        Fetch *url*, parse it, and return
        ``(html_text, title, meta_desc, h1, hrefs)``.

        Returns ``(None, "", "", "", [])`` on error.
        """
        try:
            from bs4 import BeautifulSoup  # lazy import
        except ImportError:
            logger.error("beautifulsoup4 is not installed; cannot crawl HTML pages")
            return None, "", "", "", []

        try:
            resp = self._session.get(url, timeout=self.request_timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug("Skipping non-HTML URL %s (Content-Type: %s)", url, content_type)
                return None, "", "", "", []
            html_text = resp.text
        except Exception as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None, "", "", "", []

        try:
            soup = BeautifulSoup(html_text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            meta_desc = ""
            if meta_tag and isinstance(meta_tag, object):
                meta_desc = str(meta_tag.get("content", "")).strip()
            h1_tag = soup.find("h1")
            section_h1 = h1_tag.get_text(strip=True) if h1_tag else ""
            hrefs = [
                str(a.get("href", ""))
                for a in soup.find_all("a", href=True)
                if a.get("href")
            ]
        except Exception as exc:
            logger.warning("Parse error for %s: %s", url, exc)
            return html_text, "", "", "", []

        return html_text, title, meta_desc, section_h1, hrefs

    def _collect_binary_file(
        self,
        url: str,
        depth: int,
        temp_files: list[str],
        errors: list[dict],
    ) -> tuple[str, dict] | None:
        """
        Download *url* to a temp file and return ``(tmp_path, metadata_entry)``
        for inclusion in the Phase 2 batch, or ``None`` on failure.

        XML files are pre-processed to Markdown before collection.
        """
        suffix = Path(urlparse(url).path).suffix or ".bin"
        try:
            resp = self._session.get(url, stream=True, timeout=self.request_timeout)
            resp.raise_for_status()
            tmp_path = self._save_temp_stream(resp, suffix=suffix)
            temp_files.append(tmp_path)
        except Exception as exc:
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({"url": url, "error": str(exc)})
            return None

        # XML files are not natively supported by nv-ingest — pre-process to Markdown.
        if suffix.lower() == ".xml":
            try:
                with open(tmp_path, "rb") as fh:
                    xml_bytes = fh.read()
                markdown_text = xml_to_markdown(xml_bytes)
                if not markdown_text.strip():
                    logger.warning("XML pre-processor produced no content for %s — skipping", url)
                    return None
                md_path = self._save_temp(markdown_text.encode("utf-8"), suffix=".md")
                temp_files.append(md_path)
                tmp_path = md_path
                logger.info("XML pre-processed to Markdown (%d chars): %s", len(markdown_text), url)
            except Exception as exc:
                logger.warning("Failed to pre-process XML %s: %s", url, exc)
                errors.append({"url": url, "error": str(exc)})
                return None

        return (
            tmp_path,
            {
                "filename": os.path.basename(tmp_path),
                "metadata": {
                    "source_url": url,
                    "crawl_depth": depth,
                    "source_system": "web_crawl",
                },
            },
        )

    @staticmethod
    def _save_temp(data: bytes, suffix: str = ".html") -> str:
        """Write *data* to a named temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="webcrawl_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    @staticmethod
    def _save_temp_stream(resp: requests.Response, suffix: str = ".bin") -> str:
        """Stream *resp* body to a named temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="webcrawl_")
        with os.fdopen(fd, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
        return path
