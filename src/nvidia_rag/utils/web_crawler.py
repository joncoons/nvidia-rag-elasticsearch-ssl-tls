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
Simple BFS web crawler for the NVIDIA RAG ingestor server.

Crawls HTML pages within a single domain, saves them as temporary files,
and uploads them to the vector store via NvidiaRAGIngestor.upload_documents().
Optionally downloads linked binary files (PDF, DOCX, XLSX, PPTX) found as
<a href> targets and ingests those as well.

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

if TYPE_CHECKING:
    from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor

logger = logging.getLogger(__name__)

# File extensions considered binary / document files (not crawled as HTML)
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls"}
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
    BFS web crawler that ingests HTML pages and optionally linked binary files.

    Parameters
    ----------
    start_url : str
        URL to begin crawling from.
    max_pages : int
        Maximum number of HTML pages to crawl (binary file downloads are not
        counted against this limit).
    extract_linked_files : bool
        When True, ``<a href>`` links pointing to PDF/DOCX/XLSX/PPTX files are
        downloaded and ingested in addition to HTML pages.
    use_nemoretriever_parse : bool
        Forwarded to ``upload_documents()`` for each ingested file.
    force_nemoretriever_parse : bool
        Forwarded to ``upload_documents()``; implies ``use_nemoretriever_parse``.
    request_timeout : int
        HTTP request timeout in seconds (default 30).
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
        BFS-crawl the site and upload discovered content.

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
        visited_html: set[str] = set()
        visited_files: set[str] = set()
        queue: deque[tuple[str, int]] = deque()  # (url, depth)
        queue.append((self.start_url, 0))

        pages_crawled = 0
        files_ingested = 0
        errors: list[dict] = []

        temp_files: list[str] = []

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

                # Save HTML to temp file and ingest
                tmp_path = self._save_temp(html_content.encode("utf-8"), suffix=".html")
                temp_files.append(tmp_path)
                custom_metadata = [
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
                    }
                ]
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        ingestor.upload_documents(
                            filepaths=[tmp_path],
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
                    future.result(timeout=300)
                    pages_crawled += 1
                    logger.info("Ingested HTML page %d/%d: %s", pages_crawled, self.max_pages, url)
                except Exception as exc:
                    logger.warning("Failed to ingest page %s: %s", url, exc)
                    errors.append({"url": url, "error": str(exc)})

                # Enqueue discovered links
                for href in linked_urls:
                    abs_href = urljoin(url, href)
                    if self.extract_linked_files and _is_binary_url(abs_href):
                        if abs_href not in visited_files:
                            visited_files.add(abs_href)
                            self._ingest_binary_file(
                                abs_href, depth + 1, ingestor, collection_name,
                                vdb_auth_token, loop, temp_files, errors
                            )
                            files_ingested += 1
                    elif (
                        not _is_binary_url(abs_href)
                        and _same_domain(abs_href, self._netloc)
                        and abs_href not in visited_html
                        and pages_crawled < self.max_pages
                    ):
                        queue.append((abs_href, depth + 1))


        finally:
            for tp in temp_files:
                try:
                    os.unlink(tp)
                except OSError:
                    pass

        return {
            "message": (
                f"Crawl complete: {pages_crawled} HTML pages and "
                f"{files_ingested} linked files ingested."
            ),
            "pages_crawled": pages_crawled,
            "files_ingested": files_ingested,
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

    def _ingest_binary_file(
        self,
        url: str,
        depth: int,
        ingestor: "NvidiaRAGIngestor",
        collection_name: str,
        vdb_auth_token: str,
        loop: asyncio.AbstractEventLoop,
        temp_files: list[str],
        errors: list[dict],
    ) -> None:
        """Download *url* as a binary file and upload it to the vector store."""
        suffix = Path(urlparse(url).path).suffix or ".bin"
        try:
            resp = self._session.get(url, stream=True, timeout=self.request_timeout)
            resp.raise_for_status()
            tmp_path = self._save_temp_stream(resp, suffix=suffix)
            temp_files.append(tmp_path)
        except Exception as exc:
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({"url": url, "error": str(exc)})
            return

        custom_metadata = [
            {
                "filename": os.path.basename(tmp_path),
                "metadata": {
                    "source_url": url,
                    "crawl_depth": depth,
                    "source_system": "web_crawl",
                },
            }
        ]
        try:
            future = asyncio.run_coroutine_threadsafe(
                ingestor.upload_documents(
                    filepaths=[tmp_path],
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
            future.result(timeout=300)
            logger.info("Ingested binary file: %s", url)
        except Exception as exc:
            logger.warning("Failed to ingest binary file %s: %s", url, exc)
            errors.append({"url": url, "error": str(exc)})

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
