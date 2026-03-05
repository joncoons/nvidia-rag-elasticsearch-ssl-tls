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
Streaming-batch BFS web crawler for the NVIDIA RAG ingestor server.

Crawling and ingestion are pipelined: as soon as ``batch_ingest_size``
files have been collected the crawler dispatches an ingest batch
asynchronously and immediately continues crawling.  Back-pressure is
applied via ``max_concurrent_batches`` so that at most N ingest batches
are in-flight simultaneously, preventing nv-ingest from being overwhelmed.
Completed futures are harvested eagerly (no per-batch timeout) so results
are never lost due to a timeout firing before nv-ingest finishes.

Phase 1 -- Crawl + rolling ingest dispatch:
    BFS-traverse the domain up to max_pages, fetching HTML pages and
    downloading linked binary files.  Every ``batch_ingest_size`` files a
    non-blocking ingest batch is dispatched; if ``max_concurrent_batches``
    slots are already full the dispatch point blocks (polling) until one
    completes.  Completed futures are harvested opportunistically throughout.

Phase 2 -- Drain: after the BFS loop ends the remaining files (if any) are
    dispatched as a final batch, then the loop polls until all in-flight
    futures complete and aggregates the results before returning.

Supported linked-file types (requires extract_linked_files=True):
  Documents : PDF, DOCX, XLSX, PPTX, DOC, XLS
  Text/MD   : .md, .txt
  Images    : PNG, JPG/JPEG, BMP, TIFF
  Audio     : WAV, MP3
  XML       : RSS 2.0, Atom 1.0, Sitemap, generic -- pre-processed to Markdown
  Video     : (none -- mp4/avi/mkv/mov lack an nv-ingest extractor)

Usage::

    from nvidia_rag.utils.web_crawler import SimpleWebCrawler

    crawler = SimpleWebCrawler(
        start_url="https://docs.nvidia.com/cuda/",
        max_pages=200,
        extract_linked_files=True,
        batch_ingest_size=20,
        max_concurrent_batches=3,
    )
    result = await crawler.crawl(ingestor, collection_name="nvidia-docs")
"""

import asyncio
import csv
import logging
import os
import tempfile
import time
from collections import deque
from concurrent.futures import Future as ConcurrentFuture
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
        # XML -- pre-processed to Markdown before ingestion via xml_preprocessor
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
        return True  # relative URL -- keep
    return parsed.netloc == netloc or parsed.netloc.endswith("." + netloc)


class SimpleWebCrawler:
    """
    Streaming-batch BFS web crawler with back-pressure.

    Ingest batches are dispatched as files accumulate, but no more than
    ``max_concurrent_batches`` are in-flight at once.  Completed futures are
    harvested eagerly throughout (no per-batch timeout) so nv-ingest results
    are never dropped due to a timeout.

    Parameters
    ----------
    start_url : str
        URL to begin crawling from.
    max_pages : int or None
        Maximum number of HTML pages to crawl (binary file downloads are not
        counted against this limit).  ``None`` means unlimited.
    extract_linked_files : bool
        When True, ``<a href>`` links pointing to supported binary files
        (documents, images, audio, XML, markdown) are downloaded and
        ingested in addition to HTML pages.  XML files are automatically
        pre-processed to Markdown via ``xml_preprocessor.xml_to_markdown``.
    batch_ingest_size : int
        Number of files that triggers an ingest batch dispatch.  Default 20.
    max_concurrent_batches : int
        Maximum number of ``upload_documents()`` calls in-flight at the same
        time.  Keeps nv-ingest from being overwhelmed on large crawls.
        Default 3.
    use_nemoretriever_parse : bool
        Forwarded to ``upload_documents()`` for every batch.
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
        max_pages: int | None = 50,
        extract_linked_files: bool = False,
        batch_ingest_size: int = 20,
        max_concurrent_batches: int = 3,
        use_nemoretriever_parse: bool = False,
        force_nemoretriever_parse: bool = False,
        request_timeout: int = 30,
        user_agent: str = "NVIDIA-RAG-Crawler/1.0",
    ) -> None:
        self.start_url = start_url.rstrip("/")
        self.max_pages = max_pages
        self.extract_linked_files = extract_linked_files
        self.batch_ingest_size = max(1, batch_ingest_size)
        self.max_concurrent_batches = max(1, max_concurrent_batches)
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
        Streaming-batch crawl with back-pressure and eager harvesting.

        Phase 1 -- BFS fetch with rolling dispatch:
            Every ``batch_ingest_size`` files, dispatch an ingest batch.
            If ``max_concurrent_batches`` slots are full, block (poll/sleep)
            until a batch completes before dispatching the next one.

        Phase 2 -- Drain:
            Submit the final partial batch, then poll until all in-flight
            futures complete (no timeout -- waits as long as nv-ingest needs).
        """
        visited_html: set[str] = set()
        visited_files: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])

        # All temp file paths -- cleaned up in finally after all futures settle.
        all_temp_files: list[str] = []
        errors: list[dict] = []
        pages_crawled = 0
        total_files_dispatched = 0
        files_ingested = 0

        # Current batch being accumulated before dispatch
        pending: list[tuple[str, dict]] = []

        # In-flight ingest futures: (future, batch_number, file_count)
        in_flight: list[tuple[ConcurrentFuture, int, int]] = []
        batch_num = 0

        def _harvest_done() -> None:
            """Move any completed futures out of in_flight, record results."""
            nonlocal files_ingested
            remaining: list[tuple[ConcurrentFuture, int, int]] = []
            for f, bnum, fcount in in_flight:
                if not f.done():
                    remaining.append((f, bnum, fcount))
                    continue
                try:
                    result = f.result()
                    failed_docs = (
                        result.get("failed_documents", [])
                        if isinstance(result, dict) else []
                    )
                    for fd in failed_docs:
                        errors.append({
                            "url": fd.get("document_name", "unknown"),
                            "error_type": "ingest_failure",
                            "status_code": None,
                            "error": fd.get("error", "ingest failed"),
                        })
                    files_ingested += fcount - len(failed_docs)
                    logger.info(
                        "Ingest batch %d complete (%d files, %d failed)",
                        bnum, fcount, len(failed_docs),
                    )
                except Exception as exc:
                    logger.error("Ingest batch %d failed: %r", bnum, exc)
                    errors.append({
                        "url": f"batch_{bnum}",
                        "error_type": "batch_error",
                        "status_code": None,
                        "error": repr(exc),
                    })
            in_flight[:] = remaining

        def _dispatch_batch(batch: list[tuple[str, dict]]) -> None:
            """Submit *batch* to upload_documents(), blocking if at capacity."""
            nonlocal batch_num, total_files_dispatched
            if not batch:
                return
            # Back-pressure: wait until a concurrent slot is free
            while len(in_flight) >= self.max_concurrent_batches:
                time.sleep(0.5)
                _harvest_done()
            batch_num += 1
            filepaths = [p for p, _ in batch]
            custom_metadata = [m for _, m in batch]
            logger.info(
                "Dispatching ingest batch %d: %d files  [%d/%d slots used]",
                batch_num, len(filepaths), len(in_flight), self.max_concurrent_batches,
            )
            future: ConcurrentFuture = asyncio.run_coroutine_threadsafe(
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
            in_flight.append((future, batch_num, len(filepaths)))
            total_files_dispatched += len(filepaths)

        max_pages_display = self.max_pages if self.max_pages is not None else "unlimited"
        logger.info(
            "Crawl starting at %s (max_pages=%s, batch_ingest_size=%d, "
            "max_concurrent_batches=%d)",
            self.start_url, max_pages_display,
            self.batch_ingest_size, self.max_concurrent_batches,
        )

        try:
            # ── Phase 1: BFS crawl with rolling batch dispatch ───────────────
            while queue and (self.max_pages is None or pages_crawled < self.max_pages):
                url, depth = queue.popleft()
                if url in visited_html:
                    continue
                visited_html.add(url)

                # Opportunistically harvest completed futures while crawling
                _harvest_done()

                logger.info("Crawling [depth=%d] %s", depth, url)
                html_content, page_title, meta_desc, section_h1, linked_urls, fetch_error = (
                    self._fetch_html(url)
                )
                if html_content is None:
                    if fetch_error:
                        errors.append({"url": url, **fetch_error})
                    continue

                tmp_path = self._save_temp(html_content.encode("utf-8"), suffix=".html")
                all_temp_files.append(tmp_path)
                pending.append((
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
                logger.info(
                    "Collected page %d/%s: %s  [pending=%d, in_flight=%d]",
                    pages_crawled, max_pages_display, url, len(pending), len(in_flight),
                )

                # Collect any linked binary files
                for href in linked_urls:
                    abs_href = urljoin(url, href)
                    if self.extract_linked_files and _is_binary_url(abs_href):
                        if abs_href not in visited_files:
                            visited_files.add(abs_href)
                            entry = self._collect_binary_file(
                                abs_href, depth + 1, all_temp_files, errors
                            )
                            if entry is not None:
                                pending.append(entry)
                    elif (
                        not _is_binary_url(abs_href)
                        and _same_domain(abs_href, self._netloc)
                        and abs_href not in visited_html
                        and (self.max_pages is None or pages_crawled < self.max_pages)
                    ):
                        queue.append((abs_href, depth + 1))

                # Dispatch a batch when threshold is reached
                if len(pending) >= self.batch_ingest_size:
                    _dispatch_batch(pending)
                    pending = []

            # ── Phase 2: flush remainder, then drain all in-flight batches ───
            _dispatch_batch(pending)  # back-pressure applies here too
            pending = []

            if not in_flight and total_files_dispatched == 0:
                return {
                    "message": "Crawl complete: no content collected.",
                    "pages_crawled": pages_crawled,
                    "files_ingested": 0,
                    "errors": errors,
                    "error_matrix": {
                        "broken_links": [],
                        "missing_files": [],
                        "ingest_failures": [],
                        "batch_errors": [],
                    },
                }

            logger.info(
                "Crawl BFS complete (%d pages).  Draining %d remaining in-flight batch(es)...",
                pages_crawled, len(in_flight),
            )

            # Poll until all in-flight futures complete (no timeout).
            while in_flight:
                time.sleep(2)
                _harvest_done()

        finally:
            for tp in all_temp_files:
                try:
                    os.unlink(tp)
                except OSError:
                    pass

        binary_files = total_files_dispatched - pages_crawled

        # Build structured error matrix grouped by category
        error_matrix: dict[str, list[dict]] = {
            "broken_links": [],
            "missing_files": [],
            "ingest_failures": [],
            "batch_errors": [],
        }
        for e in errors:
            etype = e.get("error_type", "other")
            entry = {k: v for k, v in e.items() if k != "error_type"}
            if etype == "broken_link":
                error_matrix["broken_links"].append(entry)
            elif etype == "missing_file":
                error_matrix["missing_files"].append(entry)
            elif etype == "ingest_failure":
                error_matrix["ingest_failures"].append(entry)
            elif etype == "batch_error":
                error_matrix["batch_errors"].append(entry)
            else:
                error_matrix.setdefault("other", []).append(entry)

        self._write_error_matrix_csv(error_matrix)

        return {
            "message": (
                f"Crawl complete: {pages_crawled} HTML pages and "
                f"{binary_files} linked files ingested."
            ),
            "pages_crawled": pages_crawled,
            "files_ingested": binary_files,
            "errors": errors,
            "error_matrix": error_matrix,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_html(
        self, url: str
    ) -> tuple[str | None, str, str, str, list[str], dict | None]:
        """
        Fetch *url*, parse it, and return
        ``(html_text, title, meta_desc, h1, hrefs, fetch_error)``.

        ``fetch_error`` is ``None`` on success or a dict with keys
        ``error_type``, ``status_code``, and ``error`` on failure.
        Returns ``(None, "", "", "", [], None)`` for non-HTML content (silently skipped).
        """
        try:
            from bs4 import BeautifulSoup  # lazy import
        except ImportError:
            logger.error("beautifulsoup4 is not installed; cannot crawl HTML pages")
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": None,
                "error": "beautifulsoup4 not installed",
            }

        try:
            resp = self._session.get(url, timeout=self.request_timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug("Skipping non-HTML URL %s (Content-Type: %s)", url, content_type)
                return None, "", "", "", [], None  # Not an error — silently skip
            html_text = resp.text
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": status_code,
                "error": str(exc),
            }
        except Exception as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": None,
                "error": str(exc),
            }

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
            return html_text, "", "", "", [], {
                "error_type": "broken_link", "status_code": None,
                "error": f"parse error: {exc}",
            }

        return html_text, title, meta_desc, section_h1, hrefs, None

    def _collect_binary_file(
        self,
        url: str,
        depth: int,
        all_temp_files: list[str],
        errors: list[dict],
    ) -> tuple[str, dict] | None:
        """
        Download *url* to a temp file and return ``(tmp_path, metadata_entry)``
        for inclusion in the next ingest batch, or ``None`` on failure.

        XML files are pre-processed to Markdown before collection.
        """
        suffix = Path(urlparse(url).path).suffix or ".bin"
        try:
            resp = self._session.get(url, stream=True, timeout=self.request_timeout)
            resp.raise_for_status()
            tmp_path = self._save_temp_stream(resp, suffix=suffix)
            all_temp_files.append(tmp_path)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({
                "url": url,
                "error_type": "missing_file",
                "status_code": status_code,
                "error": str(exc),
            })
            return None
        except Exception as exc:
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({
                "url": url,
                "error_type": "missing_file",
                "status_code": None,
                "error": str(exc),
            })
            return None

        # XML files are not natively supported by nv-ingest -- pre-process to Markdown.
        if suffix.lower() == ".xml":
            try:
                with open(tmp_path, "rb") as fh:
                    xml_bytes = fh.read()
                markdown_text = xml_to_markdown(xml_bytes)
                if not markdown_text.strip():
                    logger.warning("XML pre-processor produced no content for %s -- skipping", url)
                    return None
                md_path = self._save_temp(markdown_text.encode("utf-8"), suffix=".md")
                all_temp_files.append(md_path)
                tmp_path = md_path
                logger.info("XML pre-processed to Markdown (%d chars): %s", len(markdown_text), url)
            except Exception as exc:
                logger.warning("Failed to pre-process XML %s: %s", url, exc)
                errors.append({
                    "url": url,
                    "error_type": "missing_file",
                    "status_code": None,
                    "error": f"XML pre-process failed: {exc}",
                })
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

    def _write_error_matrix_csv(
        self,
        error_matrix: dict[str, list[dict]],
        output_dir: str = "/mnt/nvme2",
    ) -> None:
        """Write the error matrix to ``<output_dir>/<domain>_error_matrix.csv``.

        Columns: category, url, status_code, error
        One row per error entry across all categories.  Existing file is
        overwritten.  Silently skips if the output directory is not writable.
        """
        # Strip www. prefix, remove port, replace . and - with _
        netloc = self._netloc.split(":")[0]  # drop port
        if netloc.startswith("www."):
            netloc = netloc[4:]
        domain = netloc.replace(".", "_").replace("-", "_")
        csv_path = os.path.join(output_dir, f"{domain}_error_matrix.csv")
        total_errors = sum(len(v) for v in error_matrix.values())
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["category", "url", "status_code", "error"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                for category, entries in error_matrix.items():
                    for entry in entries:
                        writer.writerow({
                            "category": category,
                            "url": entry.get("url", ""),
                            "status_code": entry.get("status_code", ""),
                            "error": entry.get("error", ""),
                        })
            logger.info(
                "Error matrix written to %s (%d entries)",
                csv_path, total_errors,
            )
        except Exception as exc:
            logger.warning("Could not write error matrix CSV to %s: %s", csv_path, exc)

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
