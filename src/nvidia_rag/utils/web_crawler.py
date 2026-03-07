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

Delta / cross-run deduplication:
    A URL registry is persisted to ``<registry_dir>/<domain>_url_registry.json``
    after each crawl and loaded at the start of the next.

    * HTML pages: always fetched in full (links must be extracted for BFS).
      SHA-256 of the response body is compared against the stored hash; if
      unchanged the page is skipped for ingest but links are still followed.
      The HTTP ``Last-Modified`` and ``ETag`` headers are stored for future use.

    * Binary files: conditional GET with ``If-None-Match`` (ETag) or
      ``If-Modified-Since``; a ``304 Not Modified`` response skips download
      and ingest entirely.

    Set ``force_recrawl=True`` to ignore the registry and re-ingest everything.

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
import hashlib
import json
import logging
import os
import tempfile
import time
from collections import deque
from concurrent.futures import Future as ConcurrentFuture
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import requests

from nvidia_rag.utils.xml_preprocessor import xml_to_markdown

if TYPE_CHECKING:
    from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor

logger = logging.getLogger(__name__)

# Sentinel returned by _collect_binary_file when server responds 304 Not Modified.
_UNCHANGED: tuple = ()

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


def _sha256(data: bytes) -> str:
    """Return ``sha256:<hex>`` digest of *data*."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class SimpleWebCrawler:
    """
    Streaming-batch BFS web crawler with back-pressure and delta detection.

    Ingest batches are dispatched as files accumulate, but no more than
    ``max_concurrent_batches`` are in-flight at once.  Completed futures are
    harvested eagerly throughout (no per-batch timeout) so nv-ingest results
    are never dropped due to a timeout.

    A URL registry is persisted between runs so that unchanged pages and files
    are skipped automatically.  Use ``force_recrawl=True`` to override.

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
    force_recrawl : bool
        When True, ignore the URL registry and re-ingest all content even if
        unchanged since the last crawl.  Default False.
    registry_dir : str
        Directory where the URL registry JSON and error-matrix CSV are written.
        Default ``/mnt/nvme2``.
    collection_name : str
        Vector-store collection being populated.  When set, it is prepended to
        the domain slug in all artifact filenames so that ``cleanup_crawl_artifacts``
        can find every registry / CSV belonging to that collection regardless of
        which domain(s) were crawled.  E.g. collection ``nvidia`` + domain
        ``nvidia.com`` → ``nvidia_nvidia_com_url_registry.json``.
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
        force_recrawl: bool = False,
        registry_dir: str = "/tmp",
        collection_name: str = "",
        use_nemoretriever_parse: bool = False,
        force_nemoretriever_parse: bool = False,
        request_timeout: int = 30,
        user_agent: str = "NVIDIA-RAG-Crawler/1.0",
        html_chunk_max_tokens: int = 2048,
        export_dir: str = "",
        pdf_repo_dir: str = "",
    ) -> None:
        self.start_url = start_url.rstrip("/")
        self.max_pages = max_pages
        self.extract_linked_files = extract_linked_files
        self.batch_ingest_size = max(1, batch_ingest_size)
        self.max_concurrent_batches = max(1, max_concurrent_batches)
        self.force_recrawl = force_recrawl
        self.registry_dir = registry_dir
        self.collection_name = collection_name
        self.use_nemoretriever_parse = use_nemoretriever_parse
        self.force_nemoretriever_parse = force_nemoretriever_parse
        self.request_timeout = request_timeout
        self._html_chunk_max_tokens = html_chunk_max_tokens
        self.export_dir = export_dir
        self.pdf_repo_dir = pdf_repo_dir

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
        Streaming-batch crawl with back-pressure, eager harvesting, and delta detection.

        Phase 1 -- BFS fetch with rolling dispatch:
            Every ``batch_ingest_size`` files, dispatch an ingest batch.
            If ``max_concurrent_batches`` slots are full, block (poll/sleep)
            until a batch completes before dispatching the next one.
            Unchanged pages/files (per URL registry) are skipped.

        Phase 2 -- Drain:
            Submit the final partial batch, then poll until all in-flight
            futures complete (no timeout -- waits as long as nv-ingest needs).
            Registry is saved after all futures settle.
        """
        visited_html: set[str] = set()
        visited_files: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])

        # All temp file paths -- cleaned up in finally after all futures settle.
        all_temp_files: list[str] = []
        errors: list[dict] = []
        error_matrix: dict[str, list[dict]] = {
            "broken_links": [],
            "missing_files": [],
            "ingest_failures": [],
            "batch_errors": [],
        }
        pages_crawled = 0
        pages_skipped = 0   # unchanged HTML pages (hash match)
        files_skipped = 0   # unchanged binary files (304)
        total_files_dispatched = 0
        files_ingested = 0

        # Current batch being accumulated before dispatch
        pending: list[tuple[str, dict]] = []

        # In-flight ingest futures: (future, batch_number, file_count)
        in_flight: list[tuple[ConcurrentFuture, int, int]] = []
        batch_num = 0

        # URL registry: loaded once at start, saved in finally
        registry: dict[str, dict] = self._load_registry()
        now_iso = datetime.now(timezone.utc).isoformat()

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
            "max_concurrent_batches=%d, force_recrawl=%s, registry_entries=%d)",
            self.start_url, max_pages_display,
            self.batch_ingest_size, self.max_concurrent_batches,
            self.force_recrawl, len(registry),
        )

        # Ensure the collection exists before dispatching any ingest batches.
        try:
            result = ingestor.create_collection(collection_name=collection_name)
            logger.info("Collection '%s': %s", collection_name, result.get("message", result))
        except Exception as exc:
            logger.warning("create_collection('%s') raised: %r — proceeding anyway", collection_name, exc)

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
                html_content, page_title, meta_desc, section_h1, linked_urls, fetch_error, resp_meta = (
                    self._fetch_html(url)
                )

                if html_content is None:
                    if fetch_error:
                        errors.append({"url": url, **fetch_error})
                    # Still process any links if we got a non-HTML content type
                    # (fetch_error is None for silent skips like wrong content type)
                    continue

                pages_crawled += 1

                # ── Delta check for HTML: compare content hash ───────────────
                new_hash = resp_meta.get("content_hash", "")
                reg_entry = registry.get(url, {})
                stored_hash = reg_entry.get("content_hash", "")

                if not self.force_recrawl and stored_hash and stored_hash == new_hash:
                    pages_skipped += 1
                    logger.info(
                        "Skipping unchanged page %d/%s: %s  [hash match, skipped=%d]",
                        pages_crawled, max_pages_display, url, pages_skipped,
                    )
                    # Update last_seen but do NOT update last_ingested
                    registry[url] = {**reg_entry, "last_seen": now_iso}
                else:
                    # New or changed — extract semantic elements, chunk, queue for ingest
                    base_meta = {
                        "source_uri": url,
                        "page_title": page_title,
                        "crawl_depth": depth,
                        "section_h1": section_h1,
                        "meta_description": meta_desc,
                        "source_system": "web_crawl",
                    }
                    html_elements = self._html_to_elements(html_content)
                    if html_elements:
                        from nvidia_rag.ingestor_server.document_classifier_router import (  # noqa: PLC0415
                            DocumentClassifierRouter,
                        )
                        chunk_pairs = DocumentClassifierRouter._split_by_semantic_elements(
                            html_elements, self._html_chunk_max_tokens, chunk_overlap=150,
                        )
                        for chunk_text, section_path in chunk_pairs:
                            if not chunk_text.strip():
                                continue
                            tmp_path = self._save_temp(chunk_text.encode("utf-8"), suffix=".md")
                            all_temp_files.append(tmp_path)
                            meta = {**base_meta}
                            if section_path:
                                meta["section_path"] = section_path
                            pending.append((
                                tmp_path,
                                {"filename": os.path.basename(tmp_path), "metadata": meta},
                            ))
                    else:
                        # Fallback: no elements extracted — save raw HTML
                        tmp_path = self._save_temp(html_content.encode("utf-8"), suffix=".html")
                        all_temp_files.append(tmp_path)
                        pending.append((
                            tmp_path,
                            {"filename": os.path.basename(tmp_path), "metadata": base_meta},
                        ))
                    # Update registry entry
                    registry[url] = {
                        "last_seen": now_iso,
                        "last_ingested": now_iso,
                        "last_modified": resp_meta.get("last_modified"),
                        "etag": resp_meta.get("etag"),
                        "content_hash": new_hash,
                        "status_code": resp_meta.get("status_code", 200),
                    }
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

                            # Build conditional-GET headers from registry
                            file_reg = registry.get(abs_href, {})
                            if_none_match = (
                                file_reg.get("etag")
                                if not self.force_recrawl else None
                            )
                            if_modified_since = (
                                file_reg.get("last_modified")
                                if not self.force_recrawl and not if_none_match else None
                            )

                            entry, file_meta = self._collect_binary_file(
                                abs_href, depth + 1, all_temp_files, errors,
                                if_none_match=if_none_match,
                                if_modified_since=if_modified_since,
                            )
                            if entry is _UNCHANGED:
                                files_skipped += 1
                                # Refresh last_seen
                                registry[abs_href] = {**file_reg, "last_seen": now_iso}
                            elif entry is not None:
                                pending.append(entry)
                                registry[abs_href] = {
                                    "last_seen": now_iso,
                                    "last_ingested": now_iso,
                                    "last_modified": file_meta.get("last_modified"),
                                    "etag": file_meta.get("etag"),
                                    "content_hash": file_meta.get("content_hash"),
                                    "status_code": file_meta.get("status_code", 200),
                                }
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

                # Periodically flush registry + error CSV to disk and export
                # to the host-mounted dir so artifacts are visible mid-crawl.
                if pages_crawled % 100 == 0:
                    self._save_registry(registry)
                    self._flush_errors_to_csv(errors, error_matrix)
                    self._export_crawl_artifacts()

            # ── Phase 2: flush remainder, then drain all in-flight batches ───
            _dispatch_batch(pending)
            pending = []

            if not in_flight and total_files_dispatched == 0:
                return {
                    "message": "Crawl complete: no content collected.",
                    "pages_crawled": pages_crawled,
                    "pages_skipped": pages_skipped,
                    "files_skipped": files_skipped,
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
                "Crawl BFS complete (%d pages, %d skipped-unchanged).  "
                "Draining %d remaining in-flight batch(es)...",
                pages_crawled, pages_skipped, len(in_flight),
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
            # Always persist the updated registry + error CSV and export
            # artifacts — even on interrupt/SIGTERM.
            self._save_registry(registry)
            self._flush_errors_to_csv(errors, error_matrix)
            self._export_crawl_artifacts()

        binary_files = total_files_dispatched - (pages_crawled - pages_skipped)

        return {
            "message": (
                f"Crawl complete: {pages_crawled - pages_skipped} HTML pages and "
                f"{binary_files} linked files ingested "
                f"({pages_skipped} pages and {files_skipped} files unchanged/skipped)."
            ),
            "pages_crawled": pages_crawled,
            "pages_skipped": pages_skipped,
            "files_skipped": files_skipped,
            "files_ingested": binary_files,
            "errors": errors,
            "error_matrix": error_matrix,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _domain_slug(self) -> str:
        """Return a filesystem-safe slug prefixed by collection name (when set).

        Format: ``{collection_name}_{domain}`` or just ``{domain}`` if no
        collection name was given.  Domain processing: strips ``www.`` prefix,
        drops port, replaces ``.`` and ``-`` with ``_``.

        Examples:
            collection=``nvidia``, domain=``www.nvidia.com`` → ``nvidia_nvidia_com``
            collection=`""``,      domain=``www.nvidia.com`` → ``nvidia_com``
        """
        netloc = self._netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        domain = netloc.replace(".", "_").replace("-", "_")
        if self.collection_name:
            coll = self.collection_name.replace(".", "_").replace("-", "_")
            return f"{coll}_{domain}"
        return domain

    def _registry_path(self) -> str:
        return os.path.join(self.registry_dir, f"{self._domain_slug()}_url_registry.json")

    @staticmethod
    def cleanup_crawl_artifacts(
        collection_name: str,
        registry_dir: str = "/tmp",
    ) -> list[str]:
        """Remove URL registry and error-matrix CSV files for a collection.

        Globs ``<registry_dir>/<collection_name>_*_url_registry.json`` and
        ``<registry_dir>/<collection_name>_*_error_matrix.csv`` and deletes
        every match.  Silently skips files that cannot be removed.

        Returns the list of paths that were successfully deleted.
        """
        import glob as _glob

        coll = collection_name.replace(".", "_").replace("-", "_")
        patterns = [
            os.path.join(registry_dir, f"{coll}_*_url_registry.json"),
            os.path.join(registry_dir, f"{coll}_*_error_matrix.csv"),
        ]
        removed: list[str] = []
        for pattern in patterns:
            for path in _glob.glob(pattern):
                try:
                    os.unlink(path)
                    removed.append(path)
                    logger.info("Removed crawl artifact: %s", path)
                except OSError as exc:
                    logger.warning("Could not remove crawl artifact %s: %s", path, exc)
        if not removed:
            logger.info(
                "No crawl artifacts found for collection '%s' in %s",
                collection_name, registry_dir,
            )
        return removed

    def _load_registry(self) -> dict[str, dict]:
        """Load the URL registry from disk; return empty dict if absent or corrupt."""
        path = self._registry_path()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                logger.info("Loaded URL registry from %s (%d entries)", path, len(data))
                return data
        except FileNotFoundError:
            logger.info("No existing URL registry at %s — starting fresh", path)
        except Exception as exc:
            logger.warning("Could not load URL registry from %s: %s — starting fresh", path, exc)
        return {}

    def _save_registry(self, registry: dict[str, dict]) -> None:
        """Atomically write the URL registry to disk."""
        path = self._registry_path()
        tmp_path = path + ".tmp"
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(registry, fh, indent=2)
            os.replace(tmp_path, path)
            logger.info("URL registry saved to %s (%d entries)", path, len(registry))
        except Exception as exc:
            logger.warning("Could not save URL registry to %s: %s", path, exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _fetch_html(
        self, url: str
    ) -> tuple[str | None, str, str, str, list[str], dict | None, dict]:
        """
        Fetch *url*, parse it, and return
        ``(html_text, title, meta_desc, h1, hrefs, fetch_error, response_meta)``.

        ``fetch_error`` is ``None`` on success, or a dict with keys
        ``error_type``, ``status_code``, and ``error`` on failure.
        Returns ``(None, ..., None, {})`` for non-HTML content (silently skipped).

        ``response_meta`` contains ``last_modified``, ``etag``,
        ``content_hash`` (SHA-256 of body), and ``status_code``.
        """
        empty_meta: dict = {}
        try:
            from bs4 import BeautifulSoup  # lazy import
        except ImportError:
            logger.error("beautifulsoup4 is not installed; cannot crawl HTML pages")
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": None,
                "error": "beautifulsoup4 not installed",
            }, empty_meta

        try:
            resp = self._session.get(url, timeout=self.request_timeout)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug("Skipping non-HTML URL %s (Content-Type: %s)", url, content_type)
                return None, "", "", "", [], None, empty_meta  # silently skip
            html_text = resp.text
            resp_meta: dict = {
                "last_modified": resp.headers.get("Last-Modified"),
                "etag": resp.headers.get("ETag"),
                "content_hash": _sha256(html_text.encode("utf-8")),
                "status_code": resp.status_code,
            }
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": status_code,
                "error": str(exc),
            }, empty_meta
        except Exception as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None, "", "", "", [], {
                "error_type": "broken_link", "status_code": None,
                "error": str(exc),
            }, empty_meta

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
            }, resp_meta

        return html_text, title, meta_desc, section_h1, hrefs, None, resp_meta

    def _collect_binary_file(
        self,
        url: str,
        depth: int,
        all_temp_files: list[str],
        errors: list[dict],
        if_none_match: str | None = None,
        if_modified_since: str | None = None,
    ) -> tuple[tuple[str, dict] | tuple | None, dict]:
        """
        Download *url* to a temp file and return ``(entry, response_meta)``.

        ``entry`` is one of:
        * ``(tmp_path, metadata_dict)`` — new or changed file, add to batch.
        * ``_UNCHANGED`` (empty tuple sentinel) — server returned 304, skip.
        * ``None`` — download or pre-processing failed, error logged.

        ``response_meta`` carries ``last_modified``, ``etag``,
        ``content_hash``, and ``status_code`` for registry updates.
        XML files are pre-processed to Markdown before collection.
        """
        suffix = Path(urlparse(url).path).suffix or ".bin"
        empty_meta: dict = {}

        # Build conditional-GET headers
        headers: dict[str, str] = {}
        if if_none_match:
            headers["If-None-Match"] = if_none_match
        elif if_modified_since:
            headers["If-Modified-Since"] = if_modified_since

        try:
            resp = self._session.get(
                url, stream=True, timeout=self.request_timeout, headers=headers
            )
            if resp.status_code == 304:
                logger.debug("304 Not Modified (unchanged): %s", url)
                return _UNCHANGED, empty_meta
            resp.raise_for_status()
            # PDFs go to the persistent repo dir (if configured) so they are
            # retained after ingest and accessible on the host filesystem.
            if self.pdf_repo_dir and suffix.lower() == ".pdf":
                dest_dir = Path(self.pdf_repo_dir) / self.collection_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                filename = Path(urlparse(url).path).name or f"download{suffix}"
                tmp_path = str(dest_dir / filename)
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
                logger.debug("Downloaded PDF to persistent repo: %s", tmp_path)
            else:
                tmp_path = self._save_temp_stream(resp, suffix=suffix)
                all_temp_files.append(tmp_path)
            # Compute hash from the downloaded file
            with open(tmp_path, "rb") as fh:
                content_hash = _sha256(fh.read())
            file_meta: dict = {
                "last_modified": resp.headers.get("Last-Modified"),
                "etag": resp.headers.get("ETag"),
                "content_hash": content_hash,
                "status_code": resp.status_code,
            }
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({
                "url": url,
                "error_type": "missing_file",
                "status_code": status_code,
                "error": str(exc),
            })
            return None, empty_meta
        except Exception as exc:
            logger.warning("Failed to download binary file %s: %s", url, exc)
            errors.append({
                "url": url,
                "error_type": "missing_file",
                "status_code": None,
                "error": str(exc),
            })
            return None, empty_meta

        # XML files are not natively supported by nv-ingest -- pre-process to Markdown.
        if suffix.lower() == ".xml":
            try:
                with open(tmp_path, "rb") as fh:
                    xml_bytes = fh.read()
                markdown_text = xml_to_markdown(xml_bytes)
                if not markdown_text.strip():
                    logger.warning("XML pre-processor produced no content for %s -- skipping", url)
                    return None, empty_meta
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
                return None, empty_meta

        entry = (
            tmp_path,
            {
                "filename": os.path.basename(tmp_path),
                "metadata": {
                    "source_uri": url,
                    "crawl_depth": depth,
                    "source_system": "web_crawl",
                },
            },
        )
        return entry, file_meta

    def _flush_errors_to_csv(
        self,
        errors: list[dict],
        error_matrix: dict[str, list[dict]],
    ) -> None:
        """Rebuild error_matrix from errors and write the CSV.

        Called both mid-crawl (every 100 pages) and in the finally block so
        the host-exported CSV is always up to date.  error_matrix is mutated
        in-place so the finally block can use it without reprocessing.
        """
        # Clear and rebuild so we don't accumulate duplicates across calls.
        for key in ("broken_links", "missing_files", "ingest_failures", "batch_errors"):
            error_matrix[key] = []
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

    def _write_error_matrix_csv(
        self,
        error_matrix: dict[str, list[dict]],
    ) -> None:
        """Write the error matrix to ``<registry_dir>/<domain>_error_matrix.csv``.

        Columns: category, url, status_code, error
        One row per error entry across all categories.  Existing file is
        overwritten.  Silently skips if the output directory is not writable.
        """
        csv_path = os.path.join(self.registry_dir, f"{self._domain_slug()}_error_matrix.csv")
        total_errors = sum(len(v) for v in error_matrix.values())
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
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

    def _export_crawl_artifacts(self) -> None:
        """
        Copy the URL registry JSON and error-matrix CSV to ``self.export_dir``
        at the end of a crawl so they are accessible outside the pod.

        No-op when ``export_dir`` is empty or the source files do not exist.
        """
        if not self.export_dir:
            return
        import shutil
        slug = self._domain_slug()
        sources = [
            os.path.join(self.registry_dir, f"{slug}_url_registry.json"),
            os.path.join(self.registry_dir, f"{slug}_error_matrix.csv"),
        ]
        try:
            os.makedirs(self.export_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create export_dir '%s': %s", self.export_dir, exc)
            return
        for src in sources:
            if not os.path.exists(src):
                continue
            dst = os.path.join(self.export_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                logger.info("Exported crawl artifact: %s → %s", src, dst)
            except OSError as exc:
                logger.warning("Failed to export '%s' to '%s': %s", src, dst, exc)

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

    @staticmethod
    def _html_table_to_markdown(table_tag: Any) -> str:
        """Convert a BeautifulSoup ``<table>`` tag to GitHub-Flavored Markdown."""
        rows: list[str] = []
        for tr in table_tag.find_all("tr"):
            cells = [
                cell.get_text(separator=" ", strip=True).replace("|", "\\|")
                for cell in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if not rows:
            return ""
        header = rows[0]
        col_count = max(1, header.count("|") - 1)
        separator = "| " + " | ".join(["---"] * col_count) + " |"
        return "\n".join([header, separator] + rows[1:])

    @staticmethod
    def _html_to_elements(html_str: str) -> list[tuple[str, str]]:
        """
        Convert raw HTML to ``(class_name, text)`` element pairs using the same
        taxonomy as nemoretriever-parse (Title, Section-header, Text, List-item,
        Table, Formula, Caption).

        Navigation containers (``nav``, ``header``, ``footer``, ``aside``,
        ``script``, ``style``) are stripped before extraction.  The parser
        walks the document tree recursively so nested elements are handled
        naturally without double-counting.
        """
        try:
            from bs4 import BeautifulSoup, Tag  # lazy import — always available
        except ImportError:
            logger.warning("beautifulsoup4 not available; HTML semantic chunking skipped")
            return []

        soup = BeautifulSoup(html_str, "html.parser")

        # Strip noise containers before walking the tree
        for noise_tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
            noise_tag.decompose()

        # Prefer semantic main-content container; fall back to body / root
        main: Any = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", id="content")
            or soup.find("body")
            or soup
        )

        elements: list[tuple[str, str]] = []

        def _walk(node: Any, in_list: bool = False) -> None:
            for child in node.children:
                if not isinstance(child, Tag):
                    continue  # skip NavigableString / Comment
                name = child.name

                if name == "h1":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("Title", t))

                elif name in ("h2", "h3", "h4", "h5", "h6"):
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("Section-header", t))

                elif name == "p":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("Text", t))

                elif name in ("ul", "ol"):
                    _walk(child, in_list=True)

                elif name == "li":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("List-item", t))

                elif name == "table":
                    md = SimpleWebCrawler._html_table_to_markdown(child)
                    if md:
                        elements.append(("Table", md))

                elif name == "pre":
                    t = child.get_text(strip=True)
                    if t:
                        elements.append(("Formula", t))

                elif name == "figcaption":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("Caption", t))

                elif name == "blockquote":
                    t = child.get_text(separator=" ", strip=True)
                    if t:
                        elements.append(("Text", t))

                elif name in (
                    "div", "section", "article", "main", "figure",
                    "details", "summary", "form",
                ):
                    # Container — recurse without consuming
                    _walk(child, in_list=in_list)

                # Inline tags (span, a, strong, em, code, …) are intentionally
                # skipped here; their text is already captured by the parent
                # block tag's get_text() call.

        _walk(main)
        return elements
