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

from nvidia_rag.utils.k8s_scaler import disable_crawl_mode, enable_crawl_mode
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
    allowed_url_prefixes : list[str] or None
        When set, BFS only follows links whose URL starts with one of these
        prefixes.  The start_url is always visited regardless.  Use this to
        restrict a crawl on a shared domain (e.g. github.com) to specific
        organisation/repository paths without crawling the whole site.
        Example: ``["https://github.com/NVIDIA-AI-Blueprints",
                    "https://github.com/NVIDIA-NeMo/NeMo"]``
    use_selenium : bool
        When True, pages whose static HTML yields fewer than
        ``selenium_content_threshold`` visible characters are re-fetched via a
        headless Chromium browser so that JavaScript-rendered content is
        captured.  Requires ``selenium`` and a system ``chromium`` / Chrome
        installation (provided by the ingestor container).  Default False.
    selenium_content_threshold : int
        Minimum number of visible text characters that must be extracted from
        the static HTML before Selenium rendering is triggered.  Default 300.
    selenium_screenshot_fallback : bool
        When True *and* ``use_selenium=True``, pages that are still sparse
        after Selenium rendering are captured as a full-page JPEG screenshot
        and added to the ingest batch (so they can be routed through
        Nemotron-Parse when ``use_nemoretriever_parse=True``).  Default False.
    selenium_wait_timeout : int
        Seconds Selenium waits for a page to become non-empty after navigation.
        Default 15.
    max_depth : int or None
        Maximum BFS depth from the start URL.  Depth 0 is the start page,
        depth 1 is pages linked from it, and so on.  ``None`` means unlimited.
        Recommended values: 2 for GitHub (org → repo → top-level files),
        5–10 for standard documentation sites.  Default ``None``.
    blocked_url_patterns : list[str] or None
        URL substrings that cause a link to be skipped entirely — neither
        fetched as HTML nor collected as a binary file.  Matching is a simple
        ``in`` check against the full URL string so partial path segments work.
        Example for GitHub: ``["/stargazers", "/forks", "/commits/",
        "/blame/", "/graphs/", "/actions", "/issues", "/pull/"]``
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
        allowed_url_prefixes: list[str] | None = None,
        use_selenium: bool = False,
        selenium_content_threshold: int = 300,
        selenium_screenshot_fallback: bool = False,
        selenium_wait_timeout: int = 15,
        max_depth: int | None = None,
        blocked_url_patterns: list[str] | None = None,
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
        self.allowed_url_prefixes = [p.rstrip("/") for p in allowed_url_prefixes] if allowed_url_prefixes else None
        self.max_depth = max_depth
        self.blocked_url_patterns = blocked_url_patterns or []
        self.use_selenium = use_selenium
        self._selenium_content_threshold = selenium_content_threshold
        self._selenium_screenshot_fallback = selenium_screenshot_fallback
        self._selenium_wait_timeout = selenium_wait_timeout
        self._user_agent = user_agent
        # Shared Selenium driver — created once per crawl in _crawl_sync if
        # use_selenium=True, then reused for every JS page to avoid per-page
        # Chrome startup cost (~5-10 s).  Quit in the crawl finally block.
        self._selenium_driver: Any = None

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

        # In-flight ingest futures:
        #   (future, batch_number, file_count, urls_to_mark_ingested, uri_to_chunk_names)
        # urls_to_mark_ingested: source_uris whose last_ingested should be set on success
        #   (changed URLs whose last_ingested was cleared before dispatch).
        # uri_to_chunk_names: source_uri → [dispatched file basenames] for ALL URLs in
        #   the batch, stored in the registry so backfill scripts can reverse-map chunks
        #   to their source URLs without a re-crawl.
        in_flight: list[tuple[ConcurrentFuture, int, int, set[str], dict[str, list[str]]]] = []
        batch_num = 0

        # URL registry: loaded once at start, saved in finally
        registry: dict[str, dict] = self._load_registry()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Track URLs whose content changed since the last crawl so that stale
        # vector chunks can be deleted from ES before new chunks are written.
        # last_ingested is cleared for these URLs before dispatch and only restored
        # after the batch completes successfully (atomic upsert semantics).
        changed_urls: set[str] = set()

        # Track previously-ingested URLs that returned 404/410 during this crawl.
        # Their ES chunks and registry entries are purged in the finally block.
        deleted_urls: set[str] = set()

        def _harvest_done() -> None:
            """Move any completed futures out of in_flight, record results."""
            nonlocal files_ingested
            remaining: list[tuple[ConcurrentFuture, int, int, set[str], dict[str, list[str]]]] = []
            for f, bnum, fcount, urls_to_mark, uri_to_chunk_names in in_flight:
                if not f.done():
                    remaining.append((f, bnum, fcount, urls_to_mark, uri_to_chunk_names))
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
                    completed_iso = datetime.now(timezone.utc).isoformat()
                    # Store chunk_doc_names for ALL URLs in this batch so a backfill
                    # script can reverse-map chunk filenames → source URLs without
                    # needing a force re-crawl.
                    for uri, chunk_names in uri_to_chunk_names.items():
                        if uri in registry:
                            registry[uri] = {**registry[uri], "chunk_doc_names": chunk_names}
                    # Restore last_ingested for changed URLs now that new chunks
                    # are confirmed written to ES.
                    for uri in urls_to_mark:
                        if uri in registry:
                            registry[uri] = {**registry[uri], "last_ingested": completed_iso}
                    if uri_to_chunk_names or urls_to_mark:
                        self._save_registry(registry)
                        self._export_crawl_artifacts()
                    logger.info(
                        "Ingest batch %d complete (%d files, %d failed, "
                        "%d URLs chunk_doc_names stored, %d URLs re-marked ingested)",
                        bnum, fcount, len(failed_docs),
                        len(uri_to_chunk_names), len(urls_to_mark),
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
            # Build source_uri → [dispatched file basenames] for registry tracking.
            # Stored in registry after batch success so backfill scripts can
            # reverse-map chunk doc_names → source URLs without a force re-crawl.
            uri_to_chunk_names: dict[str, list[str]] = {}
            for fp, m in batch:
                uri = m.get("metadata", {}).get("source_uri", "")
                if uri:
                    uri_to_chunk_names.setdefault(uri, []).append(os.path.basename(fp))
            # Collect source URIs in this batch that had stale chunks in ES
            # (content changed since last crawl) so the ingestor can delete them first.
            uris_to_delete = list({
                m["metadata"]["source_uri"]
                for _, m in batch
                if m.get("metadata", {}).get("source_uri") in changed_urls
            })
            if uris_to_delete:
                # Atomic pre-delete: clear last_ingested from registry NOW (before ES
                # delete + re-ingest) so registry and ES stay in sync.  If the re-ingest
                # fails, both registry and ES show the URL as not-ingested and it will
                # be retried on the next crawl.  last_ingested is restored in
                # _harvest_done() once the batch future resolves successfully.
                for uri in uris_to_delete:
                    if uri in registry:
                        entry = dict(registry[uri])
                        entry.pop("last_ingested", None)
                        registry[uri] = entry
                self._save_registry(registry)
                self._flush_errors_to_csv(errors, error_matrix)
                self._export_crawl_artifacts()
                logger.info(
                    "Batch %d: cleared last_ingested for %d changed URL(s) — "
                    "will pre-delete stale ES chunks and restore after success",
                    batch_num, len(uris_to_delete),
                )
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
                    is_final_batch=False,
                    source_uris_to_delete=uris_to_delete or None,
                ),
                loop,
            )
            in_flight.append((future, batch_num, len(filepaths), set(uris_to_delete), uri_to_chunk_names))
            total_files_dispatched += len(filepaths)
            # Flush registry + error CSV after every batch so artifacts are
            # up-to-date on the host-mounted dir mid-crawl.
            self._save_registry(registry)
            self._flush_errors_to_csv(errors, error_matrix)
            self._export_crawl_artifacts()

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

        # Switch to crawl-optimised GPU layout (nim-llm off, max nemotron-parse replicas).
        enable_crawl_mode()

        # Initialise the shared Selenium driver once for the whole crawl so that
        # JS-heavy pages don't each pay the ~5-10 s Chrome startup cost.
        if self.use_selenium:
            self._selenium_driver = self._create_selenium_driver()

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
                        # Detect deleted pages: 404/410 on a previously-ingested URL.
                        # Queue for ES chunk purge + registry removal in finally block.
                        if fetch_error.get("status_code") in (404, 410):
                            reg_entry = registry.get(url, {})
                            if reg_entry.get("last_ingested"):
                                deleted_urls.add(url)
                                logger.info(
                                    "Deleted URL detected (HTTP %s): %s — "
                                    "will purge ES chunks and remove from registry",
                                    fetch_error["status_code"], url,
                                )
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
                    # Track changed (not new) URLs so stale ES chunks can be deleted
                    # before the new chunks are written (upsert semantics).
                    is_changed = bool(reg_entry.get("last_ingested") and stored_hash and stored_hash != new_hash)
                    if is_changed:
                        changed_urls.add(url)
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
                        # No semantic elements extracted — try screenshot before raw HTML.
                        # If Selenium screenshot fallback is enabled and Selenium is active,
                        # capture a full-page JPEG so Nemotron-Parse can read visual content.
                        screenshot_added = False
                        if self.use_selenium and self._selenium_screenshot_fallback:
                            ss_dir = tempfile.gettempdir()
                            ss_path = self._capture_screenshot(url, ss_dir)
                            if ss_path:
                                all_temp_files.append(ss_path)
                                pending.append((
                                    ss_path,
                                    {"filename": os.path.basename(ss_path), "metadata": base_meta},
                                ))
                                logger.info(
                                    "Screenshot added to batch for JS-sparse page: %s", url
                                )
                                screenshot_added = True
                        if not screenshot_added:
                            # Final fallback: save raw HTML
                            tmp_path = self._save_temp(html_content.encode("utf-8"), suffix=".html")
                            all_temp_files.append(tmp_path)
                            pending.append((
                                tmp_path,
                                {"filename": os.path.basename(tmp_path), "metadata": base_meta},
                            ))
                    # Update registry entry.  For changed URLs, omit last_ingested here —
                    # it is cleared atomically before dispatch and restored by _harvest_done
                    # only after the new chunks are confirmed written to ES.
                    # For new URLs, set last_ingested optimistically (no old chunks to worry about).
                    reg_update = {
                        "last_seen": now_iso,
                        "last_modified": resp_meta.get("last_modified"),
                        "etag": resp_meta.get("etag"),
                        "content_hash": new_hash,
                        "status_code": resp_meta.get("status_code", 200),
                    }
                    if not is_changed:
                        reg_update["last_ingested"] = now_iso
                    registry[url] = reg_update
                    logger.info(
                        "Collected page %d/%s: %s  [pending=%d, in_flight=%d]",
                        pages_crawled, max_pages_display, url, len(pending), len(in_flight),
                    )

                # Collect any linked binary files
                for href in linked_urls:
                    # Strip fragment (#...) before resolving — same-page anchors
                    # produce duplicate registry entries otherwise (e.g. page#section).
                    href = href.split("#")[0]
                    if not href:
                        continue
                    abs_href = urljoin(url, href)
                    if self._is_blocked_url(abs_href):
                        continue
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
                            elif entry is None:
                                # Fetch failed — check for 404/410 on a previously-ingested file
                                if file_meta.get("status_code") in (404, 410) and file_reg.get("last_ingested"):
                                    deleted_urls.add(abs_href)
                                    logger.info(
                                        "Deleted binary file detected (HTTP %s): %s — "
                                        "will purge ES chunks and remove from registry",
                                        file_meta["status_code"], abs_href,
                                    )
                            else:
                                # Track re-ingested binary files for stale chunk deletion.
                                # Omit last_ingested for changed files; it is restored by
                                # _harvest_done after successful ingest (atomic upsert).
                                file_is_changed = bool(file_reg.get("last_ingested"))
                                if file_is_changed:
                                    changed_urls.add(abs_href)
                                pending.append(entry)
                                file_reg_update = {
                                    "last_seen": now_iso,
                                    "last_modified": file_meta.get("last_modified"),
                                    "etag": file_meta.get("etag"),
                                    "content_hash": file_meta.get("content_hash"),
                                    "status_code": file_meta.get("status_code", 200),
                                }
                                if not file_is_changed:
                                    file_reg_update["last_ingested"] = now_iso
                                registry[abs_href] = file_reg_update
                    elif (
                        not _is_binary_url(abs_href)
                        and _same_domain(abs_href, self._netloc)
                        and self._is_allowed_url(abs_href)
                        and not self._is_blocked_url(abs_href)
                        and abs_href not in visited_html
                        and (self.max_pages is None or pages_crawled < self.max_pages)
                        and (self.max_depth is None or depth + 1 <= self.max_depth)
                    ):
                        queue.append((abs_href, depth + 1))

                # Dispatch a batch when threshold is reached
                if len(pending) >= self.batch_ingest_size:
                    _dispatch_batch(pending)
                    pending = []


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

            # Quit the shared Selenium driver if one was created.
            if self._selenium_driver is not None:
                try:
                    self._selenium_driver.quit()
                except Exception:
                    pass
                self._selenium_driver = None

            # Purge ES chunks + registry entries for URLs that returned 404/410.
            if deleted_urls:
                logger.info(
                    "Purging %d deleted URL(s) from ES and registry...", len(deleted_urls)
                )
                try:
                    purge_future = asyncio.run_coroutine_threadsafe(
                        ingestor.purge_deleted_urls(
                            collection_name=collection_name,
                            source_uris=list(deleted_urls),
                            vdb_auth_token=vdb_auth_token,
                        ),
                        loop,
                    )
                    purge_future.result(timeout=120)
                except Exception as exc:
                    logger.warning("ES purge of deleted URLs failed: %s", exc)
                # Remove deleted URLs from registry regardless of ES purge outcome.
                for uri in deleted_urls:
                    registry.pop(uri, None)
                logger.info(
                    "Removed %d deleted URL(s) from registry", len(deleted_urls)
                )

            # Always persist the updated registry + error CSV and export
            # artifacts — even on interrupt/SIGTERM.
            self._save_registry(registry)
            self._flush_errors_to_csv(errors, error_matrix)
            self._export_crawl_artifacts()

            # Restore inference GPU layout in a background thread (non-blocking).
            disable_crawl_mode()

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

    def _collection_slug(self) -> str:
        """Return a filesystem-safe slug based on collection name.

        Uses the collection name when set, otherwise falls back to the domain.
        Replaces ``.`` and ``-`` with ``_``.

        Examples:
            collection=``nvidia``, domain=``www.nvidia.com`` → ``nvidia``
            collection=`""``,      domain=``www.nvidia.com`` → ``nvidia_com``
        """
        if self.collection_name:
            return self.collection_name.replace(".", "_").replace("-", "_")
        netloc = self._netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.replace(".", "_").replace("-", "_")

    def _registry_path(self) -> str:
        return os.path.join(self.registry_dir, f"{self._collection_slug()}_url_registry.json")

    def _is_blocked_url(self, url: str) -> bool:
        """Return True if *url* contains any of the ``blocked_url_patterns`` substrings.

        Matching is a simple ``in`` check so partial path segments work, e.g.
        ``"/stargazers"`` blocks ``https://github.com/org/repo/stargazers``.
        """
        return any(pat in url for pat in self.blocked_url_patterns)

    def _is_allowed_url(self, url: str) -> bool:
        """Return True if *url* passes the ``allowed_url_prefixes`` filter.

        When ``allowed_url_prefixes`` is ``None`` (the default), every URL on
        the same domain is allowed.  When a prefix list is set, only URLs that
        start with at least one listed prefix are queued for crawling.

        The start_url itself is always visited regardless of this filter.
        """
        if not self.allowed_url_prefixes:
            return True
        for prefix in self.allowed_url_prefixes:
            # Exact match OR URL continues with '/', '?', or '#' after the prefix
            # so that "github.com/nvidia" does not match "github.com/nvidia-cosmos".
            if url == prefix or url.startswith(prefix + "/") \
                    or url.startswith(prefix + "?") or url.startswith(prefix + "#"):
                return True
        return False

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
            os.path.join(registry_dir, f"{coll}_url_registry.json"),
            os.path.join(registry_dir, f"{coll}_error_matrix.csv"),
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

        # ── Selenium JS-rendering fallback ───────────────────────────────────
        # If the static HTML is sparse (JS-rendered SPA) and Selenium is
        # enabled, re-fetch with headless Chromium so BS4 sees rendered content.
        if self.use_selenium and self._is_js_sparse(html_text):
            logger.info("Static HTML sparse — retrying with Selenium: %s", url)
            rendered_html = self._render_with_selenium(url)
            if rendered_html:
                html_text = rendered_html
                resp_meta["content_hash"] = _sha256(html_text.encode("utf-8"))
                logger.debug("Selenium rendered %d chars for %s", len(html_text), url)
            else:
                logger.warning("Selenium render failed for %s — using static HTML", url)

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

    def _create_selenium_driver(self) -> Any:
        """Create and return a configured headless Chromium WebDriver.

        Called once per crawl when ``use_selenium=True``.  The driver is
        stored on ``self._selenium_driver`` and reused across all JS pages so
        that the ~5-10 s Chrome startup cost is paid only once.

        Chrome binary is resolved from the environment variable
        ``CHROMIUM_BIN`` (default: ``/usr/bin/chromium``).  A per-crawl
        user-data-dir is created in a temp directory to prevent profile lock
        issues on restart.
        """
        try:
            from selenium import webdriver  # noqa: PLC0415
            from selenium.webdriver.chrome.options import Options  # noqa: PLC0415
        except ImportError:
            logger.warning("selenium is not installed; use_selenium has no effect")
            return None

        # Google Chrome stable (Ubuntu 22.04 base image); override with CHROMIUM_BIN if needed.
        chromium_bin = os.environ.get("CHROMIUM_BIN", "/usr/bin/google-chrome-stable")
        if not os.path.exists(chromium_bin):
            chromium_bin = "/usr/bin/google-chrome"  # fallback symlink
        user_data_dir = tempfile.mkdtemp(prefix="crawler-chrome-")

        opts = Options()
        opts.binary_location = chromium_bin
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(f"--user-data-dir={user_data_dir}")
        opts.add_argument(f"--user-agent={self._user_agent}")

        try:
            driver = webdriver.Chrome(options=opts)
            driver.implicitly_wait(2)
            driver.set_page_load_timeout(self._selenium_wait_timeout + 10)
            logger.info("Selenium shared driver initialised (chromium: %s)", chromium_bin)
            return driver
        except Exception as exc:
            logger.warning("Failed to start Selenium driver: %s", exc)
            return None

    def _is_js_sparse(self, html_text: str) -> bool:
        """Return True if *html_text* likely needs JS rendering.

        Two independent heuristics (either one triggers):
        1. Visible text (scripts/styles stripped) is below the configured
           threshold — classic SPA shell with minimal server-side content.
        2. Body contains script/noscript tags but has very few real elements
           (≤ 4 total tags) — the ``<div id="root"></div>`` SPA pattern.
        """
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
            soup = BeautifulSoup(html_text, "html.parser")
            # Heuristic 2: SPA root pattern — script-heavy body with almost no elements
            if soup.body:
                has_js_tags = bool(soup.body.find_all(["script", "noscript"]))
                total_elements = len(soup.body.find_all(True))
                if has_js_tags and total_elements <= 4:
                    return True
            # Heuristic 1: thin visible text after stripping noise tags
            for tag in soup(["script", "style", "noscript", "head"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return len(text) < self._selenium_content_threshold
        except Exception:
            return False

    def _render_with_selenium(self, url: str) -> str | None:
        """Navigate the shared Selenium driver to *url* and return rendered HTML.

        Reuses ``self._selenium_driver`` (created once per crawl) so Chrome
        startup cost is not paid per page.  Returns None on failure.
        """
        driver = self._selenium_driver
        if driver is None:
            return None
        try:
            from selenium.webdriver.support.ui import WebDriverWait  # noqa: PLC0415
            driver.get(url)
            try:
                WebDriverWait(driver, self._selenium_wait_timeout).until(
                    lambda d: len(d.find_element("tag name", "body").text.strip()) > 100
                )
            except Exception:
                pass  # Timeout non-fatal — use whatever the page has rendered so far
            return driver.page_source
        except Exception as exc:
            logger.warning("Selenium render error for %s: %s", url, exc)
            return None

    def _capture_screenshot(self, url: str, dest_dir: str) -> str | None:
        """Capture a full-page JPEG screenshot of *url* via the shared driver.

        Resizes the window to the full ``scrollWidth × scrollHeight`` of the
        page (matching the original training-data crawler approach) before
        capturing so the entire page is visible in one shot.

        Saves as JPEG to *dest_dir* and returns the path, or None on failure.
        """
        driver = self._selenium_driver
        if driver is None:
            return None
        try:
            from selenium.webdriver.support.ui import WebDriverWait  # noqa: PLC0415
            driver.get(url)
            try:
                WebDriverWait(driver, self._selenium_wait_timeout).until(
                    lambda d: len(d.find_element("tag name", "body").text.strip()) > 50
                )
            except Exception:
                pass
            # Measure full page dimensions (body vs documentElement — take the max)
            width = driver.execute_script(
                "return Math.max(document.body.scrollWidth, "
                "document.documentElement.scrollWidth);"
            )
            height = driver.execute_script(
                "return Math.max(document.body.scrollHeight, "
                "document.documentElement.scrollHeight);"
            )
            driver.set_window_size(width, min(height, 16000))
            time.sleep(2)  # allow any resize-triggered reflows to settle
            png_data = driver.get_screenshot_as_png()
        except Exception as exc:
            logger.warning("Screenshot capture error for %s: %s", url, exc)
            return None

        try:
            from PIL import Image  # noqa: PLC0415
            import io  # noqa: PLC0415
            img = Image.open(io.BytesIO(png_data)).convert("RGB")
            safe_name = hashlib.sha256(url.encode()).hexdigest()[:16]
            jpg_path = os.path.join(dest_dir, f"screenshot_{safe_name}.jpg")
            img.save(jpg_path, "JPEG", quality=85)
            logger.info("Screenshot saved (%dx%d) → %s  [%s]", width, height, jpg_path, url)
            return jpg_path
        except Exception as exc:
            logger.warning("Failed to convert screenshot to JPEG for %s: %s", url, exc)
            return None

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
            return None, {"status_code": status_code}
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
        csv_path = os.path.join(self.registry_dir, f"{self._collection_slug()}_error_matrix.csv")
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
        slug = self._collection_slug()
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
