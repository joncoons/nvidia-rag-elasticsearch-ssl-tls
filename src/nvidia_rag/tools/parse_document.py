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
parse_document tool — PDF → Nemotron-Parse → semantic chunks → Elasticsearch.

This module is the canonical home for the Nemotron-Parse document processing
pipeline.  It owns the full journey from a PDF file to indexed ES chunks,
using nvidia_rag.storage.embed_store as the storage primitive.

Two entry points
----------------
High-level (tool):
    parse_and_ingest(filepath, collection_name, config, embed_semaphore, ...)
        Runs the full pipeline end-to-end and returns a result dict.
        Handles temp-file lifecycle internally — no cleanup needed by caller.

Low-level (class):
    DocumentClassifierRouter
        Classifies and semantically chunks PDFs; returns temp .md files.
        Use this when the caller needs to manage files or ingest separately
        (e.g. main.py batching multiple documents before embedding).

Single-pass pipeline per PDF page using vLLM's OpenAI-compatible API with
prompt-based inference.  The model outputs structured text with embedded bbox
coordinates and class tags that are parsed to recover element types and content.

Model output format per element:
    <x_X1><y_Y1>TEXT CONTENT<x_X2><y_Y2><class_CLASSNAME>

Detected element classes (13 total):
    Text, Title, Section-header, List-item, TOC, Bibliography, Footnote,
    Page-header, Page-footer, Picture, Formula, Table, Caption

Semantic chunking (no overlap):
    Pages are processed in parallel.  After all pages complete, a cross-page
    stitch pass merges elements that span page boundaries (e.g. a paragraph
    whose last sentence wraps to the next page, or a table split across pages).
    The stitched element list is then chunked by semantic class transitions:

    * Title / Section-header  → always flush current chunk and start a new one.
    * Table + following Caption → emitted as an atomic unit (never split apart).
    * Formula                 → emitted as an atomic unit.
    * Picture                 → skipped; following Caption becomes "[Image: ...]".
    * Text, List-item, etc.   → accumulated under the current heading.
    * Page-header/footer, TOC → stripped (noise).

    When an accumulating section would exceed ``max_tokens`` the current chunk
    is flushed and the section heading is carried forward as context — no token
    overlap is needed because every split lands on a semantic boundary.

Routing behaviour:
    If ANY page contains a complex element (Table, Picture) the stitched content
    is semantically chunked and returned as temp ``.md`` files.  If no complex
    elements are found, ``None`` is returned so the caller uses the standard
    NV-Ingest pipeline.  When ``force=True`` routing runs unconditionally.

Usage (tool)::

    import asyncio
    from nvidia_rag.tools.parse_document import parse_and_ingest
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    config = NvidiaRAGConfig()
    sem = asyncio.Semaphore(8)
    result = await parse_and_ingest(
        filepath="/data/report.pdf",
        collection_name="my_collection",
        config=config,
        embed_semaphore=sem,
        source_uri="https://example.com/report.pdf",
    )
    # result: {"ingested": 42, "chunks": 42, "pipeline_type": "nemoretriever_parse",
    #          "fallback": False, "filename": "report.pdf"}

Usage (class)::

    from nvidia_rag.tools.parse_document import DocumentClassifierRouter
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    config = NvidiaRAGConfig()
    router = DocumentClassifierRouter.from_config(config)
    replacements = router.route_documents(filepaths)
    # replacements: {original_path: [(temp_md_path, chunk_meta), ...] | None}
"""

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from PIL import Image as PILImage

if TYPE_CHECKING:
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMO_PARSE_MODEL_DEFAULT = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
VLM_DESCRIBE_MODEL_DEFAULT = "nvidia/nemotron-nano-12b-v2-vl"

# Prompt tokens that instruct the model to produce structured markdown output.
# <predict_no_text_in_pic> suppresses transcription of text inside Picture elements.
_EXTRACTION_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)

# Prompt sent to the figure-description VLM for each cropped Picture region.
_FIGURE_DESCRIBE_PROMPT = (
    "Describe this figure in detail for a retrieval-augmented generation system. "
    "Include all visible text, labels, axis titles and values, data trends, "
    "legend entries, and any key observations. Be specific and thorough."
)

# Minimum fraction of page area a Picture region must occupy to be described.
# Filters out tiny decorative elements (logos, icons, horizontal rules).
_MIN_PICTURE_AREA_FRACTION = 0.015

# Regex to parse one element from the model's structured output:
#   <x_X1><y_Y1>CONTENT<x_X2><y_Y2><class_CLASSNAME>
_RE_ELEMENT = re.compile(
    r"<x_[\d.]+><y_[\d.]+>(.*?)<x_[\d.]+><y_[\d.]+><class_([^>]+)>",
    re.DOTALL,
)

# Regex to extract (x1, y1, x2, y2) for Picture-class elements specifically.
# Handles empty content between the two coordinate pairs.
_RE_PICTURE_BBOX = re.compile(
    r"<x_([\d.]+)><y_([\d.]+)>[^<]*?<x_([\d.]+)><y_([\d.]+)><class_Picture>",
    re.DOTALL | re.IGNORECASE,
)

# Element classes that trigger VLM-quality routing (trigger route_document to
# return chunked output instead of None).
COMPLEX_ELEMENT_CLASSES: frozenset[str] = frozenset({"table", "picture"})

# Semantic chunking taxonomy
_SECTION_STARTERS: frozenset[str] = frozenset({"title", "section-header"})
_ATOMIC_CLASSES: frozenset[str] = frozenset({"table", "formula"})
_STITCHABLE_CLASSES: frozenset[str] = frozenset({"text", "list-item"})
_TERMINAL_PUNCT: frozenset[str] = frozenset({".", "!", "?", ":", ";"})
_SKIP_CLASSES: frozenset[str] = frozenset({"page-header", "page-footer", "toc"})
_CAPTION_CLASS: str = "caption"


# ---------------------------------------------------------------------------
# High-level tool entry point
# ---------------------------------------------------------------------------

async def parse_and_ingest(
    filepath: str,
    collection_name: str,
    config: "NvidiaRAGConfig",
    embed_semaphore: asyncio.Semaphore,
    source_uri: str | None = None,
    extra_meta: dict | None = None,
    force: bool = False,
) -> dict:
    """
    Parse a PDF with Nemotron-Parse and ingest the chunks into Elasticsearch.

    This is the tool entry point — it owns the full pipeline from file to
    indexed chunks and handles temp file cleanup internally.

    Parameters
    ----------
    filepath : str
        Path to the PDF file to parse.
    collection_name : str
        Elasticsearch index / collection to write chunks into.
    config : NvidiaRAGConfig
        Configuration (embedding endpoint, ES connection, nemo_parse settings).
    embed_semaphore : asyncio.Semaphore
        Shared semaphore capping concurrent embedding API calls.
    source_uri : str | None
        Original source URL for the document (stored as content_url for upsert).
        Defaults to filepath if not provided.
    extra_meta : dict | None
        Additional metadata fields merged into each chunk's content_metadata.
    force : bool
        When True, bypass complex-element detection and always use
        Nemotron-Parse regardless of document content.

    Returns
    -------
    dict
        {
            "ingested": int,       # chunks written to ES
            "chunks": int,         # chunks produced by parser
            "pipeline_type": str,  # "nemoretriever_parse" | "nemoretriever_parse_forced"
            "fallback": bool,      # True if no complex elements → standard pipeline needed
            "filename": str,       # basename of the input file
        }
    """
    from nvidia_rag.storage.embed_store import ingest_chunk_files  # noqa: PLC0415

    filename = os.path.basename(filepath)
    uri = source_uri or filepath

    router = DocumentClassifierRouter.from_config(config)

    # Run the synchronous parse step in a thread so we don't block the event loop
    temp_pairs = await asyncio.to_thread(router.route_document, filepath, force)

    if temp_pairs is None:
        # No complex elements detected and force=False → signal fallback needed
        return {
            "ingested": 0,
            "chunks": 0,
            "pipeline_type": "standard",
            "fallback": True,
            "filename": filename,
        }

    temp_files = [tp for tp, _ in temp_pairs]
    chunk_metas = {os.path.basename(tp): meta for tp, meta in temp_pairs}
    pipeline_type = (
        "nemoretriever_parse_forced" if force else "nemoretriever_parse"
    )

    # Build per-file metadata maps for embed_store
    source_uri_map = {os.path.basename(tp): uri for tp in temp_files}
    extra_meta_map: dict[str, dict] = {}
    for tp, chunk_meta in temp_pairs:
        merged = {**chunk_meta}
        if extra_meta:
            merged.update(extra_meta)
        extra_meta_map[os.path.basename(tp)] = merged

    try:
        ingested = await ingest_chunk_files(
            chunk_files=temp_files,
            source_uri_map=source_uri_map,
            collection_name=collection_name,
            config=config,
            embed_semaphore=embed_semaphore,
            extra_meta_map=extra_meta_map,
        )
    finally:
        # Always clean up temp files regardless of ingest success/failure
        for tp in temp_files:
            try:
                os.unlink(tp)
            except OSError:
                pass

    return {
        "ingested": ingested,
        "chunks": len(temp_files),
        "pipeline_type": pipeline_type,
        "fallback": False,
        "filename": filename,
    }


# ---------------------------------------------------------------------------
# DocumentClassifierRouter — low-level class
# ---------------------------------------------------------------------------


class DocumentClassifierRouter:
    """
    Classifies and semantically chunks PDF documents using Nemotron-Parse.

    Pages are rasterised and processed in parallel.  Cross-page element
    boundaries are stitched before semantic chunking.  Each chunk corresponds
    to a complete semantic unit — no token overlap is required.

    Parameters
    ----------
    endpoint_url : str
        Full URL of the nemoretriever-parse inference endpoint.
    model_name : str
        Model identifier forwarded in the API payload.
    api_key : str
        Bearer token (empty string → no auth header).
    parse_max_tokens : int
        ``max_tokens`` cap for the extraction pass.  Default 8990.
    dpi : int
        DPI used when rasterising PDF pages (default 300, as recommended by
        the Nemotron-Parse documentation).
    max_tokens : int
        Maximum tokens per semantic chunk.  Non-homogeneous: most chunks will
        be smaller; this is only a ceiling.  Default 1024.
    max_parallel_pages : int
        ThreadPoolExecutor workers per document (pages in parallel).  Default 8.
    max_parallel_docs : int
        Number of documents processed concurrently.  Default 4.
        Each document spawns max_parallel_pages workers, so total in-flight
        requests = max_parallel_docs × max_parallel_pages.  Size to match
        nemotron-parse replica count × max-num-seqs (e.g. 7 replicas × 4 = 28
        slots → max_parallel_docs=4 × max_parallel_pages=8 = 32 ≈ saturated).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = NEMO_PARSE_MODEL_DEFAULT,
        api_key: str = "",
        parse_max_tokens: int = 8990,
        dpi: int = 300,
        max_tokens: int = 2048,
        chunk_overlap: int = 150,
        max_parallel_pages: int = 8,
        max_parallel_docs: int = 4,
        page_batch_size: int = 32,
        vlm_endpoint_url: str = "",
        vlm_model_name: str = VLM_DESCRIBE_MODEL_DEFAULT,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.parse_max_tokens = parse_max_tokens
        self.dpi = dpi
        self.max_parallel_pages = max(1, max_parallel_pages)
        self.max_parallel_docs = max(1, max_parallel_docs)
        self.page_batch_size = max(1, page_batch_size)
        self._max_tokens = max_tokens
        self._chunk_overlap = chunk_overlap
        self.vlm_endpoint_url = vlm_endpoint_url.rstrip("/") if vlm_endpoint_url else ""
        self.vlm_model_name = vlm_model_name

        # Thread-local sessions: requests.Session is not thread-safe for
        # concurrent use.  Each worker thread gets its own Session on first
        # access.  Connection: close forces a new TCP connection per request
        # so the k8s Service round-robins across all nemotron-parse replicas.
        self._session_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
        }
        if api_key:
            self._session_headers["Authorization"] = f"Bearer {api_key}"
        self._tls = threading.local()

    @property
    def _session(self) -> requests.Session:
        """Return a thread-local requests.Session, creating it on first access."""
        if not hasattr(self._tls, "session"):
            s = requests.Session()
            s.headers.update(self._session_headers)
            self._tls.session = s
        return self._tls.session

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "NvidiaRAGConfig") -> "DocumentClassifierRouter":
        """Construct a router from a NvidiaRAGConfig instance."""
        return cls(
            endpoint_url=config.nemo_parse.endpoint_url,
            model_name=config.nemo_parse.model_name,
            api_key=config.nemo_parse.api_key,
            max_tokens=config.nv_ingest.chunk_size,
            chunk_overlap=config.nv_ingest.chunk_overlap,
            max_parallel_docs=config.nemo_parse.max_parallel_docs,
            page_batch_size=config.nemo_parse.page_batch_size,
            vlm_endpoint_url=config.nemo_parse.figure_describe_endpoint,
            vlm_model_name=config.nemo_parse.figure_describe_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_document(self, filepath: str, force: bool = False) -> list[tuple[str, dict]] | None:
        """
        Classify and semantically chunk a single PDF file.

        Parameters
        ----------
        filepath : str
            Path to the PDF file.
        force : bool
            When ``True``, skip complex-element detection and unconditionally
            return chunked output for every PDF.

        Returns
        -------
        list[tuple[str, dict]]
            One or more ``(temp_md_path, chunk_meta)`` pairs.  The caller is
            responsible for deleting all returned paths after ingest.
        None
            No complex elements detected; use the standard NV-Ingest pipeline.
        """
        if not filepath.lower().endswith(".pdf"):
            logger.debug("Skipping non-PDF: %s", filepath)
            return None

        logger.info("Starting nemoretriever-parse pass for: %s", os.path.basename(filepath))

        try:
            from pdf2image import convert_from_path, pdfinfo_from_path  # lazy import
            page_count = pdfinfo_from_path(filepath)["Pages"]
        except Exception as exc:
            logger.warning(
                "Could not read page count for '%s': %s — falling back to standard pipeline",
                filepath, exc,
            )
            return None

        # ── Batched page rasterisation + parallel extraction ─────────────
        # Pages are loaded in batches of page_batch_size to cap peak memory.
        # Each batch is rasterised, processed by the thread pool, then freed
        # before the next batch is loaded.  Cross-batch element boundaries
        # are stitched using the same rules as within-page boundaries.
        logger.info(
            "Processing %d pages of '%s' in batches of %d with %d parallel workers",
            page_count, os.path.basename(filepath),
            self.page_batch_size, self.max_parallel_pages,
        )

        all_elements: list[tuple[str, str]] = []

        for batch_start in range(0, page_count, self.page_batch_size):
            batch_end = min(batch_start + self.page_batch_size, page_count)
            try:
                pages = convert_from_path(
                    filepath, dpi=self.dpi,
                    first_page=batch_start + 1,  # 1-indexed
                    last_page=batch_end,
                )
            except Exception as exc:
                logger.warning(
                    "Could not rasterise pages %d-%d of '%s': %s — skipping batch",
                    batch_start + 1, batch_end, os.path.basename(filepath), exc,
                )
                continue

            with ThreadPoolExecutor(max_workers=self.max_parallel_pages) as executor:
                batch_results: list[tuple[int, list[tuple[str, str]]]] = list(
                    executor.map(self._process_page, enumerate(pages, start=batch_start))
                )

            # Stitch within this batch
            page_element_lists = [elems for _, elems in batch_results]
            batch_elements = self._stitch_page_boundaries(page_element_lists)

            # Cross-batch stitch: attempt to join the last element of the
            # previous batch with the first element of this batch using the
            # same rules as _stitch_page_boundaries.
            if all_elements and batch_elements:
                prev_cls, prev_text = all_elements[-1]
                next_cls, next_text = batch_elements[0]
                prev_l = prev_cls.lower()
                next_l = next_cls.lower()
                should_stitch = False
                if prev_l in _STITCHABLE_CLASSES and prev_l == next_l:
                    if prev_text and prev_text.rstrip()[-1] not in _TERMINAL_PUNCT:
                        should_stitch = True
                elif prev_l in _ATOMIC_CLASSES and prev_l == next_l:
                    should_stitch = True
                if should_stitch:
                    sep = " " if prev_l in _STITCHABLE_CLASSES else "\n"
                    stitched = prev_text.rstrip() + sep + next_text.lstrip()
                    all_elements[-1] = (prev_cls, stitched)
                    batch_elements = batch_elements[1:]

            all_elements.extend(batch_elements)
            # PIL images freed here as `pages` goes out of scope

        # ── Routing decision ─────────────────────────────────────────────
        if not force:
            all_classes = {cls.lower() for cls, _ in all_elements}
            if not (all_classes & COMPLEX_ELEMENT_CLASSES):
                logger.info(
                    "No complex elements in '%s' — standard NV-Ingest pipeline will be used",
                    os.path.basename(filepath),
                )
                return None

        if not all_elements:
            logger.warning(
                "nemoretriever-parse returned no content for '%s' — falling back",
                filepath,
            )
            return None

        # ── Semantic chunking ────────────────────────────────────────────
        detected = {cls.lower() for cls, _ in all_elements} & COMPLEX_ELEMENT_CLASSES
        logger.info(
            "Routing '%s' through nemoretriever-parse "
            "(detected: %s, %d elements from %d pages)",
            os.path.basename(filepath),
            detected or "forced",
            len(all_elements),
            page_count,
        )

        chunk_pairs = self._split_by_semantic_elements(
            all_elements, self._max_tokens, self._chunk_overlap
        )

        # ── Write temp markdown files ────────────────────────────────────
        stem = Path(filepath).stem
        pipeline = "nemoretriever_parse_forced" if force else "nemoretriever_parse"
        temp_pairs: list[tuple[str, dict]] = []
        try:
            total = sum(1 for ct, _ in chunk_pairs if ct.strip())
            idx = 0
            for chunk_text, section_path in chunk_pairs:
                if not chunk_text.strip():
                    continue
                suffix = f"_{idx + 1:03d}.md" if total > 1 else ".md"
                tmp_fd, tmp_path = tempfile.mkstemp(
                    suffix=suffix, prefix=f"{stem}_nemoparse_"
                )
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(chunk_text)
                chunk_meta: dict = {
                    "chunk_index": idx,
                    "total_chunks": total,
                    "page_count": page_count,
                    "pipeline_type": pipeline,
                }
                if section_path:
                    chunk_meta["section_path"] = section_path
                if detected:
                    chunk_meta["detected_element_types"] = sorted(detected)
                temp_pairs.append((tmp_path, chunk_meta))
                idx += 1
        except Exception:
            for tp, _ in temp_pairs:
                try:
                    os.unlink(tp)
                except OSError:
                    pass
            raise

        logger.info(
            "nemoretriever-parse output for '%s' written to %d semantic chunk file(s)",
            os.path.basename(filepath), len(temp_pairs),
        )
        return temp_pairs if temp_pairs else None

    def route_documents(
        self, filepaths: list[str], force: bool = False
    ) -> dict[str, list[tuple[str, dict]] | None]:
        """Classify and route a batch of file paths in parallel across documents.

        Uses a two-level thread pool:
          Outer (this method): max_parallel_docs PDFs concurrently.
          Inner (route_document): max_parallel_pages pages per PDF concurrently.

        Total in-flight nemotron-parse requests ≈ max_parallel_docs × max_parallel_pages.
        """
        if len(filepaths) <= 1:
            return {fp: self.route_document(fp, force=force) for fp in filepaths}

        results: dict[str, list[tuple[str, dict]] | None] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel_docs) as executor:
            future_to_fp = {
                executor.submit(self.route_document, fp, force): fp
                for fp in filepaths
            }
            for future in future_to_fp:
                fp = future_to_fp[future]
                try:
                    results[fp] = future.result()
                except Exception as exc:
                    logger.warning("route_document failed for '%s': %s", fp, exc)
                    results[fp] = None
        return results

    # ------------------------------------------------------------------
    # Private helpers — image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(img: PILImage.Image) -> tuple[str, str]:
        """Encode a PIL image as base64 JPEG; return (b64_string, mime_type).

        JPEG at quality=95 is 3-5x smaller and faster to encode than lossless
        PNG with no meaningful loss for document text at 300 DPI.
        """
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, "image/jpeg"

    # ------------------------------------------------------------------
    # Private helpers — nemoretriever-parse API
    # ------------------------------------------------------------------

    def _call_nemo_parse(self, b64: str, mime: str, max_tokens: int) -> str:
        """Issue one nemoretriever-parse request; return raw model output text."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "repetition_penalty": 1.1,
            "top_k": 1,
            "skip_special_tokens": False,
        }
        resp = self._session.post(self.endpoint_url, json=payload, timeout=180)
        resp.raise_for_status()
        r = resp.json()
        return r.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _call_vlm_describe(self, b64: str, mime: str) -> str:
        """Call the figure-description VLM on a cropped Picture region."""
        payload = {
            "model": self.vlm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": _FIGURE_DESCRIBE_PROMPT},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }
        resp = self._session.post(self.vlm_endpoint_url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------
    # Private helpers — page processing (ThreadPoolExecutor worker)
    # ------------------------------------------------------------------

    def _process_page(self, args: tuple[int, Any]) -> tuple[int, list[tuple[str, str]]]:
        """
        Worker target for parallel page processing.

        Parameters
        ----------
        args : tuple[int, PIL.Image]
            ``(page_index, page_image)`` from ``enumerate(pages)``.

        Returns
        -------
        tuple[int, list[tuple[str, str]]]
            ``(page_index, elements)`` where each element is
            ``(class_name, text)`` in reading order.
        """
        i, page_img = args
        b64, mime = self._pil_to_base64(page_img)
        try:
            raw = self._call_nemo_parse(b64, mime, self.parse_max_tokens)
        except Exception as exc:
            logger.warning("nemoretriever-parse call failed on page %d: %s", i + 1, exc)
            return i, []
        elements = self._parse_model_output_elements(raw)
        if self.vlm_endpoint_url:
            elements = self._describe_pictures(raw, page_img, elements, i + 1)
        return i, elements

    def _describe_pictures(
        self,
        raw: str,
        page_img: "PILImage.Image",
        elements: list[tuple[str, str]],
        page_num: int,
    ) -> list[tuple[str, str]]:
        """
        Replace empty Picture elements with nim-vlm figure descriptions.

        Finds each Picture bounding box in the raw nemoretriever-parse output,
        crops the corresponding region from *page_img*, sends it to the VLM,
        and substitutes the description as the element text so the downstream
        chunker can emit it as searchable content.
        """
        bboxes = self._parse_picture_bboxes(raw)
        if not bboxes:
            return elements

        W, H = page_img.size
        page_area = W * H
        bbox_iter = iter(bboxes)
        result: list[tuple[str, str]] = []

        for cls, text in elements:
            if cls.lower() != "picture":
                result.append((cls, text))
                continue

            bbox = next(bbox_iter, None)
            if bbox is None:
                result.append((cls, text))
                continue

            x1f, y1f, x2f, y2f = bbox
            # Normalise so x1 < x2, y1 < y2
            x1f, x2f = min(x1f, x2f), max(x1f, x2f)
            y1f, y2f = min(y1f, y2f), max(y1f, y2f)

            # Skip regions that are too small to be meaningful figures
            crop_area = (x2f - x1f) * W * (y2f - y1f) * H
            if crop_area < _MIN_PICTURE_AREA_FRACTION * page_area:
                logger.debug(
                    "Page %d: skipping tiny Picture region (%.1f%% of page)",
                    page_num, 100 * crop_area / page_area,
                )
                result.append((cls, text))
                continue

            # Crop with a small margin and clamp to image bounds
            margin_x = int(0.005 * W)
            margin_y = int(0.005 * H)
            left   = max(0, int(x1f * W) - margin_x)
            upper  = max(0, int(y1f * H) - margin_y)
            right  = min(W, int(x2f * W) + margin_x)
            lower  = min(H, int(y2f * H) + margin_y)

            try:
                crop = page_img.crop((left, upper, right, lower))
                cb64, cmime = self._pil_to_base64(crop)
                description = self._call_vlm_describe(cb64, cmime)
                logger.debug(
                    "Page %d: VLM described Picture (%.0f×%.0f px) → %d chars",
                    page_num, right - left, lower - upper, len(description),
                )
                result.append((cls, description))
            except Exception as exc:
                logger.warning(
                    "Page %d: VLM figure description failed: %s", page_num, exc
                )
                result.append((cls, text))

        return result

    @staticmethod
    def _parse_picture_bboxes(raw: str) -> list[tuple[float, float, float, float]]:
        """Return (x1, y1, x2, y2) normalised coords for every Picture element."""
        return [
            (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            for m in _RE_PICTURE_BBOX.finditer(raw)
        ]

    # ------------------------------------------------------------------
    # Private helpers — model output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_output_elements(raw: str) -> list[tuple[str, str]]:
        """
        Parse structured model output into ``[(class_name, text), ...]``.

        Noise classes (Page-header, Page-footer, TOC) are filtered out.
        Atomic elements (Table, Formula) are included even when text is empty
        so that downstream stitching and chunking can detect them.
        Picture elements are always preserved (text is empty; descriptions are
        filled in later by ``_describe_pictures`` when VLM captioning is active).
        """
        elements: list[tuple[str, str]] = []
        for m in _RE_ELEMENT.finditer(raw):
            text = m.group(1).strip()
            cls = m.group(2)
            cls_lower = cls.lower()
            if cls_lower in _SKIP_CLASSES:
                continue
            if text or cls_lower in _ATOMIC_CLASSES or cls_lower == "picture":
                elements.append((cls, text))
        return elements

    # ------------------------------------------------------------------
    # Private helpers — cross-page element stitching
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_page_boundaries(
        page_element_lists: list[list[tuple[str, str]]],
    ) -> list[tuple[str, str]]:
        """
        Merge elements that span a PDF page break.

        Rules
        -----
        Text / List-item
            Stitch when the last element on page N has the same class as the
            first element on page N+1 AND the last character of page N's text
            is not terminal punctuation (``. ! ? : ;``).

        Table / Formula
            Always stitch same-class adjacency across a page break (partial
            table rows or formula lines split at the page boundary).

        Section-header / Title
            Never stitch — a new heading always starts a fresh semantic unit.
        """
        all_elements: list[tuple[str, str]] = []
        for elements in page_element_lists:
            if not elements:
                continue
            if all_elements:
                prev_cls, prev_text = all_elements[-1]
                next_cls, next_text = elements[0]
                prev_l = prev_cls.lower()
                next_l = next_cls.lower()

                should_stitch = False
                if prev_l in _STITCHABLE_CLASSES and prev_l == next_l:
                    # Text / List-item: stitch only when clearly mid-sentence
                    if prev_text and prev_text.rstrip()[-1] not in _TERMINAL_PUNCT:
                        should_stitch = True
                elif prev_l in _ATOMIC_CLASSES and prev_l == next_l:
                    # Table / Formula continuation across page — always stitch
                    should_stitch = True

                if should_stitch:
                    sep = " " if prev_l in _STITCHABLE_CLASSES else "\n"
                    stitched = prev_text.rstrip() + sep + next_text.lstrip()
                    all_elements[-1] = (prev_cls, stitched)
                    elements = elements[1:]

            all_elements.extend(elements)
        return all_elements

    # ------------------------------------------------------------------
    # Private helpers — semantic chunker
    # ------------------------------------------------------------------

    @staticmethod
    def _split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
        """
        Split *text* into sub-chunks when it exceeds *max_chars*.

        Tries to split on sentence boundaries (``'. '``), then word boundaries,
        falling back to a hard character split.  Each sub-chunk after the first
        starts *overlap_chars* before the previous split point so that context
        is not lost mid-thought.

        This is the failsafe for individual elements that are themselves larger
        than the semantic chunk ceiling — it is NOT applied to atomic structural
        units (Table, Formula) whose content must not be broken mid-structure.
        """
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_chars
            if end >= text_len:
                tail = text[start:].strip()
                if tail:
                    chunks.append(tail)
                break

            # Prefer sentence boundary within the window
            split_at = text.rfind(". ", start + overlap_chars, end)
            if split_at > start:
                split_at += 1  # include the period
            else:
                # Fall back to word boundary
                split_at = text.rfind(" ", start + overlap_chars, end)
                if split_at <= start:
                    split_at = end  # hard split — no whitespace found

            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            # Step forward, keeping overlap_chars of context in the next chunk
            start = max(start + 1, split_at - overlap_chars)

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _split_by_semantic_elements(
        elements: list[tuple[str, str]],
        max_tokens: int,
        chunk_overlap: int = 150,
    ) -> list[tuple[str, str]]:
        """
        Build non-homogeneous semantic chunks from classified elements.

        No overlap.  Chunk boundaries are always semantic transitions:

        * ``Title`` / ``Section-header`` → flush current chunk, start new.
        * ``Table`` + following ``Caption`` → atomic unit.
        * ``Formula`` → atomic unit; ``Caption`` bound if immediately following.
        * ``Picture`` → skipped; following ``Caption`` becomes ``[Image: ...]``.
        * ``Text``, ``List-item``, ``Footnote``, ``Bibliography`` → accumulated.
        * ``Page-header``, ``Page-footer``, ``TOC`` → discarded.

        When adding the next element would exceed ``max_tokens``, the current
        chunk is flushed and the active section heading is carried forward as
        context — no token overlap is required.

        Parameters
        ----------
        elements : list[tuple[str, str]]
            ``(class_name, text)`` pairs in reading order (post-stitch).
        max_tokens : int
            Ceiling on chunk size.  4 chars ≈ 1 token (rough approximation).

        Returns
        -------
        list[tuple[str, str]]
            ``(chunk_text, section_path)`` pairs.  At least one entry is
            always returned; ``section_path`` may be an empty string.
        """
        max_chars = max_tokens * 4      # ~4 chars per token
        overlap_chars = chunk_overlap * 4  # same approximation

        chunks: list[tuple[str, str]] = []
        parts: list[str] = []
        chars: int = 0
        headers: dict[int, str] = {}
        section_path: str = ""

        def flush() -> None:
            nonlocal parts, chars
            body = "\n\n".join(p for p in parts if p.strip()).strip()
            if body:
                chunks.append((body, section_path))
            parts = []
            chars = 0

        def _add(unit: str) -> None:
            """Append *unit* to current chunk, flushing with context carry if needed."""
            nonlocal parts, chars
            sep = 2 if parts else 0
            if parts and chars + sep + len(unit) > max_chars:
                flush()
                ctx = DocumentClassifierRouter._build_header_context(headers)
                if ctx:
                    parts.append(ctx)
                    chars = len(ctx)
                    sep = 2
                parts.append(unit)
                chars += sep + len(unit)
            else:
                parts.append(unit)
                chars += sep + len(unit)

        def _fmt(cls_l: str, text: str) -> str:
            """Apply element-class-specific markdown formatting."""
            if cls_l == "table":
                return text  # already markdown from nemotron-parse
            if cls_l == "formula":
                if not text.startswith(("```", "$$")):
                    return f"```\n{text}\n```"
                return text
            if cls_l == "list-item":
                lines = [
                    f"- {ln}" if not ln.startswith(("- ", "* ", "• ")) else ln
                    for ln in text.splitlines()
                    if ln.strip()
                ]
                return "\n".join(lines) if lines else f"- {text}"
            if cls_l == "footnote":
                return f"> {text}"
            return text

        i = 0
        while i < len(elements):
            cls, text = elements[i]
            cls_l = cls.lower()
            text = text.strip()
            i += 1

            if cls_l in _SKIP_CLASSES or (not text and cls_l not in _ATOMIC_CLASSES):
                continue

            # ── Section boundary ──────────────────────────────────────────
            if cls_l in _SECTION_STARTERS:
                flush()
                level = 1 if cls_l == "title" else 2
                headers[level] = text
                for k in list(headers):
                    if k > level:
                        del headers[k]
                section_path = DocumentClassifierRouter._section_path_string(headers)
                header_md = "#" * level + " " + text
                parts.append(header_md)
                chars = len(header_md)
                continue

            # ── Picture: emit VLM description (if available) + caption ───
            if cls_l == "picture":
                cap = ""
                if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                    cap = elements[i][1].strip()
                    i += 1
                if text:
                    # VLM description was generated for this region
                    parts_list = []
                    if cap:
                        parts_list.append(f"**[Figure]** *{cap}*")
                    parts_list.append(text)
                    _add("\n\n".join(parts_list))
                elif cap:
                    _add(f"[Image: {cap}]")
                continue

            # ── Atomic element (Table / Formula) with optional Caption ─────
            if cls_l in _ATOMIC_CLASSES:
                formatted = _fmt(cls_l, text)
                atom_parts_list = [formatted]
                atom_chars = len(formatted)
                if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                    cap = elements[i][1].strip()
                    if cap:
                        cap_md = f"*{cap}*"
                        atom_parts_list.append(cap_md)
                        atom_chars += 2 + len(cap_md)
                    i += 1
                atom_block = "\n\n".join(atom_parts_list)
                if atom_chars >= max_chars:
                    # Oversized atom: emit standalone
                    flush()
                    chunks.append((atom_block, section_path))
                else:
                    _add(atom_block)
                continue

            # ── Regular content ───────────────────────────────────────────
            unit = f"*{text}*" if cls_l == _CAPTION_CLASS else _fmt(cls_l, text)
            if overlap_chars > 0 and len(unit) > max_chars:
                # Failsafe: single element exceeds ceiling — split with overlap.
                # Only text-like content reaches here (atomics use the path above).
                for sub in DocumentClassifierRouter._split_oversized_text(
                    unit, max_chars, overlap_chars
                ):
                    _add(sub)
            else:
                _add(unit)

        flush()
        return chunks if chunks else [("", "")]

    # ------------------------------------------------------------------
    # Private helpers — header breadcrumb utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header_context(headers: dict[int, str]) -> str:
        """Return H1–H3 header lines as context prefix for overflow chunks."""
        lines = [headers[lvl] for lvl in sorted(headers) if lvl <= 3]
        return "\n".join(lines)

    @staticmethod
    def _section_path_string(headers: dict[int, str]) -> str:
        """Return human-readable section breadcrumb, e.g. 'Overview > Architecture'.

        Strips leading ``#`` markers so the value is suitable for metadata storage
        and natural-language filter generation.
        """
        parts = [headers[lvl].lstrip("#").strip() for lvl in sorted(headers) if lvl <= 3]
        return " > ".join(p for p in parts if p)
