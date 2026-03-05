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
Document Classifier and Router for nemoretriever-parse VLM.

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

Usage::

    from nvidia_rag.ingestor_server.document_classifier_router import (
        DocumentClassifierRouter,
    )
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    config = NvidiaRAGConfig()
    router = DocumentClassifierRouter.from_config(config)
    replacements = router.route_documents(filepaths)
    # replacements: {original_path: [(temp_md_path, chunk_meta), ...] | None}
"""

import base64
import io
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMO_PARSE_MODEL_DEFAULT = "nvidia/NVIDIA-Nemotron-Parse-v1.2"

# Prompt tokens that instruct the model to produce structured markdown output.
# <predict_no_text_in_pic> suppresses transcription of text inside Picture elements.
_EXTRACTION_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)

# Regex to parse one element from the model's structured output:
#   <x_X1><y_Y1>CONTENT<x_X2><y_Y2><class_CLASSNAME>
_RE_ELEMENT = re.compile(
    r"<x_[\d.]+><y_[\d.]+>(.*?)<x_[\d.]+><y_[\d.]+><class_([^>]+)>",
    re.DOTALL,
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
# Main class
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
        ThreadPoolExecutor workers per document batch.  Default 8 (matches
        3 replica pods × 4 max-num-seqs = 12 slots; 8 workers saturates the
        pool with 2 concurrent document batches = 16 in-flight requests).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = NEMO_PARSE_MODEL_DEFAULT,
        api_key: str = "",
        parse_max_tokens: int = 8990,
        dpi: int = 300,
        max_tokens: int = 1024,
        chunk_overlap: int = 150,
        max_parallel_pages: int = 8,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.parse_max_tokens = parse_max_tokens
        self.dpi = dpi
        self.max_parallel_pages = max(1, max_parallel_pages)
        self._max_tokens = max_tokens
        self._chunk_overlap = chunk_overlap

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
    def from_config(cls, config: "NvidiaRAGConfig") -> "DocumentClassifierRouter":  # noqa: F821
        """Construct a router from a NvidiaRAGConfig instance."""
        return cls(
            endpoint_url=config.nemo_parse.endpoint_url,
            model_name=config.nemo_parse.model_name,
            api_key=config.nemo_parse.api_key,
            max_tokens=config.nv_ingest.chunk_size,
            chunk_overlap=config.nv_ingest.chunk_overlap,
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
            from pdf2image import convert_from_path  # lazy import

            pages = convert_from_path(filepath, dpi=self.dpi)
            page_count = len(pages)
        except Exception as exc:
            logger.warning(
                "Could not rasterise '%s': %s — falling back to standard pipeline",
                filepath, exc,
            )
            return None

        # ── Parallel single-pass extraction ─────────────────────────────
        # 4 replica pods × 4 max-num-seqs = 16 concurrent inference slots.
        # 8 workers × 2 concurrent document batches = 16 in-flight requests.
        logger.info(
            "Processing %d pages of '%s' with %d parallel workers",
            page_count, os.path.basename(filepath), self.max_parallel_pages,
        )

        with ThreadPoolExecutor(max_workers=self.max_parallel_pages) as executor:
            page_results: list[tuple[int, list[tuple[str, str]]]] = list(
                executor.map(self._process_page, enumerate(pages))
            )

        # ── Cross-page stitch ────────────────────────────────────────────
        page_element_lists = [elems for _, elems in page_results]
        all_elements = self._stitch_page_boundaries(page_element_lists)

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
        """Classify and route a batch of file paths."""
        return {fp: self.route_document(fp, force=force) for fp in filepaths}

    # ------------------------------------------------------------------
    # Private helpers — image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(img: PILImage.Image) -> tuple[str, str]:
        """Encode a PIL image as base64 PNG; return (b64_string, mime_type)."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, "image/png"

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
        return i, elements

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
        """
        elements: list[tuple[str, str]] = []
        for m in _RE_ELEMENT.finditer(raw):
            text = m.group(1).strip()
            cls = m.group(2)
            cls_lower = cls.lower()
            if cls_lower in _SKIP_CLASSES:
                continue
            if text or cls_lower in _ATOMIC_CLASSES:
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

            # ── Picture: skip body, capture caption as alt-text ───────────
            if cls_l == "picture":
                if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                    cap = elements[i][1].strip()
                    if cap:
                        _add(f"[Image: {cap}]")
                    i += 1
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
