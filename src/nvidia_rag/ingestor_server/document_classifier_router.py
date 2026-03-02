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
prompt-based inference.  No tool calls are required — the model outputs
structured text with embedded bbox coordinates and class tags that are parsed
to recover element types and content.

Model output format per element:
    <x_X1><y_Y1>TEXT CONTENT<x_X2><y_Y2><class_CLASSNAME>

Complex element trigger classes (triggers VLM-quality markdown extraction):
    table, picture

Routing behaviour (document-level):
    If ANY page contains a complex element the entire document text (extracted
    by the VLM) is compiled into chunks and returned.  The caller substitutes
    these files in the NV-Ingest file list.  If no complex elements are found
    the caller proceeds with the standard NV-Ingest pipeline.

Implementation notes:
    - A single API call per page gives both detection AND extraction.
    - All pages are always processed so that context around complex elements
      is preserved in the final output.
    - Simple documents (no complex classes) return None; the inference cost
      is low since plain-text pages produce short outputs.

Usage::

    from nvidia_rag.ingestor_server.document_classifier_router import (
        DocumentClassifierRouter,
    )
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    config = NvidiaRAGConfig()
    router = DocumentClassifierRouter.from_config(config)

    replacements = router.route_documents(filepaths)
    # replacements: {original_path: temp_md_path | None}
    # temp_md_path is None  → use standard NV-Ingest pipeline
    # temp_md_path is a str → replace with the markdown file, delete after ingest
"""

import base64
import io
import logging
import os
import re
import tempfile
from pathlib import Path

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
# Non-greedy so each finditer call yields exactly one element at a time.
_RE_ELEMENT = re.compile(
    r"<x_[\d.]+><y_[\d.]+>(.*?)<x_[\d.]+><y_[\d.]+><class_([^>]+)>",
    re.DOTALL,
)

# Element classes from the model that signal complex data-presentation content
# warranting VLM-quality markdown extraction.  These are the actual class names
# the model emits (case-insensitive comparison applied on match).
COMPLEX_ELEMENT_CLASSES: frozenset[str] = frozenset({"table", "picture"})

# Element classes that produce no useful text and should be omitted from the
# compiled markdown output.
_SKIP_CLASSES: frozenset[str] = frozenset({"page-header", "page-footer", "picture"})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DocumentClassifierRouter:
    """
    Classifies PDF pages with Nemotron-Parse and routes documents that contain
    complex data elements (tables, figures) through VLM-quality extraction.

    Parameters
    ----------
    endpoint_url : str
        Full URL of the nemoretriever-parse inference endpoint, e.g.
        ``http://nemotron-parse-v12:8000/v1/chat/completions``.
    model_name : str
        Model identifier forwarded in the API payload.  For standalone vLLM
        served from a local path, pass the full container path
        (e.g. ``/hf-models/nvidia/NVIDIA-Nemotron-Parse-v1.2``).
    api_key : str
        Bearer token for the inference endpoint (empty string → no auth header).
    parse_max_tokens : int
        ``max_tokens`` cap for the extraction pass.  Must be less than the
        model's ``max_sequence_length`` (9000) minus the prompt token count
        (~6).  Default 8990 leaves a small safety margin.
    dpi : int
        DPI used when rasterising PDF pages (default 200).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = NEMO_PARSE_MODEL_DEFAULT,
        api_key: str = "",
        parse_max_tokens: int = 8990,
        dpi: int = 200,
        chunk_size: int = 512,
        chunk_overlap: int = 150,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.parse_max_tokens = parse_max_tokens
        self.dpi = dpi
        # Markdown pre-chunking: convert token counts to approximate char counts (4 chars/token)
        self._chunk_max_chars = max(chunk_size * 4, 512)
        self._chunk_overlap_chars = chunk_overlap * 4

        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

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
            chunk_size=config.nv_ingest.chunk_size,
            chunk_overlap=config.nv_ingest.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_document(self, filepath: str, force: bool = False) -> list[tuple[str, dict]] | None:
        """
        Classify and (if warranted) parse a single PDF file.

        Parameters
        ----------
        filepath : str
            Path to the PDF file to process.
        force : bool
            When ``True``, skip complex-element detection and unconditionally
            include all extracted VLM content in the output.  The
            ``pipeline_type`` metadata field will be set to
            ``"nemoretriever_parse_forced"`` instead of
            ``"nemoretriever_parse"``.

        Returns
        -------
        list[tuple[str, dict]]
            One or more ``(path, chunk_meta)`` pairs where ``path`` is a
            temporary ``.md`` chunk file and ``chunk_meta`` is a dict of
            per-chunk metadata.  **The caller is responsible for deleting all
            returned files after they have been submitted to NV-Ingest.**
        None
            The document contains no complex data elements; use the standard
            NV-Ingest pipeline.
        """
        if not filepath.lower().endswith(".pdf"):
            logger.debug("Skipping non-PDF file for nemoretriever-parse routing: %s", filepath)
            return None

        logger.info("Starting classifier pass for: %s", os.path.basename(filepath))

        try:
            from pdf2image import convert_from_path  # lazy import — optional dep

            pages = convert_from_path(filepath, dpi=self.dpi)
            page_count = len(pages)
        except Exception as exc:
            logger.warning(
                "Could not rasterise PDF '%s': %s — falling back to standard pipeline",
                filepath,
                exc,
            )
            return None

        # ----------------------------------------------------------------
        # Single-pass extraction: each page is processed once with the full
        # extraction prompt.  The model output contains both element class
        # labels (for detection) and text content (for extraction), so no
        # separate detection API call is needed.
        # ----------------------------------------------------------------
        all_detected_types: set[str] = set()
        page_parts: list[str] = []

        for i, page_img in enumerate(pages):
            b64, mime = self._pil_to_base64(page_img)
            page_classes, page_markdown = self._extract_page(b64, mime)

            complex_on_page = page_classes & COMPLEX_ELEMENT_CLASSES
            all_detected_types |= complex_on_page

            if complex_on_page:
                logger.info(
                    "Complex element(s) detected on page %d of '%s': %s",
                    i + 1,
                    os.path.basename(filepath),
                    complex_on_page,
                )

            if page_markdown:
                page_parts.append(f"<!-- Page {i + 1} -->\n\n{page_markdown}")

        # ----------------------------------------------------------------
        # Routing decision
        # ----------------------------------------------------------------
        if not force and not all_detected_types:
            logger.info(
                "No complex elements in '%s' — standard NV-Ingest pipeline will be used",
                os.path.basename(filepath),
            )
            return None

        if not page_parts:
            logger.warning(
                "nemoretriever-parse returned no usable content for '%s' — "
                "falling back to standard pipeline",
                filepath,
            )
            return None

        logger.info(
            "Routing '%s' through nemoretriever-parse "
            "(detected: %s, %d pages processed)",
            os.path.basename(filepath),
            all_detected_types or "forced",
            page_count,
        )

        # ----------------------------------------------------------------
        # Pre-chunk with markdown-aware splitter; write one temp file per chunk
        # ----------------------------------------------------------------
        stem = Path(filepath).stem
        full_text = f"# {stem}\n\n" + "\n\n---\n\n".join(page_parts)
        chunk_pairs = self._split_markdown(
            full_text,
            max_chars=self._chunk_max_chars,
            overlap_chars=self._chunk_overlap_chars,
        )

        temp_pairs: list[tuple[str, dict]] = []
        try:
            for idx, (chunk_text, section_path) in enumerate(chunk_pairs):
                suffix = f"_{idx + 1:03d}.md" if len(chunk_pairs) > 1 else ".md"
                tmp_fd, tmp_path = tempfile.mkstemp(
                    suffix=suffix, prefix=f"{stem}_nemoparse_"
                )
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(chunk_text)
                chunk_meta: dict = {
                    "chunk_index": idx,
                    "total_chunks": len(chunk_pairs),
                    "page_count": page_count,
                }
                if force:
                    chunk_meta["pipeline_type"] = "nemoretriever_parse_forced"
                if section_path:
                    chunk_meta["section_path"] = section_path
                if all_detected_types:
                    chunk_meta["detected_element_types"] = sorted(all_detected_types)
                temp_pairs.append((tmp_path, chunk_meta))
        except Exception:
            for tp, _ in temp_pairs:
                try:
                    os.unlink(tp)
                except OSError:
                    pass
            raise

        logger.info(
            "nemoretriever-parse output for '%s' written to %d chunk file(s)",
            os.path.basename(filepath),
            len(temp_pairs),
        )
        return temp_pairs

    def route_documents(
        self, filepaths: list[str], force: bool = False
    ) -> dict[str, list[tuple[str, dict]] | None]:
        """
        Classify and route a batch of file paths.

        Parameters
        ----------
        filepaths : list[str]
            Paths to the files to process.
        force : bool
            When ``True``, skip Pass 1 for all PDFs and unconditionally run
            Pass 2.  Forwarded to :meth:`route_document`.

        Returns
        -------
        dict[str, list[tuple[str, dict]] | None]
            Maps each original file path to either:
              - A list of ``(path, chunk_meta)`` pairs (caller must delete all paths after ingest), or
              - ``None`` (no complex elements; use standard NV-Ingest pipeline).
        """
        results: dict[str, list[tuple[str, dict]] | None] = {}
        for fp in filepaths:
            results[fp] = self.route_document(fp, force=force)
        return results

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
    # Private helpers — nemoretriever-parse API call
    # ------------------------------------------------------------------

    def _call_nemo_parse(self, b64: str, mime: str, max_tokens: int) -> str:
        """
        Issue a single nemoretriever-parse request via the vLLM OpenAI-compatible
        chat completions endpoint.

        Returns the raw model output text (may contain bbox/class tags).
        """
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

    def _extract_page(self, b64: str, mime: str) -> tuple[set[str], str]:
        """
        Single-pass extraction for one page image.

        Calls the vLLM endpoint once with the structured extraction prompt,
        then parses the model output to recover element classes and text.

        Returns
        -------
        tuple[set[str], str]
            ``(detected_classes, markdown_text)`` where ``detected_classes``
            contains the lowercased class names found on the page and
            ``markdown_text`` is the extracted content suitable for embedding.
            On any API or parse error returns ``(set(), "")``.
        """
        try:
            raw = self._call_nemo_parse(b64, mime, self.parse_max_tokens)
        except Exception as exc:
            logger.warning("nemoretriever-parse call failed: %s", exc)
            return set(), ""

        return self._parse_model_output(raw)

    # ------------------------------------------------------------------
    # Private helpers — model output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_output(raw: str) -> tuple[set[str], str]:
        """
        Parse the structured model output into element classes and markdown text.

        The model output format is::

            <x_X1><y_Y1>TEXT<x_X2><y_Y2><class_CLASSNAME>\\n\\n...

        Each ``finditer`` match yields exactly one element (non-greedy regex).

        Parameters
        ----------
        raw : str
            Raw text from the model response.

        Returns
        -------
        tuple[set[str], str]
            ``(detected_classes, markdown_text)``.
            ``detected_classes`` contains the lowercased class name of every
            element found.  ``markdown_text`` joins text content of elements
            whose class is not in ``_SKIP_CLASSES``, separated by blank lines.
        """
        detected: set[str] = set()
        parts: list[str] = []

        for m in _RE_ELEMENT.finditer(raw):
            text, cls = m.group(1).strip(), m.group(2)
            cls_lower = cls.lower()
            detected.add(cls_lower)
            if text and cls_lower not in _SKIP_CLASSES:
                parts.append(text)

        return detected, "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers — markdown-aware chunker (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_markdown(
        text: str, max_chars: int = 2048, overlap_chars: int = 600
    ) -> list[tuple[str, str]]:
        """
        Split markdown text into chunks that respect header, table, and
        code-fence boundaries.

        Rules
        -----
        - Never split inside a fenced code block or a markdown table.
        - Flush a chunk at every header (H1–H4) boundary.
        - When a new chunk starts below a header, prepend the most recent
          H1–H3 breadcrumb so the chunk retains navigational context.
        - Fall back to paragraph boundaries when a non-atomic section
          exceeds ``max_chars``.

        Returns a list of ``(chunk_text, section_path)`` tuples; never empty.
        ``section_path`` is a human-readable breadcrumb of the H1–H3 headers
        active at the start of each chunk, e.g. ``"Introduction > Background"``.
        Empty string if no headers precede the chunk.
        """
        import re as _re  # already imported at module level; local alias for static

        if len(text) <= max_chars:
            return [(text.strip(), "")] if text.strip() else [(text, "")]

        units = DocumentClassifierRouter._tokenise_markdown_units(text)
        chunks: list[tuple[str, str]] = []
        current_parts: list[str] = []
        current_len = 0
        last_headers: dict[int, str] = {}
        current_section_path: str = ""

        def flush() -> None:
            nonlocal current_parts, current_len
            body = "\n\n".join(p for p in current_parts if p).strip()
            if body:
                chunks.append((body, current_section_path))
            current_parts = []
            current_len = 0

        for unit_type, unit_text in units:
            unit_len = len(unit_text)

            if unit_type == "header":
                m = _re.match(r"^(#+)", unit_text)
                if m:
                    level = len(m.group(1))
                    last_headers[level] = unit_text.rstrip()
                    for k in list(last_headers):
                        if k > level:
                            del last_headers[k]
                if current_parts:
                    flush()
                current_section_path = DocumentClassifierRouter._section_path_string(last_headers)
                current_parts = [unit_text]
                current_len = unit_len
                continue

            needed = current_len + (2 if current_parts else 0) + unit_len
            if needed <= max_chars or not current_parts:
                current_parts.append(unit_text)
                current_len += (2 if len(current_parts) > 1 else 0) + unit_len
            else:
                flush()
                current_section_path = DocumentClassifierRouter._section_path_string(last_headers)
                ctx = DocumentClassifierRouter._build_header_context(last_headers)
                current_parts = ([ctx] if ctx else []) + [unit_text]
                current_len = (len(ctx) + 2 if ctx else 0) + unit_len

        flush()
        return chunks if chunks else [(text.strip(), "")]

    @staticmethod
    def _tokenise_markdown_units(text: str) -> list[tuple[str, str]]:
        """
        Break markdown text into a list of (type, content) tuples.

        Types: ``"header"``, ``"code_fence"``, ``"table"``, ``"paragraph"``

        Atomic units (``"code_fence"`` and ``"table"``) are never split by
        the caller.
        """
        import re as _re

        units: list[tuple[str, str]] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Fenced code block
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = "```" if stripped.startswith("```") else "~~~"
                block = [line]
                i += 1
                while i < len(lines):
                    block.append(lines[i])
                    if lines[i].strip().startswith(marker) and i > (len(block) - 2):
                        i += 1
                        break
                    i += 1
                units.append(("code_fence", "\n".join(block)))
                continue

            # Markdown table (contiguous pipe-prefixed rows)
            if stripped.startswith("|") and "|" in stripped:
                block = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|") and "|" in lines[i]:
                    block.append(lines[i])
                    i += 1
                units.append(("table", "\n".join(block)))
                continue

            # Header (H1–H6)
            if _re.match(r"^#{1,6}\s", line):
                units.append(("header", line))
                i += 1
                continue

            # Blank line — skip (paragraph boundaries are implicit)
            if not stripped:
                i += 1
                continue

            # Paragraph — collect until a blank line or structural element
            block = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if (
                    not ns
                    or _re.match(r"^#{1,6}\s", nxt)
                    or ns.startswith("```")
                    or ns.startswith("~~~")
                    or (ns.startswith("|") and "|" in ns)
                ):
                    break
                block.append(nxt)
                i += 1
            units.append(("paragraph", "\n".join(block)))

        return units

    @staticmethod
    def _build_header_context(last_headers: dict[int, str]) -> str:
        """Return a breadcrumb of the last H1–H3 headers for chunk context carry-forward."""
        lines = [last_headers[lvl] for lvl in sorted(last_headers) if lvl <= 3]
        return "\n".join(lines)

    @staticmethod
    def _section_path_string(last_headers: dict[int, str]) -> str:
        """Return a human-readable section path for metadata, e.g. 'Introduction > Background'.

        Strips leading ``#`` symbols so the value is suitable for storage as a
        metadata field and natural-language filter generation.
        """
        parts = [last_headers[lvl].lstrip("#").strip() for lvl in sorted(last_headers) if lvl <= 3]
        return " > ".join(p for p in parts if p)
