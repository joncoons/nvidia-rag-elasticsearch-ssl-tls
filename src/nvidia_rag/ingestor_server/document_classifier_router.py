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

Two-pass pipeline per PDF page:
  Pass 1 (detection_only)   — lightweight classification call to identify element types.
  Pass 2 (markdown_no_bbox) — full VLM parse, called only on pages that contain complex
                               data presentation elements.

Complex element trigger types (any one is sufficient to route the document):
    table, chart, graph, infographic, figure, diagram

Routing behaviour (document-level):
    If ANY page contains a complex element the entire document is processed by
    nemoretriever-parse and a temporary Markdown file is written.  The caller
    substitutes this Markdown file in the NV-Ingest file list.  If no complex
    elements are found the caller proceeds with the standard NV-Ingest pipeline.

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
import json
import logging
import os
import tempfile
from pathlib import Path

import requests
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMO_PARSE_MODEL_DEFAULT = "nvdev/nvidia/nemoretriever-parse"

# Element types returned by detection_only that indicate complex data
# presentation elements warranting VLM-quality markdown extraction.
COMPLEX_ELEMENT_TYPES: frozenset[str] = frozenset(
    {"table", "chart", "graph", "infographic", "figure", "diagram"}
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DocumentClassifierRouter:
    """
    Classifies PDF pages with nemoretriever-parse ``detection_only`` and routes
    documents that contain complex data elements to ``markdown_no_bbox`` for
    high-quality structured Markdown extraction.

    Parameters
    ----------
    endpoint_url : str
        Full URL of the nemoretriever-parse inference endpoint, e.g.
        ``http://nemoretriever-parse-ms:8000/v1/chat/completions``.
    model_name : str
        Model identifier forwarded in the API payload.
    api_key : str
        Bearer token for the inference endpoint (empty string → no auth header).
    detect_max_tokens : int
        ``max_tokens`` cap for the cheap detection pass (default 512).
    parse_max_tokens : int
        ``max_tokens`` cap for the full markdown extraction pass (default 4096).
    dpi : int
        DPI used when rasterising PDF pages (default 200).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = NEMO_PARSE_MODEL_DEFAULT,
        api_key: str = "",
        detect_max_tokens: int = 512,
        parse_max_tokens: int = 4096,
        dpi: int = 200,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.detect_max_tokens = detect_max_tokens
        self.parse_max_tokens = parse_max_tokens
        self.dpi = dpi

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
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_document(self, filepath: str) -> str | None:
        """
        Classify and (if warranted) parse a single PDF file.

        Returns
        -------
        str
            Path to a temporary ``.md`` file containing the nemoretriever-parse
            output for all pages.  **The caller is responsible for deleting this
            file after it has been submitted to NV-Ingest.**
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
        except Exception as exc:
            logger.warning(
                "Could not rasterise PDF '%s': %s — falling back to standard pipeline",
                filepath,
                exc,
            )
            return None

        # ----------------------------------------------------------------
        # Pass 1 — lightweight classification (detection_only)
        # ----------------------------------------------------------------
        has_complex = False
        page_is_complex: list[bool] = []

        for i, page_img in enumerate(pages):
            b64, mime = self._pil_to_base64(page_img)
            detected = self._classify_page(b64, mime)
            is_complex = bool(detected & COMPLEX_ELEMENT_TYPES)
            page_is_complex.append(is_complex)
            if is_complex:
                has_complex = True
                logger.info(
                    "Complex element(s) detected on page %d of '%s': %s",
                    i + 1,
                    os.path.basename(filepath),
                    detected,
                )

        if not has_complex:
            logger.info(
                "No complex elements in '%s' — standard NV-Ingest pipeline will be used",
                os.path.basename(filepath),
            )
            return None

        logger.info(
            "Routing '%s' through nemoretriever-parse (%d pages total)",
            os.path.basename(filepath),
            len(pages),
        )

        # ----------------------------------------------------------------
        # Pass 2 — full markdown extraction (markdown_no_bbox)
        # All pages are parsed so that context around complex elements is
        # preserved; simpler pages cost little at this DPI.
        # ----------------------------------------------------------------
        output_parts: list[str] = []

        for i, page_img in enumerate(pages):
            b64, mime = self._pil_to_base64(page_img)
            try:
                markdown = self._parse_page(b64, mime)
                if markdown:
                    output_parts.append(f"<!-- Page {i + 1} -->\n\n{markdown}")
                else:
                    logger.debug("Empty markdown returned for page %d", i + 1)
            except Exception as exc:
                logger.warning(
                    "nemoretriever-parse failed on page %d of '%s': %s",
                    i + 1,
                    filepath,
                    exc,
                )

        if not output_parts:
            logger.warning(
                "nemoretriever-parse returned no usable content for '%s' — "
                "falling back to standard pipeline",
                filepath,
            )
            return None

        # ----------------------------------------------------------------
        # Write temp Markdown file — NV-Ingest will embed this as text
        # ----------------------------------------------------------------
        stem = Path(filepath).stem
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix=f"{stem}_nemoparse_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(f"# {stem}\n\n")
                fh.write("\n\n---\n\n".join(output_parts))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "nemoretriever-parse output for '%s' written to: %s",
            os.path.basename(filepath),
            tmp_path,
        )
        return tmp_path

    def route_documents(self, filepaths: list[str]) -> dict[str, str | None]:
        """
        Classify and route a batch of file paths.

        Returns
        -------
        dict[str, str | None]
            Maps each original file path to either:
              - A temp ``.md`` path (caller must delete after ingest), or
              - ``None`` (no complex elements; use standard NV-Ingest pipeline).
        """
        results: dict[str, str | None] = {}
        for fp in filepaths:
            results[fp] = self.route_document(fp)
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
    # Private helpers — nemoretriever-parse API calls
    # ------------------------------------------------------------------

    def _call_nemo_parse(
        self, b64: str, mime: str, tool: str, max_tokens: int
    ) -> dict:
        """Issue a single nemoretriever-parse request and return the parsed JSON."""
        media_tag = f'<img src="data:{mime};base64,{b64}" />'
        tool_spec = [{"type": "function", "function": {"name": tool}}]
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": media_tag}],
            "tools": tool_spec,
            "tool_choice": {"type": "function", "function": {"name": tool}},
            "max_tokens": max_tokens,
        }
        resp = self._session.post(self.endpoint_url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()

    def _classify_page(self, b64: str, mime: str) -> set[str]:
        """Run ``detection_only`` and return the set of detected type strings."""
        try:
            result = self._call_nemo_parse(
                b64, mime, "detection_only", self.detect_max_tokens
            )
        except Exception as exc:
            logger.warning("detection_only call failed: %s", exc)
            return set()
        return self._extract_detected_types(result)

    def _parse_page(self, b64: str, mime: str) -> str:
        """Run ``markdown_no_bbox`` and return the extracted markdown string."""
        result = self._call_nemo_parse(
            b64, mime, "markdown_no_bbox", self.parse_max_tokens
        )
        return self._extract_markdown_text(result)

    # ------------------------------------------------------------------
    # Private helpers — response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_detected_types(result: dict) -> set[str]:
        """Parse a ``detection_only`` response into a set of element type strings."""
        detected: set[str] = set()
        try:
            for choice in result.get("choices", []):
                for tc in (choice.get("message") or {}).get("tool_calls", []):
                    raw = (tc.get("function") or {}).get("arguments", "[]")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    items = args if isinstance(args, list) else [args]
                    for item in items:
                        el_type = str(item.get("type", "")).lower().strip()
                        if el_type:
                            detected.add(el_type)
        except Exception as exc:
            logger.warning("Failed to parse detection_only response: %s", exc)
        return detected

    @staticmethod
    def _extract_markdown_text(result: dict) -> str:
        """Extract text content from a ``markdown_no_bbox`` response."""
        parts: list[str] = []
        try:
            for choice in result.get("choices", []):
                for tc in (choice.get("message") or {}).get("tool_calls", []):
                    raw = (tc.get("function") or {}).get("arguments", "{}")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    items = args if isinstance(args, list) else [args]
                    for item in items:
                        text = str(item.get("text", "")).strip()
                        if text:
                            parts.append(text)
        except Exception as exc:
            logger.warning("Failed to extract markdown from response: %s", exc)
        return "\n\n".join(parts)
