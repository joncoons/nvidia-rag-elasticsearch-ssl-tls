"""
Web Crawl Classification Agent for NVIDIA RAG Blueprint
========================================================

Combines three content extraction strategies per HTML page:

  1. BeautifulSoup4 prose extraction
     Fast, zero-GPU cost.  Recovers paragraphs, headings, code blocks, and
     lists in DOM order.  Sufficient for 90-95 % of technical documentation.

  2. Selenium full-page screenshot
     Captures JavaScript-rendered content that BS4 cannot reach.  Also
     provides the pixel canvas used by the nemoretriever-parse classifier.

  3. nemoretriever-parse two-pass classification (per detected element region)
     Pass 1  detection_only  — lightweight classification of the full-page
                               screenshot; returns element type + bbox for
                               every detected complex element.
     Pass 2  markdown_no_bbox — called only on PIL-cropped regions for
                               detected tables, charts, graphs, infographics.
                               Preserves structural context that BS4 loses.

Position-aware merging
----------------------
After loading the page, Selenium queries getBoundingClientRect() for every
visible prose element (headings, paragraphs, lists, code blocks).  This
produces a list of (y_top, text) tuples in document order.  The VLM results
carry the y_top of their bbox.  Both lists are sorted by y_top and interleaved
so that tables and charts appear in their correct position within the prose flow.

Output
------
  <output_dir>/
    pages/      <slug>.md          one merged Markdown file per crawled page
    screenshots/<slug>.png         full-page screenshots (kept for audit/debug)
  crawl_manifest.csv               per-page crawl record

  Optionally submits all .md files directly to the RAG ingestor API.

Usage
-----
  python web_crawl_classification_agent.py \\
      --start-url  https://docs.nvidia.com/ai-enterprise/ \\
      --output-dir ./crawl_output \\
      --nemo-parse-url http://nemoretriever-parse-ms:8000/v1/chat/completions \\
      --ingestor-url  http://localhost:8082 \\
      --collection    web-docs

  Run without --nemo-parse-url to use BS4-only mode (no GPU required).
  Run without --ingestor-url  to write files to disk only.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit

import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webcrawl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLEX_ELEMENT_TYPES: frozenset[str] = frozenset(
    {"table", "chart", "graph", "infographic", "figure", "diagram"}
)

NEMO_PARSE_MODEL_DEFAULT = "nvdev/nvidia/nemoretriever-parse"

# DOM tags whose text content is treated as prose blocks
PROSE_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th")

# Minimum pixel height for a detected element region to be worth cropping
MIN_REGION_HEIGHT_PX = 20

# How long to wait for JS-heavy pages to settle after driver.get()
PAGE_SETTLE_SECONDS = 3

# Extra padding (pixels) added around a detected bbox before cropping
CROP_PADDING_PX = 8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedElement:
    """A complex element found by nemoretriever-parse detection_only."""
    element_type: str
    # Absolute pixel coordinates in the screenshot image: (x1, y1, x2, y2)
    x1: float
    y1: float
    x2: float
    y2: float
    markdown: str = ""      # filled in after Pass 2


@dataclass(order=True)
class ContentBlock:
    """A unit of page content with its vertical position for merge sorting."""
    y_top: float
    content_type: str = field(compare=False)   # "prose" | element type string
    markdown: str = field(compare=False)


@dataclass
class PageResult:
    url: str
    slug: str
    method: str              # "bs4_only" | "bs4+vlm"
    detected_types: list[str]
    md_paths: list[str]      # one entry per pre-chunked .md file (usually just one)
    screenshot_path: str
    status: str              # "ok" | "error"
    page_title: str = ""
    section_h1: str = ""
    meta_description: str = ""
    crawl_depth: int = 0
    error: str = ""

    @property
    def md_path(self) -> str:
        """Return the first chunk path for backward-compatible single-file access."""
        return self.md_paths[0] if self.md_paths else ""

    @property
    def chunk_count(self) -> int:
        return len(self.md_paths)


# ---------------------------------------------------------------------------
# Markdown-aware chunker (shared by PageProcessor and ingestor submission)
# ---------------------------------------------------------------------------

# Target chunk size in characters (512 tokens × ~4 chars/token)
_CHUNK_MAX_CHARS: int = 2048
_CHUNK_OVERLAP_CHARS: int = 600


def _split_markdown(
    text: str, max_chars: int = _CHUNK_MAX_CHARS, overlap_chars: int = _CHUNK_OVERLAP_CHARS
) -> list[str]:
    """
    Split markdown text into chunks that respect header, table, and code-fence boundaries.

    - Never splits inside a fenced code block or a markdown table.
    - Flushes a chunk at every header boundary.
    - Prepends the last H1–H3 breadcrumb when starting a new chunk mid-section.
    - Falls back to paragraph boundaries for oversized prose sections.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else [text]

    units = _tokenise_markdown_units(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    last_headers: dict[int, str] = {}

    def flush() -> None:
        nonlocal current_parts, current_len
        body = "\n\n".join(p for p in current_parts if p).strip()
        if body:
            chunks.append(body)
        current_parts = []
        current_len = 0

    for unit_type, unit_text in units:
        unit_len = len(unit_text)

        if unit_type == "header":
            m = re.match(r"^(#+)", unit_text)
            if m:
                level = len(m.group(1))
                last_headers[level] = unit_text.rstrip()
                for k in list(last_headers):
                    if k > level:
                        del last_headers[k]
            if current_parts:
                flush()
            current_parts = [unit_text]
            current_len = unit_len
            continue

        needed = current_len + (2 if current_parts else 0) + unit_len
        if needed <= max_chars or not current_parts:
            current_parts.append(unit_text)
            current_len += (2 if len(current_parts) > 1 else 0) + unit_len
        else:
            flush()
            ctx = _build_header_context(last_headers)
            current_parts = ([ctx] if ctx else []) + [unit_text]
            current_len = (len(ctx) + 2 if ctx else 0) + unit_len

    flush()
    return chunks if chunks else [text.strip()]


def _tokenise_markdown_units(text: str) -> list[tuple[str, str]]:
    """Tokenise markdown into (type, content) tuples: header/code_fence/table/paragraph."""
    units: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

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

        if stripped.startswith("|") and "|" in stripped:
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|") and "|" in lines[i]:
                block.append(lines[i])
                i += 1
            units.append(("table", "\n".join(block)))
            continue

        if re.match(r"^#{1,6}\s", line):
            units.append(("header", line))
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        block = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            ns = nxt.strip()
            if (
                not ns
                or re.match(r"^#{1,6}\s", nxt)
                or ns.startswith("```")
                or ns.startswith("~~~")
                or (ns.startswith("|") and "|" in ns)
            ):
                break
            block.append(nxt)
            i += 1
        units.append(("paragraph", "\n".join(block)))

    return units


def _build_header_context(last_headers: dict[int, str]) -> str:
    """Return H1–H3 breadcrumb for chunk context carry-forward."""
    lines = [last_headers[lvl] for lvl in sorted(last_headers) if lvl <= 3]
    return "\n".join(lines)


def _compute_content_hash(filepath: str) -> str:
    """Return the hex SHA-256 digest of the file, or '' on read error."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# nemoretriever-parse client
# ---------------------------------------------------------------------------

class NemoParseClient:
    """
    Thin wrapper around the nemoretriever-parse inference endpoint.

    Mirrors the private helpers in DocumentClassifierRouter but exposes the
    bbox list from detection_only so callers can perform region crops.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = NEMO_PARSE_MODEL_DEFAULT,
        api_key: str = "",
        detect_max_tokens: int = 1024,
        parse_max_tokens: int = 4096,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.detect_max_tokens = detect_max_tokens
        self.parse_max_tokens = parse_max_tokens

        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, image: PILImage.Image) -> list[DetectedElement]:
        """
        Run detection_only on the image.  Returns a list of DetectedElement
        objects with absolute pixel coordinates in the image.
        """
        b64, mime = self._pil_to_base64(image)
        try:
            result = self._call(b64, mime, "detection_only", self.detect_max_tokens)
        except Exception as exc:
            logger.warning("detection_only failed: %s", exc)
            return []
        return self._parse_detections(result, image.width, image.height)

    def parse_region(self, region: PILImage.Image) -> str:
        """Run markdown_no_bbox on a cropped region image; return markdown text."""
        b64, mime = self._pil_to_base64(region)
        try:
            result = self._call(b64, mime, "markdown_no_bbox", self.parse_max_tokens)
        except Exception as exc:
            logger.warning("markdown_no_bbox failed: %s", exc)
            return ""
        return self._extract_markdown(result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(img: PILImage.Image) -> tuple[str, str]:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/png"

    def _call(self, b64: str, mime: str, tool: str, max_tokens: int) -> dict:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": f'<img src="data:{mime};base64,{b64}" />'}],
            "tools": [{"type": "function", "function": {"name": tool}}],
            "tool_choice": {"type": "function", "function": {"name": tool}},
            "max_tokens": max_tokens,
        }
        resp = self._session.post(self.endpoint_url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_detections(
        result: dict, img_w: int, img_h: int
    ) -> list[DetectedElement]:
        """
        Extract DetectedElement objects from a detection_only response.
        Handles both normalised [0,1] and absolute-pixel bbox formats.
        """
        elements: list[DetectedElement] = []
        try:
            for choice in result.get("choices", []):
                for tc in (choice.get("message") or {}).get("tool_calls", []):
                    raw = (tc.get("function") or {}).get("arguments", "[]")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    items = args if isinstance(args, list) else [args]
                    for item in items:
                        el_type = str(item.get("type", "")).lower().strip()
                        bbox = item.get("bbox") or item.get("bounding_box")
                        if not el_type or not bbox or len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        # Normalise if coordinates are in [0,1] range
                        if max(x1, y1, x2, y2) <= 1.0:
                            x1, y1, x2, y2 = (
                                x1 * img_w, y1 * img_h,
                                x2 * img_w, y2 * img_h,
                            )
                        elements.append(DetectedElement(
                            element_type=el_type,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                        ))
        except Exception as exc:
            logger.warning("Failed to parse detections: %s", exc)
        return elements

    @staticmethod
    def _extract_markdown(result: dict) -> str:
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
            logger.warning("Failed to extract markdown: %s", exc)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Page processor
# ---------------------------------------------------------------------------

class PageProcessor:
    """
    Processes a single web page into a position-ordered ContentBlock list.

    Strategy
    --------
    1. Extract prose ContentBlocks with their page Y positions via Selenium JS.
    2. Screenshot the full page.
    3. If a NemoParseClient is available:
       a. Run detection_only on the screenshot.
       b. For each complex element: crop the region, run markdown_no_bbox.
       c. Emit a ContentBlock for each VLM result at its bbox y_top.
    4. Sort all ContentBlocks by y_top and serialise to Markdown.
    """

    # JS that returns [{tag, y_top, text}] for all visible prose elements
    _PROSE_POSITIONS_JS = """
    const tags = ['h1','h2','h3','h4','h5','h6','p','li','pre','blockquote','td','th'];
    const results = [];
    for (const tag of tags) {
        for (const el of document.querySelectorAll(tag)) {
            const rect = el.getBoundingClientRect();
            const text = el.innerText ? el.innerText.trim() : '';
            if (text.length > 0 && rect.height > 0) {
                results.push({tag: tag, y_top: rect.top + window.scrollY, text: text});
            }
        }
    }
    return results;
    """

    def __init__(self, nemo_client: "NemoParseClient | None" = None) -> None:
        self.nemo_client = nemo_client

    def process(
        self,
        url: str,
        driver: webdriver.Chrome,
        screenshot_path: str,
    ) -> tuple[str, str, list[str]]:
        """
        Process a loaded page.  Caller must have already called driver.get(url)
        and waited for the page to settle.

        Returns
        -------
        (merged_markdown, method, detected_type_names)
            method is "bs4_only" or "bs4+vlm"
        """
        # ----------------------------------------------------------------
        # Step 1 — collect prose blocks with Y positions via JS
        # ----------------------------------------------------------------
        prose_blocks = self._extract_prose_blocks(driver)

        # ----------------------------------------------------------------
        # Step 2 — full-page screenshot
        # ----------------------------------------------------------------
        screenshot = self._take_screenshot(driver, screenshot_path)

        # ----------------------------------------------------------------
        # Step 3 — VLM classification and region extraction (if available)
        # ----------------------------------------------------------------
        vlm_blocks: list[ContentBlock] = []
        detected_type_names: list[str] = []
        method = "bs4_only"

        if self.nemo_client is not None and screenshot is not None:
            detections = self.nemo_client.detect(screenshot)
            complex_detections = [
                d for d in detections
                if d.element_type in COMPLEX_ELEMENT_TYPES
            ]
            if complex_detections:
                method = "bs4+vlm"
                detected_type_names = list({d.element_type for d in complex_detections})
                logger.info(
                    "  Detected complex elements: %s",
                    detected_type_names,
                )
                for det in complex_detections:
                    region_md = self._extract_region(screenshot, det)
                    if region_md:
                        vlm_blocks.append(ContentBlock(
                            y_top=det.y1,
                            content_type=det.element_type,
                            markdown=self._annotate_vlm_block(det.element_type, region_md),
                        ))

        # ----------------------------------------------------------------
        # Step 4 — merge by Y position
        # ----------------------------------------------------------------
        all_blocks = sorted(prose_blocks + vlm_blocks)
        merged_md = self._blocks_to_markdown(all_blocks, url)
        return merged_md, method, detected_type_names

    # ------------------------------------------------------------------
    # Private — prose extraction
    # ------------------------------------------------------------------

    def _extract_prose_blocks(self, driver: webdriver.Chrome) -> list[ContentBlock]:
        """Use JS to collect prose elements with their page Y coordinates."""
        blocks: list[ContentBlock] = []
        try:
            raw = driver.execute_script(self._PROSE_POSITIONS_JS)
            seen: set[str] = set()
            for item in raw or []:
                text = _clean_text(str(item.get("text", "")))
                if not text or text in seen:
                    continue
                seen.add(text)
                tag = item.get("tag", "p")
                y_top = float(item.get("y_top", 0))
                md_text = _html_tag_to_md_prefix(tag) + text
                blocks.append(ContentBlock(
                    y_top=y_top,
                    content_type="prose",
                    markdown=md_text,
                ))
        except Exception as exc:
            logger.debug("JS prose extraction failed (%s) — falling back to page_source BS4", exc)
            blocks = self._bs4_fallback(driver)
        return blocks

    @staticmethod
    def _bs4_fallback(driver: webdriver.Chrome) -> list[ContentBlock]:
        """Pure BS4 extraction without Y positions — used as JS fallback."""
        blocks: list[ContentBlock] = []
        try:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for tag in soup.find_all(PROSE_TAGS):
                text = _clean_text(tag.get_text(separator=" ", strip=True))
                if text:
                    blocks.append(ContentBlock(
                        y_top=float(len(blocks)),  # DOM order as proxy
                        content_type="prose",
                        markdown=_html_tag_to_md_prefix(tag.name) + text,
                    ))
        except Exception as exc:
            logger.debug("BS4 fallback failed: %s", exc)
        return blocks

    # ------------------------------------------------------------------
    # Private — screenshot
    # ------------------------------------------------------------------

    @staticmethod
    def _take_screenshot(
        driver: webdriver.Chrome, screenshot_path: str
    ) -> PILImage.Image | None:
        """Resize window to full page and capture screenshot; return PIL Image."""
        try:
            width = driver.execute_script(
                "return Math.max(document.body.scrollWidth,"
                " document.documentElement.scrollWidth);"
            )
            height = driver.execute_script(
                "return Math.max(document.body.scrollHeight,"
                " document.documentElement.scrollHeight);"
            )
            driver.set_window_size(width, height)
            time.sleep(0.5)   # let layout reflow after resize
            driver.save_screenshot(screenshot_path)
            return PILImage.open(screenshot_path).convert("RGB")
        except Exception as exc:
            logger.warning("Screenshot failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private — region extraction
    # ------------------------------------------------------------------

    def _extract_region(
        self, screenshot: PILImage.Image, det: DetectedElement
    ) -> str:
        """Crop the detected element's bbox from the screenshot and VLM-parse it."""
        img_w, img_h = screenshot.size
        x1 = max(0, int(det.x1) - CROP_PADDING_PX)
        y1 = max(0, int(det.y1) - CROP_PADDING_PX)
        x2 = min(img_w, int(det.x2) + CROP_PADDING_PX)
        y2 = min(img_h, int(det.y2) + CROP_PADDING_PX)

        if (y2 - y1) < MIN_REGION_HEIGHT_PX or (x2 - x1) < 10:
            logger.debug("Skipping tiny region %s (%dx%d)", det.element_type, x2 - x1, y2 - y1)
            return ""

        region = screenshot.crop((x1, y1, x2, y2))
        markdown = self.nemo_client.parse_region(region)
        logger.debug(
            "  VLM parsed %s region (%dx%d) → %d chars",
            det.element_type, x2 - x1, y2 - y1, len(markdown),
        )
        return markdown

    # ------------------------------------------------------------------
    # Private — markdown assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_vlm_block(element_type: str, markdown: str) -> str:
        """Wrap VLM-extracted content with a provenance comment."""
        label = element_type.capitalize()
        return f"<!-- {label}: nemoretriever-parse -->\n\n{markdown}"

    @staticmethod
    def _blocks_to_markdown(blocks: list[ContentBlock], url: str) -> str:
        """Serialise sorted ContentBlocks to a single Markdown string."""
        parts = [f"<!-- Source: {url} -->\n"]
        prev_type = None
        for block in blocks:
            if prev_type == "prose" and block.content_type != "prose":
                parts.append("\n---\n")   # visual separator before VLM sections
            elif prev_type != "prose" and block.content_type == "prose":
                parts.append("\n---\n")   # visual separator after VLM sections
            parts.append(block.markdown)
            prev_type = block.content_type
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class WebCrawlClassificationAgent:
    """
    Domain-scoped web crawler that combines BS4, Selenium, and
    nemoretriever-parse to produce ingestion-ready Markdown files.

    Parameters
    ----------
    start_url       : Seed URL; crawl is restricted to the same netloc.
    output_dir      : Root directory for pages/, screenshots/, and manifest CSV.
    nemo_parse_url  : nemoretriever-parse endpoint URL.  If empty, BS4-only mode.
    nemo_model      : Model identifier forwarded to the API.
    nemo_api_key    : Bearer token for the endpoint.
    ingestor_url    : Optional RAG ingestor base URL for direct submission.
    collection_name : Collection name used when submitting to the ingestor.
    headless        : Run Chrome in headless mode (default True).
    max_pages       : Stop after crawling this many HTML pages (0 = unlimited).
    """

    def __init__(
        self,
        start_url: str,
        output_dir: str = "./crawl_output",
        nemo_parse_url: str = "",
        nemo_model: str = NEMO_PARSE_MODEL_DEFAULT,
        nemo_api_key: str = "",
        ingestor_url: str = "",
        collection_name: str = "web-docs",
        headless: bool = True,
        max_pages: int = 0,
    ) -> None:
        self.start_url = start_url.rstrip("/")
        self.start_netloc = urlparse(start_url).netloc
        self.output_dir = Path(output_dir)
        self.ingestor_url = ingestor_url.rstrip("/") if ingestor_url else ""
        self.collection_name = collection_name
        self.max_pages = max_pages

        # Output directories
        self.pages_dir = self.output_dir / "pages"
        self.screenshots_dir = self.output_dir / "screenshots"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "crawl_manifest.csv"

        # nemoretriever-parse client (None → BS4-only mode)
        self.nemo_client: NemoParseClient | None = None
        if nemo_parse_url:
            self.nemo_client = NemoParseClient(
                endpoint_url=nemo_parse_url,
                model_name=nemo_model,
                api_key=nemo_api_key,
            )
            logger.info("nemoretriever-parse endpoint: %s", nemo_parse_url)
        else:
            logger.info("No nemo-parse-url provided — running in BS4-only mode")

        self.processor = PageProcessor(nemo_client=self.nemo_client)

        # Selenium setup
        chrome_options = Options()
        chrome_options.binary_location = "/usr/bin/google-chrome"
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--window-size=1920,8000")
        user_data_dir = tempfile.mkdtemp(prefix="chromesession-")
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(2)

        # Crawl state
        self.visited: set[str] = set()
        self.queue: list[str] = [self.start_url]
        self.results: list[PageResult] = []
        # Lineage tracking
        self.crawl_session_id: str = str(uuid.uuid4())
        self.crawl_depth: dict[str, int] = {self.start_url: 0}
        logger.info("Crawl session ID: %s", self.crawl_session_id)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[PageResult]:
        """
        Crawl the domain, process each HTML page, and optionally submit to
        the RAG ingestor.  Returns the list of PageResult records.
        """
        logger.info("Crawl starting: %s", self.start_url)
        pages_processed = 0

        try:
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "url", "slug", "method", "detected_types",
                    "md_path", "chunk_count", "screenshot_path",
                    "section_h1", "meta_description",
                    "crawl_depth", "crawl_session_id", "domain",
                    "status", "error",
                ])
                writer.writeheader()

                while self.queue:
                    if self.max_pages and pages_processed >= self.max_pages:
                        logger.info("Reached max_pages limit (%d)", self.max_pages)
                        break

                    url = self.queue.pop(0)
                    if url in self.visited:
                        continue
                    if urlparse(url).netloc != self.start_netloc:
                        continue

                    self.visited.add(url)
                    logger.info("[%d] Processing: %s", pages_processed + 1, url)

                    result = self._process_page(url)
                    self.results.append(result)
                    writer.writerow({
                        "url": result.url,
                        "slug": result.slug,
                        "method": result.method,
                        "detected_types": "|".join(result.detected_types),
                        "md_path": result.md_path,
                        "chunk_count": result.chunk_count,
                        "screenshot_path": result.screenshot_path,
                        "section_h1": result.section_h1,
                        "meta_description": result.meta_description,
                        "crawl_depth": result.crawl_depth,
                        "crawl_session_id": self.crawl_session_id,
                        "domain": self.start_netloc,
                        "status": result.status,
                        "error": result.error,
                    })
                    csvfile.flush()
                    pages_processed += 1

        finally:
            self.driver.quit()

        logger.info(
            "Crawl complete: %d pages processed, %d errors",
            pages_processed,
            sum(1 for r in self.results if r.status == "error"),
        )
        self._log_summary()

        if self.ingestor_url:
            self._submit_to_ingestor()

        return self.results

    # ------------------------------------------------------------------
    # Private — page processing
    # ------------------------------------------------------------------

    def _process_page(self, url: str) -> PageResult:
        slug = _url_to_slug(url)
        md_path = str(self.pages_dir / f"{slug}.md")
        screenshot_path = str(self.screenshots_dir / f"{slug}.png")

        try:
            # Load page with Selenium (handles JS rendering)
            self.driver.get(url)
            time.sleep(PAGE_SETTLE_SECONDS)

            # Discover and enqueue links before processing content
            self._enqueue_links()

            # Get page title for Markdown header
            title = self.driver.title or slug

            # Extract H1 and meta description for metadata fields (single parse)
            _page_soup = BeautifulSoup(self.driver.page_source, "html.parser")
            _h1 = _page_soup.find("h1")
            section_h1 = _h1.get_text(strip=True) if _h1 else ""
            _meta_tag = (_page_soup.find("meta", {"name": "description"}) or
                         _page_soup.find("meta", {"property": "og:description"}))
            meta_description = _meta_tag.get("content", "").strip() if _meta_tag else ""

            # Process through PageProcessor (prose + optional VLM)
            merged_md, method, detected_types = self.processor.process(
                url, self.driver, screenshot_path
            )

            # Pre-chunk the markdown and write one file per semantic chunk
            full_md = f"# {title}\n\n{merged_md}"
            chunks = _split_markdown(full_md)
            md_paths: list[str] = []
            for idx, chunk_text in enumerate(chunks):
                if len(chunks) == 1:
                    chunk_path = md_path  # keep original slug.md name for single chunks
                else:
                    chunk_path = str(self.pages_dir / f"{slug}_{idx + 1:03d}.md")
                with open(chunk_path, "w", encoding="utf-8") as fh:
                    fh.write(chunk_text)
                md_paths.append(chunk_path)

            return PageResult(
                url=url,
                slug=slug,
                method=method,
                detected_types=detected_types,
                md_paths=md_paths,
                screenshot_path=screenshot_path,
                status="ok",
                page_title=title,
                section_h1=section_h1,
                meta_description=meta_description,
                crawl_depth=self.crawl_depth.get(url, 0),
            )

        except Exception as exc:
            logger.warning("Error processing %s: %s", url, exc)
            return PageResult(
                url=url,
                slug=slug,
                method="error",
                detected_types=[],
                md_paths=[],
                screenshot_path="",
                status="error",
                crawl_depth=self.crawl_depth.get(url, 0),
                error=str(exc),
            )

    def _enqueue_links(self) -> None:
        """Collect all same-domain links from the current page and add to queue."""
        try:
            current_depth = self.crawl_depth.get(self.driver.current_url, 0)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            for tag in soup.find_all("a", href=True):
                link = urljoin(self.driver.current_url, tag["href"])
                parsed = urlparse(link)
                # Strip fragment; keep path + query
                clean = parsed._replace(fragment="").geturl()
                if (
                    parsed.netloc == self.start_netloc
                    and clean not in self.visited
                    and clean not in self.queue
                    and not _is_binary_url(link)
                ):
                    self.queue.append(clean)
                    if clean not in self.crawl_depth:
                        self.crawl_depth[clean] = current_depth + 1
        except Exception as exc:
            logger.debug("Link discovery failed: %s", exc)

    # ------------------------------------------------------------------
    # Private — ingestor submission
    # ------------------------------------------------------------------

    def _submit_to_ingestor(self) -> None:
        """POST all successfully generated .md chunk files to the RAG ingestor API."""
        # Collect all chunk paths and their parent PageResult for lineage metadata
        chunk_to_result: dict[str, PageResult] = {}
        chunk_to_index: dict[str, int] = {}
        for r in self.results:
            if r.status == "ok":
                for idx, p in enumerate(r.md_paths):
                    if p:
                        chunk_to_result[p] = r
                        chunk_to_index[p] = idx

        md_files = list(chunk_to_result.keys())
        if not md_files:
            logger.info("No markdown files to submit to ingestor")
            return

        logger.info(
            "Submitting %d chunk file(s) to ingestor at %s (collection: %s, session: %s)",
            len(md_files),
            self.ingestor_url,
            self.collection_name,
            self.crawl_session_id,
        )

        ingested_at = datetime.now(timezone.utc).isoformat()

        # Submit in batches of 16 to stay within the ingestor's default batch size
        batch_size = 16
        for batch_start in range(0, len(md_files), batch_size):
            batch = md_files[batch_start: batch_start + batch_size]
            try:
                files = [
                    ("documents", (Path(fp).name, open(fp, "rb"), "text/markdown"))
                    for fp in batch
                ]

                # Build per-file lineage metadata
                custom_metadata = []
                for fp in batch:
                    result = chunk_to_result.get(fp)
                    meta: dict = {
                        "content_hash": _compute_content_hash(fp),
                        "ingested_at": ingested_at,
                        "document_type": "md",
                        "pipeline_type": (
                            "web_crawl_bs4_vlm"
                            if result and result.method == "bs4+vlm"
                            else "web_crawl_bs4"
                        ),
                        "crawl_session_id": self.crawl_session_id,
                        "domain": self.start_netloc,
                        "last_crawled_at": ingested_at,
                    }
                    meta["chunk_index"] = chunk_to_index.get(fp, 0)
                    meta["total_chunks"] = result.chunk_count if result else 1
                    if result:
                        meta["source_uri"] = result.url
                        meta["crawl_depth"] = result.crawl_depth
                        if result.page_title:
                            meta["page_title"] = result.page_title
                        if result.section_h1:
                            meta["section_h1"] = result.section_h1
                        if result.meta_description:
                            meta["meta_description"] = result.meta_description
                        if result.detected_types:
                            meta["detected_element_types"] = result.detected_types
                    custom_metadata.append({
                        "filename": Path(fp).name,
                        "metadata": meta,
                    })

                payload = json.dumps({
                    "collection_name": self.collection_name,
                    "blocking": False,
                    # VLM extraction already done; standard NV-Ingest pipeline for text splitting
                    "use_nemoretriever_parse": False,
                    "split_options": {"chunk_size": 512, "chunk_overlap": 150},
                    "custom_metadata": custom_metadata,
                })
                resp = requests.post(
                    f"{self.ingestor_url}/documents",
                    files=files,
                    data={"data": payload},
                    timeout=60,
                )
                resp.raise_for_status()
                response_data = resp.json()
                task_id = response_data.get("task_id", "")
                logger.info(
                    "  Batch %d–%d submitted (task_id: %s)",
                    batch_start + 1,
                    batch_start + len(batch),
                    task_id or "n/a",
                )
                for _, file_tuple in files:
                    file_tuple[1].close()
            except Exception as exc:
                logger.error("Ingestor submission failed for batch starting at %d: %s", batch_start, exc)

    # ------------------------------------------------------------------
    # Private — summary logging
    # ------------------------------------------------------------------

    def _log_summary(self) -> None:
        total = len(self.results)
        ok = sum(1 for r in self.results if r.status == "ok")
        vlm = sum(1 for r in self.results if r.method == "bs4+vlm")
        bs4_only = sum(1 for r in self.results if r.method == "bs4_only")
        errors = total - ok

        type_counts: dict[str, int] = {}
        for r in self.results:
            for t in r.detected_types:
                type_counts[t] = type_counts.get(t, 0) + 1

        logger.info("=" * 60)
        logger.info("Crawl summary")
        logger.info("  Total pages   : %d", total)
        logger.info("  Successful    : %d", ok)
        logger.info("  Errors        : %d", errors)
        logger.info("  BS4-only      : %d", bs4_only)
        logger.info("  BS4 + VLM     : %d", vlm)
        if type_counts:
            logger.info("  Detected types:")
            for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                logger.info("    %-20s %d pages", t, count)
        logger.info("  Manifest      : %s", self.manifest_path)
        logger.info("  Pages dir     : %s", self.pages_dir)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise Unicode, collapse whitespace, strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return " ".join(text.split())


def _html_tag_to_md_prefix(tag: str) -> str:
    """Return the Markdown prefix for a given HTML tag name."""
    prefixes = {"h1": "# ", "h2": "## ", "h3": "### ",
                "h4": "#### ", "h5": "##### ", "h6": "###### ",
                "pre": "```\n", "blockquote": "> "}
    return prefixes.get(tag, "")


def _url_to_slug(url: str) -> str:
    """Convert a URL to a safe filesystem slug."""
    parsed = urlsplit(url)
    path = (parsed.netloc + parsed.path).strip("/")
    slug = re.sub(r"[^\w\-]", "_", path)
    return slug[:200] or "index"


def _is_binary_url(url: str) -> bool:
    """Return True if the URL points to a non-HTML binary file."""
    binary_exts = {
        "pdf", "xls", "xlsx", "csv", "jpg", "jpeg", "png", "gif",
        "bmp", "svg", "mp4", "avi", "mov", "wmv", "webm", "mp3",
        "wav", "aac", "flac", "ogg", "doc", "docx", "ppt", "pptx",
        "zip", "tar", "gz", "exe", "dmg",
    }
    ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".")
    return ext in binary_exts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Web Crawl Classification Agent — NVIDIA RAG Blueprint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-url", required=True,
                   help="Seed URL; crawl stays within the same netloc")
    p.add_argument("--output-dir", default="./crawl_output",
                   help="Root output directory for pages, screenshots, and manifest")
    p.add_argument("--nemo-parse-url", default="",
                   help="nemoretriever-parse endpoint URL "
                        "(e.g. http://nemoretriever-parse-ms:8000/v1/chat/completions). "
                        "Omit to run in BS4-only mode.")
    p.add_argument("--nemo-model", default=NEMO_PARSE_MODEL_DEFAULT,
                   help="Model identifier sent in the nemoretriever-parse request")
    p.add_argument("--nemo-api-key", default="",
                   help="Bearer token for the nemoretriever-parse endpoint")
    p.add_argument("--ingestor-url", default="",
                   help="RAG ingestor base URL for direct submission "
                        "(e.g. http://localhost:8082). Omit to write files to disk only.")
    p.add_argument("--collection", default="web-docs",
                   help="Collection name used when submitting to the ingestor")
    p.add_argument("--max-pages", type=int, default=0,
                   help="Stop after this many HTML pages (0 = unlimited)")
    p.add_argument("--no-headless", action="store_true",
                   help="Show Chrome browser window (useful for debugging)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    agent = WebCrawlClassificationAgent(
        start_url=args.start_url,
        output_dir=args.output_dir,
        nemo_parse_url=args.nemo_parse_url,
        nemo_model=args.nemo_model,
        nemo_api_key=args.nemo_api_key,
        ingestor_url=args.ingestor_url,
        collection_name=args.collection,
        headless=not args.no_headless,
        max_pages=args.max_pages,
    )
    results = agent.run()
    sys.exit(0 if all(r.status == "ok" for r in results) else 1)
