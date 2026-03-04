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
XML → Markdown pre-processor for the NVIDIA RAG ingestor.

Converts XML content to clean Markdown before ingestion so that the standard
nv-ingest text pipeline (chunking → embedding) can process it.  Schema
detection is automatic:

  - RSS 2.0          root tag ``rss``
  - Atom 1.0         root tag ``feed`` in the Atom namespace
  - XML Sitemap      root tag ``urlset`` or ``sitemapindex``
  - Generic XML      everything else — top-level elements become sections

Usage::

    from nvidia_rag.utils.xml_preprocessor import xml_to_markdown

    with open("feed.xml", "rb") as fh:
        md = xml_to_markdown(fh.read())
"""

import logging
import re

logger = logging.getLogger(__name__)

# Well-known XML namespace URIs
_NS_ATOM = "http://www.w3.org/2005/Atom"
_NS_CONTENT = "http://purl.org/rss/1.0/modules/content/"  # <content:encoded>
_NS_DC = "http://purl.org/dc/elements/1.1/"               # <dc:creator> etc.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xml_to_markdown(content: bytes) -> str:
    """
    Parse *content* as XML and return a Markdown string.

    Automatically detects RSS 2.0, Atom 1.0, XML Sitemap, and generic XML.
    Returns an empty string if parsing fails entirely.
    """
    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml is not installed; cannot pre-process XML files")
        return ""

    try:
        root = etree.fromstring(content)
    except etree.XMLSyntaxError:
        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(content, parser=parser)
        except Exception as exc:
            logger.warning("Failed to parse XML content: %s", exc)
            return ""

    tag = _local_name(root.tag)
    ns = _namespace(root.tag)

    if tag == "rss":
        logger.debug("XML detected as RSS 2.0")
        return _rss_to_markdown(root)
    elif tag == "feed" and ns == _NS_ATOM:
        logger.debug("XML detected as Atom 1.0")
        return _atom_to_markdown(root)
    elif tag in ("urlset", "sitemapindex"):
        logger.debug("XML detected as Sitemap")
        return _sitemap_to_markdown(root)
    else:
        logger.debug("XML detected as generic (root: %s)", tag)
        return _generic_to_markdown(root)


# ---------------------------------------------------------------------------
# Schema-specific converters
# ---------------------------------------------------------------------------

def _rss_to_markdown(root) -> str:
    """Convert RSS 2.0 feed to Markdown."""
    channel = root.find("channel")
    if channel is None:
        return _generic_to_markdown(root)

    lines: list[str] = []

    title = _text(channel, "title")
    if title:
        lines.append(f"# {title}\n")

    desc = _text(channel, "description")
    if desc:
        lines.append(f"{_strip_tags(desc)}\n")

    link = _text(channel, "link")
    if link:
        lines.append(f"Source: {link}\n")

    for item in channel.findall("item"):
        lines.append("\n---\n")

        item_title = _text(item, "title")
        if item_title:
            lines.append(f"## {item_title}\n")

        pub_date = _text(item, "pubDate")
        item_link = _text(item, "link")
        if pub_date:
            lines.append(f"*Published: {pub_date}*  ")
        if item_link:
            lines.append(f"*Link: {item_link}*\n")

        # Prefer full content:encoded over description
        content_el = item.find(f"{{{_NS_CONTENT}}}encoded")
        body = (content_el.text if content_el is not None else None) or _text(item, "description")
        if body:
            lines.append(f"\n{_strip_tags(body)}\n")

        # dc:creator if present
        creator = item.find(f"{{{_NS_DC}}}creator")
        if creator is not None and creator.text:
            lines.append(f"\n*Author: {creator.text.strip()}*\n")

    return "\n".join(lines)


def _atom_to_markdown(root) -> str:
    """Convert Atom 1.0 feed to Markdown."""
    ns = f"{{{_NS_ATOM}}}"
    lines: list[str] = []

    title_el = root.find(f"{ns}title")
    if title_el is not None and title_el.text:
        lines.append(f"# {title_el.text.strip()}\n")

    subtitle_el = root.find(f"{ns}subtitle")
    if subtitle_el is not None and subtitle_el.text:
        lines.append(f"{subtitle_el.text.strip()}\n")

    for entry in root.findall(f"{ns}entry"):
        lines.append("\n---\n")

        entry_title = entry.find(f"{ns}title")
        if entry_title is not None and entry_title.text:
            lines.append(f"## {entry_title.text.strip()}\n")

        published = entry.find(f"{ns}published")
        if published is None:
            published = entry.find(f"{ns}updated")
        if published is not None and published.text:
            lines.append(f"*Published: {published.text.strip()}*  ")

        link_el = entry.find(f"{ns}link")
        if link_el is not None:
            href = link_el.get("href", "")
            if href:
                lines.append(f"*Link: {href}*\n")

        content_el = entry.find(f"{ns}content") or entry.find(f"{ns}summary")
        if content_el is not None and content_el.text:
            lines.append(f"\n{_strip_tags(content_el.text.strip())}\n")

    return "\n".join(lines)


def _sitemap_to_markdown(root) -> str:
    """Convert XML sitemap to a URL list in Markdown."""
    lines = ["# Sitemap\n"]
    for el in root.iter():
        if _local_name(el.tag) == "loc" and el.text:
            lines.append(f"- {el.text.strip()}")
    return "\n".join(lines)


def _generic_to_markdown(root) -> str:
    """
    Generic XML → Markdown.

    Top-level child elements become ## sections; their full text content
    (tags stripped, whitespace normalised) becomes the body.  Falls back
    to a flat text dump when the root has no meaningful children.
    """
    root_tag = _local_name(root.tag)
    lines: list[str] = [f"# {root_tag}\n"]

    # Root-level text (before first child)
    if root.text and root.text.strip():
        lines.append(root.text.strip() + "\n")

    for child in root:
        tag = _local_name(child.tag)
        body = _extract_text(child).strip()
        if body:
            lines.append(f"\n## {tag}\n\n{body}")

    if len(lines) == 1:
        # No structured children — dump everything
        full_text = _extract_text(root).strip()
        if full_text:
            lines.append(full_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_name(tag: str) -> str:
    """Strip namespace URI from ``{http://...}name`` → ``name``."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _namespace(tag: str) -> str:
    """Extract namespace URI from ``{ns}name`` → ``ns``."""
    if tag.startswith("{"):
        return tag[1:].split("}")[0]
    return ""


def _text(el, *paths: str, default: str = "") -> str:
    """Return stripped text of the first matching sub-element path."""
    for path in paths:
        found = el.find(path)
        if found is not None and found.text:
            return found.text.strip()
    return default


def _strip_tags(text: str) -> str:
    """Remove HTML/XML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text(el) -> str:
    """Recursively extract all text content from an element, space-joined."""
    parts: list[str] = []
    if el.text and el.text.strip():
        parts.append(el.text.strip())
    for child in el:
        child_text = _extract_text(child)
        if child_text:
            parts.append(child_text)
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return " ".join(parts)
