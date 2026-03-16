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
Storage primitive: embed text chunks and bulk-write to Elasticsearch.

This module is the shared transport layer used by all ingest pipelines
(nemotron-parse, web crawler, direct upload).  It has no domain knowledge
of how chunks were produced — it only embeds and stores them.

ES document schema (mirrors ElasticVDB.write_to_index output):

    {
        "text": <str>,
        "vector": <list[float]>,
        "metadata": {
            "source": {
                "source_id": <str>,
                "source_name": <str>,
                "source_type": "web",
                "date_created": <ISO-8601>,
            },
            "content_metadata": {
                "type": "text",
                "content_url": <source URI>,
                "chunk_index": <int>,
                "document_type": "text",
                "filename": <str>,
                ...extra_meta fields...
            },
        },
    }

Public API
----------
chunk_text(text, chunk_size, chunk_overlap) -> list[str]
embed_chunks(chunks, session, semaphore, config) -> list[vector | None]
bulk_write_to_es(docs, index, session, config, ssl_ctx) -> int
ingest_text(text, source_uri, collection_name, config, embed_semaphore) -> int
ingest_chunk_files(chunk_files, source_uri_map, collection_name, config, ...) -> int
"""

import asyncio
import hashlib
import json
import logging
import os
import ssl
from datetime import UTC, datetime

import aiohttp

from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)

# ── tunables ────────────────────────────────────────────────────────────────
# Chunks per POST /embeddings request.  Embedding service batches well up to
# ~32; larger values risk hitting the NIM max-batch-size limit.
EMBED_BATCH_SIZE = 32

# Documents per ES _bulk request.  200 is safe for 2 KB average text.
ES_BULK_BATCH_SIZE = 50

# Rough chars-per-token ratio used when converting token-based config values
# (chunk_size, chunk_overlap) to character counts for client-side chunking.
CHARS_PER_TOKEN = 4


# ── SSL context ─────────────────────────────────────────────────────────────

def _build_ssl_ctx(config: NvidiaRAGConfig) -> ssl.SSLContext | None:
    """Return an SSLContext for ES connections, or None if SSL is disabled."""
    if not config.vector_store.ssl_enabled:
        return None

    ca = config.vector_store.ca_certs or os.environ.get("SSL_CERT_FILE")
    if ca:
        ctx = ssl.create_default_context(cafile=ca)
    else:
        ctx = ssl.create_default_context()

    if not config.vector_store.verify_certs:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    return ctx


# ── client-side text chunking ────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 8192,   # chars ≈ 2048 tokens
    chunk_overlap: int = 600,  # chars ≈ 150 tokens
) -> list[str]:
    """
    Split *text* into overlapping chunks at paragraph boundaries.

    Paragraphs are concatenated greedily until *chunk_size* is reached, then
    the buffer is flushed.  The tail of each chunk is kept as overlap context
    for the next chunk (up to *chunk_overlap* chars).
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len: int = 0

    for para in paragraphs:
        plen = len(para) + 2  # +2 for "\n\n" separator
        if buf and buf_len + plen > chunk_size:
            chunks.append("\n\n".join(buf))
            # seed next buffer with overlap tail
            overlap_buf: list[str] = []
            overlap_len = 0
            for p in reversed(buf):
                pl = len(p) + 2
                if overlap_len + pl > chunk_overlap:
                    break
                overlap_buf.insert(0, p)
                overlap_len += pl
            buf = overlap_buf
            buf_len = overlap_len
        buf.append(para)
        buf_len += plen

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


# ── embedding ────────────────────────────────────────────────────────────────

async def embed_chunks(
    chunks: list[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    config: NvidiaRAGConfig,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float] | None]:
    """
    Embed *chunks* in batches via the configured embedding service.

    Returns a list of vectors in the same order as *chunks*.  On per-batch
    failure the corresponding slots are set to ``None`` and a warning is
    logged so the caller can skip those documents.
    """
    server_url = config.embeddings.server_url.rstrip("/")
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    embed_url = f"{server_url}/embeddings"
    model = config.embeddings.model_name

    results: list[list[float] | None] = [None] * len(chunks)

    async def _embed_one_batch(batch_start: int, batch: list[str]) -> None:
        async with semaphore:
            # input_type="passage" required for NVIDIA asymmetric embedding NIMs
            # (query embedding uses input_type="query" at retrieval time).
            payload = {"input": batch, "model": model, "input_type": "passage"}
            try:
                async with session.post(
                    embed_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                for item in data.get("data", []):
                    results[batch_start + item["index"]] = item["embedding"]
            except Exception as exc:
                logger.warning(
                    "embed_chunks: batch starting at %d failed: %r", batch_start, exc
                )

    tasks = [
        _embed_one_batch(i, chunks[i : i + batch_size])
        for i in range(0, len(chunks), batch_size)
    ]
    await asyncio.gather(*tasks)
    return results


# ── ES bulk write ─────────────────────────────────────────────────────────────

async def bulk_write_to_es(
    docs: list[dict],
    index: str,
    session: aiohttp.ClientSession,
    config: NvidiaRAGConfig,
    ssl_ctx: ssl.SSLContext | None = None,
    batch_size: int = ES_BULK_BATCH_SIZE,
) -> int:
    """
    Bulk-index *docs* into ES *index*.  Returns successfully indexed count.
    """
    es_url = config.vector_store.url.rstrip("/")
    bulk_url = f"{es_url}/{index}/_bulk"

    username = config.vector_store.username or ""
    password = (
        config.vector_store.password.get_secret_value()
        if config.vector_store.password
        else ""
    )
    auth = aiohttp.BasicAuth(username, password) if username else None
    ssl_param: ssl.SSLContext | bool = ssl_ctx if ssl_ctx is not None else False

    total_indexed = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        lines: list[str] = []
        for doc in batch:
            lines.append(json.dumps({"index": {"_index": index}}))
            lines.append(json.dumps(doc))
        body = "\n".join(lines) + "\n"

        try:
            async with session.post(
                bulk_url,
                data=body,
                headers={"Content-Type": "application/x-ndjson"},
                auth=auth,
                ssl=ssl_param,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

            if result.get("errors"):
                failed = sum(
                    1
                    for item in result.get("items", [])
                    if "error" in item.get("index", {})
                )
                logger.warning(
                    "bulk_write_to_es: %d/%d items failed in batch at offset %d",
                    failed, len(batch), i,
                )
                total_indexed += len(batch) - failed
            else:
                total_indexed += len(batch)
        except Exception as exc:
            logger.warning(
                "bulk_write_to_es: batch at offset %d failed: %r", i, exc
            )

    return total_indexed


# ── document builder ─────────────────────────────────────────────────────────

def _build_doc(
    text: str,
    vector: list[float],
    source_uri: str,
    chunk_index: int,
    filename: str,
    extra_meta: dict | None = None,
) -> dict:
    """
    Build an ES document matching the nvidia_rag VectorStore schema.

    ``source_uri`` is stored as ``content_url`` (keyword) so
    ``delete_by_content_url`` can remove stale chunks on re-crawl.
    """
    now = datetime.now(UTC).isoformat()
    doc_id = hashlib.sha256(
        f"{source_uri}:{chunk_index}".encode()
    ).hexdigest()[:16]

    content_meta: dict = {
        "type": "text",
        "content_url": source_uri,
        "chunk_index": chunk_index,
        "document_type": "text",
        "filename": filename,
    }
    if extra_meta:
        content_meta.update(extra_meta)

    return {
        "text": text,
        "vector": vector,
        "metadata": {
            "source": {
                "source_id": doc_id,
                "source_name": source_uri,
                "source_type": "web",
                "date_created": now,
            },
            "content_metadata": content_meta,
        },
    }


# ── high-level API ────────────────────────────────────────────────────────────

async def ingest_text(
    text: str,
    source_uri: str,
    collection_name: str,
    config: NvidiaRAGConfig,
    embed_semaphore: asyncio.Semaphore,
    filename: str | None = None,
) -> int:
    """
    Chunk, embed, and write plain text directly to ES.

    Suitable for HTML page text extracted by the web crawler when
    nemoretriever-parse is not in the loop.  Returns chunk count written.
    """
    chunk_size = config.nv_ingest.chunk_size * CHARS_PER_TOKEN
    chunk_overlap = config.nv_ingest.chunk_overlap * CHARS_PER_TOKEN
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.debug("ingest_text: no chunks from %s — skipping", source_uri)
        return 0

    ssl_ctx = _build_ssl_ctx(config)
    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else aiohttp.TCPConnector()

    async with aiohttp.ClientSession(connector=connector) as session:
        vectors = await embed_chunks(chunks, session, embed_semaphore, config)
        docs = [
            _build_doc(
                text=chunk,
                vector=vec,
                source_uri=source_uri,
                chunk_index=i,
                filename=filename or source_uri,
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            if vec is not None
        ]
        written = await bulk_write_to_es(docs, collection_name, session, config, ssl_ctx)

    logger.info(
        "embed_store.ingest_text: %d/%d chunks written — %s",
        written, len(chunks), source_uri,
    )
    return written


async def ingest_chunk_files(
    chunk_files: list[str],
    source_uri_map: dict[str, str],
    collection_name: str,
    config: NvidiaRAGConfig,
    embed_semaphore: asyncio.Semaphore,
    extra_meta_map: dict[str, dict] | None = None,
    pipeline_batch_size: int = ES_BULK_BATCH_SIZE,
) -> int:
    """
    Embed nemoretriever-parse pre-chunked text files and write them to ES.

    Processes files in pipelined stages of ``pipeline_batch_size`` chunks: each
    stage embeds its texts (in EMBED_BATCH_SIZE sub-batches via the semaphore)
    then immediately bulk-writes the resulting vectors to ES before moving on.
    This bounds peak memory to one stage at a time and lets ES writes begin as
    soon as the first stage of embeddings is ready.

    Args:
        chunk_files:         Paths to the chunk .md/.txt files to ingest.
        source_uri_map:      Maps ``os.path.basename(fp)`` → original source URL.
                             Used to set ``content_url`` for upsert support.
        collection_name:     ES index / collection to write into.
        config:              NvidiaRAGConfig with embeddings + vector_store settings.
        embed_semaphore:     Semaphore capping concurrent embedding API calls.
        extra_meta_map:      Optional per-file extra metadata keyed by basename.
        pipeline_batch_size: Chunks per embed→write stage (default: ES_BULK_BATCH_SIZE).

    Returns:
        Number of ES documents successfully written.
    """
    texts: list[str] = []
    metas: list[dict] = []

    for fp in chunk_files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
        except OSError as exc:
            logger.warning("ingest_chunk_files: cannot read %s: %r", fp, exc)
            continue

        if not content:
            continue

        basename = os.path.basename(fp)
        texts.append(content)
        metas.append({
            "basename": basename,
            "source_uri": source_uri_map.get(basename, fp),
            "extra": (extra_meta_map or {}).get(basename) or {},
        })

    if not texts:
        logger.warning(
            "ingest_chunk_files: no readable content in %d file(s)", len(chunk_files)
        )
        return 0

    ssl_ctx = _build_ssl_ctx(config)
    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else aiohttp.TCPConnector()
    total_written = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        for stage_start in range(0, len(texts), pipeline_batch_size):
            stage_texts = texts[stage_start : stage_start + pipeline_batch_size]
            stage_metas = metas[stage_start : stage_start + pipeline_batch_size]

            vectors = await embed_chunks(stage_texts, session, embed_semaphore, config)

            docs = [
                _build_doc(
                    text=text,
                    vector=vec,
                    # chunk_index uses global offset so doc_id is unique across stages
                    source_uri=meta["source_uri"],
                    chunk_index=stage_start + i,
                    filename=meta["basename"],
                    extra_meta=meta["extra"] or None,
                )
                for i, (text, vec, meta) in enumerate(zip(stage_texts, vectors, stage_metas))
                if vec is not None
            ]

            stage_written = await bulk_write_to_es(
                docs, collection_name, session, config, ssl_ctx
            )
            total_written += stage_written
            logger.debug(
                "ingest_chunk_files: stage [%d:%d] — %d/%d written",
                stage_start, stage_start + len(stage_texts),
                stage_written, len(stage_texts),
            )

    logger.info(
        "embed_store.ingest_chunk_files: %d/%d chunks written to collection=%s",
        total_written, len(texts), collection_name,
    )
    return total_written
