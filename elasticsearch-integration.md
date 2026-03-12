# Elasticsearch Integration Changes

**Baseline**: [NVIDIA-AI-Blueprints/rag](https://github.com/NVIDIA-AI-Blueprints/rag)

The baseline ships with rudimentary Elasticsearch support targeting a plain HTTP endpoint with no TLS, and still contains dead code paths that call back into Milvus utilities even in the ES code path. The sections below document all changes required to go from baseline to a fully working production deployment, organized by scope.

---

## Section 1 — Changes Required to Run the Baseline RAG Against ECK/ES

These changes are required simply to connect the unmodified NVIDIA RAG Blueprint to an ECK-managed Elasticsearch cluster. Without them the pods fail at startup with `SSL: CERTIFICATE_VERIFY_FAILED`, and metadata/document-info operations silently write nothing.

### 1.1 ECK / TLS Infrastructure

ECK 3.x automatically enables `xpack.security.*` and provisions a self-signed CA. The upstream blueprint assumes plain HTTP — no changes are made to the client for TLS.

**What ECK provides automatically:**
- HTTPS on port 9200 with a self-signed CA
- `elastic` superuser; password stored in secret `rag-eck-elasticsearch-es-elastic-user` (key: `elastic`)
- CA certificate in secret `rag-eck-elasticsearch-es-http-certs-public` (key: `ca.crt`)
- **Do NOT** set `xpack.security.*` in the ECK nodeSet config — ECK 3.x manages it internally; setting it disables security entirely
- Only safe nodeSet config key: `node.store.allow_mmap: false`

**Pre-requisite: create the CA ConfigMap** (not managed by Helm — must be created manually before first deploy):

```bash
kubectl create configmap eck-es-ca-cert -n rag \
  --from-literal=ca.crt="$(kubectl get secret rag-eck-elasticsearch-es-http-certs-public \
    -n rag -o jsonpath='{.data.ca\.crt}' | base64 -d)"
```

After any Helm upgrade touching the ECK CR, delete the ES pod to pick up the refreshed config:
```bash
kubectl delete pod -n rag -l common.k8s.elastic.co/type=elasticsearch
```

**Pod volume mounts** (both `rag-server` and `ingestor-server`):
```yaml
volumes:
  - name: eck-ca
    configMap:
      name: eck-es-ca-cert
volumeMounts:
  - name: eck-ca
    mountPath: /etc/ssl/eck
```

**Environment variables** (both pods):
```yaml
APP_VECTORSTORE_URL:          "https://rag-eck-elasticsearch-es-http:9200"
APP_VECTORSTORE_NAME:         "elasticsearch"
APP_VECTORSTORE_SSL_ENABLED:  "true"
APP_VECTORSTORE_USERNAME:     "elastic"
APP_VECTORSTORE_PASSWORD:     "<from secret rag-eck-elasticsearch-es-elastic-user>"
SSL_CERT_FILE:                "/etc/ssl/eck/ca.crt"
REQUESTS_CA_BUNDLE:           "/etc/ssl/eck/ca.crt"
```

`APP_VECTORSTORE_CA_CERTS` is intentionally left unset — `elastic_transport` reads `SSL_CERT_FILE` automatically, which is more reliable than passing a path through config.

**Retrieve the elastic password:**
```bash
kubectl get secret rag-eck-elasticsearch-es-elastic-user \
  -n rag -o jsonpath='{.data.elastic}' | base64 -d
```

### 1.2 `configuration.py` — New SSL Fields

**File**: `src/nvidia_rag/utils/configuration.py`

Three fields added to `VectorStoreConfig` that do not exist in the baseline:

```python
ssl_enabled: bool = Field(
    default=False,
    env="APP_VECTORSTORE_SSL_ENABLED",
)
ca_certs: str | None = Field(
    default=None,
    env="APP_VECTORSTORE_CA_CERTS",
    description="Path to CA bundle PEM. Leave unset to use SSL_CERT_FILE env var.",
)
verify_certs: bool = Field(
    default=True,
    env="APP_VECTORSTORE_VERIFY_CERTS",
)
```

All other `VectorStoreConfig` fields (`url`, `username`, `password`, `api_key`, `search_type`, `ranker_type`, etc.) are unchanged from the baseline.

### 1.3 `elastic_vdb.py` — SSL Context Wiring

**File**: `src/nvidia_rag/utils/vdb/elasticsearch/elastic_vdb.py`

The baseline `__init__` builds `es_conn_params` with only `hosts`, `request_timeout`, and auth — no SSL. The overlay adds a static SSL context builder and wires it into both the low-level client and the LangChain retrieval store.

**New static method:**
```python
@staticmethod
def _build_ssl_context(ca_certs: str | None, verify_certs: bool) -> ssl.SSLContext:
    if not verify_certs:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if ca_certs:
        return ssl.create_default_context(cafile=ca_certs)
    return ssl.create_default_context()
```

**Wired into `Elasticsearch()` client:**
```python
if self.config.vector_store.ssl_enabled:
    es_conn_params["ssl_context"] = self._build_ssl_context(
        ca_certs=self.config.vector_store.ca_certs,
        verify_certs=self.config.vector_store.verify_certs,
    )
self._es_connection = Elasticsearch(**es_conn_params)
```

**Wired into `ElasticsearchStore` (LangChain retrieval):**
```python
if self.config.vector_store.ssl_enabled:
    es_params["ssl_context"] = self._build_ssl_context(
        ca_certs=self.config.vector_store.ca_certs,
        verify_certs=self.config.vector_store.verify_certs,
    )
```

### 1.4 Milvus Removal — Metadata and Document-Info Operations

The upstream `elastic_vdb.py` carries code paths that delegate to Milvus utilities even when the ES backend is selected. These silently no-op in the ES path, meaning metadata schema and document-info writes never reach the index.

**Root cause — silent no-op in `write_to_index()`:**
```python
# UPSTREAM import — designed for Milvus, silently fails in ES path
from nv_ingest_client.util.milvus import cleanup_records, pandas_file_reader
```

The `add_metadata()` call that follows this import writes nothing to ES. The overlay documents this explicitly and bypasses it entirely (see Section 3 for the `content_url` injection that replaces it).

**Metadata schema and document-info collections** were rewritten from Milvus collection operations to native ES index operations. The upstream abstract base delegated these to a Milvus client — the overlay replaces every call with direct ES API calls.

**Metadata schema index** (`DEFAULT_METADATA_SCHEMA_COLLECTION`):
```python
# create
self._es_connection.indices.create(
    index=DEFAULT_METADATA_SCHEMA_COLLECTION,
    body=create_metadata_collection_mapping(),
)

# write (upsert = delete then index)
self._es_connection.delete_by_query(
    index=DEFAULT_METADATA_SCHEMA_COLLECTION,
    body=get_delete_metadata_schema_query(collection_name),
)
self._es_connection.index(
    index=DEFAULT_METADATA_SCHEMA_COLLECTION,
    body={"collection_name": collection_name, "metadata_schema": metadata_schema},
)

# read
response = self._es_connection.search(
    index=DEFAULT_METADATA_SCHEMA_COLLECTION,
    body=get_metadata_schema_query(collection_name),
)
```

**Document info index** (`DEFAULT_DOCUMENT_INFO_COLLECTION`) — same pattern:
```python
self._es_connection.delete_by_query(
    index=DEFAULT_DOCUMENT_INFO_COLLECTION,
    body=get_delete_document_info_query(collection_name, document_name, info_type),
)
self._es_connection.index(
    index=DEFAULT_DOCUMENT_INFO_COLLECTION,
    body={"collection_name": collection_name, "info_type": info_type,
          "document_name": document_name, "info_value": info_value},
)
```

### 1.5 Index Mappings (`es_queries.py`)

**File**: `src/nvidia_rag/utils/vdb/elasticsearch/es_queries.py`

The baseline creates system indexes with no explicit mapping, relying on ES dynamic mapping. Dynamic mapping analyzes string fields as `text`, which causes exact-match `term` queries to fail. The overlay defines explicit mappings with `keyword` type on all query fields.

**Metadata schema index mapping:**
```python
{
    "mappings": {
        "properties": {
            "collection_name": {"type": "keyword"},
            "metadata_schema":  {"type": "object", "enabled": True},
        }
    }
}
```

**Document info index mapping:**
```python
{
    "mappings": {
        "properties": {
            "collection_name": {"type": "keyword"},
            "info_type":       {"type": "keyword"},
            "document_name":   {"type": "keyword"},
            "info_value":      {"type": "object", "enabled": True},
        }
    }
}
```

**Summary of Milvus-to-ES method rewrites:**

| Method | Upstream | Overlay |
|---|---|---|
| `create_metadata_schema_collection()` | Milvus-backed abstract base | ES `indices.create()` with explicit keyword mapping |
| `add_metadata_schema()` | Milvus collection upsert | ES `delete_by_query` + `index()` |
| `get_metadata_schema()` | Milvus query | ES `search()` with `term` on `collection_name.keyword` |
| `create_document_info_collection()` | Milvus-backed | ES `indices.create()` with explicit keyword mapping |
| `add_document_info()` | Milvus collection upsert | ES `delete_by_query` + `index()` |
| `get_document_info()` | Milvus query | ES `search()` with compound `term` query |

### 1.6 Files Changed — Section 1

| File | Change |
|---|---|
| `src/nvidia_rag/utils/configuration.py` | Added `ssl_enabled`, `ca_certs`, `verify_certs` to `VectorStoreConfig` |
| `src/nvidia_rag/utils/vdb/elasticsearch/elastic_vdb.py` | `_build_ssl_context()`, SSL wiring into `Elasticsearch()` and `ElasticsearchStore`; rewrote metadata schema and document-info methods from Milvus to ES |
| `src/nvidia_rag/utils/vdb/elasticsearch/es_queries.py` | Explicit keyword mappings for metadata schema and document-info indexes |
| `deploy/helm/values-local.yaml` | ECK CR, CA ConfigMap volume mounts, SSL env vars for both pods |

---

## Section 2 — Direct Ingest to ES Without nv-ingest (Custom Modification)

The baseline always routes ingested documents through the nv-ingest / Ray pipeline regardless of document type. This is slow (~26s per batch) and unnecessary for pre-chunked text content such as HTML pages processed by the web crawler or PDFs processed by nemoretriever-parse. This section describes the bypass pipeline that embeds and writes chunks directly to ES.

### 2.1 Overview

The direct ingest path skips nv-ingest entirely:

```
Pre-chunked .md files
    ↓
ingest_chunk_files()         # read file contents into memory
    ↓
embed_chunks()               # POST /embeddings in batches of 32 (passage input_type)
    ↓
bulk_write_to_es()           # POST /<index>/_bulk in batches of 200
```

Performance: ~1s vs ~26s for 145 chunks — a 26× speedup.

### 2.2 `direct_ingest.py` (New File)

**File**: `src/nvidia_rag/utils/direct_ingest.py`

This file does not exist in the baseline. It provides two entry points:

- `ingest_text(text, source_uri, ...)` — for single HTML pages converted to markdown
- `ingest_chunk_files(chunk_files, source_uri_map, ...)` — for batches of pre-chunked `.md` files produced by nemoretriever-parse or the web crawler

Key implementation details:
- Embedding uses `input_type="passage"` (required by NVIDIA NIM — omitting this returns HTTP 400)
- `EMBED_BATCH_SIZE = 32` — max tokens per embedding request
- `ES_BULK_BATCH_SIZE = 200` — chunks per ES `_bulk` call
- `embed_semaphore = asyncio.Semaphore(8)` — caps concurrent embedding requests

**ES document schema written by direct ingest:**
```python
{
    "text": "<chunk text>",
    "vector": [0.1, ...],
    "metadata": {
        "source": {
            "source_id":   "<sha256(source_uri:chunk_index)[:16]>",
            "source_name": "<source_uri>",
            "source_type": "web",
            "date_created": "<ISO-8601>",
        },
        "content_metadata": {
            "type":          "text",
            "content_url":   "<source_uri>",   # keyword — used for delta upsert
            "chunk_index":   0,
            "document_type": "text",
            "filename":      "<basename>",
            "section_path":  "...",
            "total_chunks":  12,
        },
    },
}
```

### 2.3 `document_classifier_router.py` (New File)

**File**: `src/nvidia_rag/ingestor_server/document_classifier_router.py`

Not present in the baseline. Implements a two-pass nemoretriever-parse classifier for PDFs:

- **Pass 1**: Rasterise each page (DPI=300 via `pdf2image`) → POST to vLLM endpoint → parse `<x_X1><y_Y1>TEXT<x_X2><y_Y2><class_CLASSNAME>` tokens
- **Pass 2**: If any `table` or `picture` class found (or `force=True`), route to full VLM pipeline; otherwise return `None` → falls through to nv-ingest
- `skip_special_tokens=False` is critical — without it the coordinate tokens are stripped and layout is lost
- Pages processed in parallel via `ThreadPoolExecutor`; cross-page elements stitched with `_stitch_page_boundaries()`
- Output: list of `(tmp_md_path, chunk_meta)` tuples — `.md` files that feed directly into `ingest_chunk_files()`

Activated by: `APP_NEMOPARSE_ENABLED=true` (server-side default on) + `APP_NEMOPARSE_SERVERURL=http://nemotron-parse-v12:8000/v1/chat/completions`

### 2.4 `main.py` — Direct Ingest Routing

**File**: `src/nvidia_rag/ingestor_server/main.py`

The baseline `upload_documents()` always dispatches to nv-ingest. The overlay adds routing logic that detects pre-chunked content and bypasses nv-ingest.

**Two trigger cases inside `upload_documents()`:**

```python
# Case A: nemoretriever-parse ran on PDFs — chunk_to_original populated
_use_direct_ingest = bool(chunk_to_original)

# Case B: web crawler path — all files are .md/.txt and chunk_to_original is empty
_all_md_pre_chunked = (
    all(fp.endswith((".md", ".txt")) for fp in filepaths)
    and not chunk_to_original
)
_use_direct_ingest = _use_direct_ingest or (
    _all_md_pre_chunked and config.nv_ingest.enable_direct_ingest
)
```

**Routing branch:**
```python
if _use_direct_ingest:
    await __run_background_ingest_task(direct_ingest=True, ...)
else:
    await __run_nvingest_batched_ingestion(...)
```

**Direct ingest response path** (`__build_ingestion_response` with `direct_ingest=True`) skips:
- Unsupported file-type check (`.md` is not in nv-ingest `SUPPORTED_FILE_TYPES`)
- VDB `document_info` existence check (direct path never populates `document_info` via nv-ingest)
- Calls `add_document_info()` explicitly for each file in the final batch

**Controlled by env var**: `APP_NVINGEST_ENABLE_DIRECT_INGEST=true`

### 2.5 Ingest Mode — GPU Layout for Bulk Ingest

When running large ingest jobs the GPU layout must shift to free GPU0 slots for nemoretriever-parse:

```
Ingest mode:    nim-llm=0, gpu0-placeholder=0, nemotron-parse=7, nim-vlm=1
Inference mode: nim-llm=1, gpu0-placeholder=3, nemotron-parse=1, nim-vlm=1
```

`k8s_scaler.py` (`src/nvidia_rag/utils/k8s_scaler.py`) handles automatic switching: `enable_crawl_mode()` is called at ingest start, `disable_crawl_mode()` in the task `finally` block.

**Restore ordering is critical** — always: nemotron→0 (wait down) → nim-llm→1 (wait Running) → placeholder→3 → nemotron→1. Out-of-order scaling can assign nim-llm to GPU0 instead of GPU1, causing OOM.

### 2.6 Files Changed — Section 2

| File | Change |
|---|---|
| `src/nvidia_rag/utils/direct_ingest.py` | New — async embed + ES bulk write path |
| `src/nvidia_rag/ingestor_server/document_classifier_router.py` | New — nemoretriever-parse two-pass PDF classifier |
| `src/nvidia_rag/ingestor_server/main.py` | Added `_all_md_pre_chunked` / `_use_direct_ingest` detection, direct ingest branch, `APP_NEMOPARSE_ENABLED` wiring |
| `src/nvidia_rag/utils/k8s_scaler.py` | New — GPU mode switching for ingest vs. inference |
| `src/nvidia_rag/utils/configuration.py` | Added `NemoParseConfig`, `nv_ingest.enable_direct_ingest`, `pdf_repo_dir`, `docs_repo_dir` |
| `Dockerfile.ingestor-patch` | Added `poppler-utils`, `pdf2image` for PDF rasterisation |

---

## Section 3 — Web Crawl Functionality (Custom Modification)

The baseline has no web crawling capability. This section describes the full BFS web crawler and its ES integration for delta upsert, redirect tracking, and binary file management.

### 3.1 Overview

The web crawler (`src/nvidia_rag/utils/web_crawler.py`) is a new file with no upstream counterpart. It implements:

- BFS crawl over HTML pages with configurable depth, page limits, and URL prefix filters
- Selenium fallback for JS-rendered pages (`_is_js_sparse()` threshold: 300 visible chars)
- Semantic HTML chunking (headings → flush, paragraphs/lists → accumulate, tables/pictures → atomic)
- Three-phase pipeline: (1) HTML BFS + ingest, (2) NFS binary file recording, (3) binary file ingest
- Delta re-crawl via content hash comparison — unchanged pages skipped
- Atomic upsert: clears `last_ingested` before dispatch, restores only on success
- Redirect tracking (301/302): old URL chunks purged from ES, registry entry updated with `redirect_to`
- Cancellation via `threading.Event` (`POST /cancel`)
- `skip_phase3` flag to defer binary file ingestion to a later run

### 3.2 `content_url` — The ES Field That Enables Delta Upsert

The `content_url` keyword field (written by `write_to_index()` in `elastic_vdb.py`) is the key that makes web-crawl upsert possible. Without it there is no way to find and delete all chunks belonging to a specific URL — document filenames are random temp paths and do not map back to source URLs.

**Write path** (every chunk written by the web crawler or nemoretriever-parse):
```
source_uri passed in custom_metadata
    ↓
meta_dataframe built in upload_documents()
    ↓
write_to_index() in elastic_vdb.py injects:
    metadata.content_metadata.content_url = source_uri  (keyword field)
```

**Delete path** (before re-ingesting a changed URL):
```python
# elastic_vdb.py
def delete_by_content_url(self, collection_name: str, source_uris: list[str]) -> int:
    query = {
        "query": {
            "terms": {
                "metadata.content_metadata.content_url.keyword": source_uris,
            }
        }
    }
    response = self._es_connection.delete_by_query(index=collection_name, body=query)
    self._es_connection.indices.refresh(index=collection_name)
    return response.get("deleted", 0)
```

The `.keyword` sub-field is required — without it the `terms` query would run against an analyzed `text` field and fail to match exact URLs.

### 3.3 `main.py` — Upsert and Deletion Wiring

**File**: `src/nvidia_rag/ingestor_server/main.py`

**`source_uris_to_delete` parameter on `upload_documents()`** (new):

Before dispatching any ingest batch, stale chunks for changed URLs are purged:
```python
if source_uris_to_delete and hasattr(vdb_op, "delete_by_content_url"):
    vdb_op.delete_by_content_url(collection_name, source_uris_to_delete)
```

**`purge_deleted_urls()` method** (new):

Called by the crawler for URLs returning HTTP 404 or 410:
```python
async def purge_deleted_urls(
    self, collection_name: str, source_uris: list[str], vdb_auth_token: str = ""
) -> int:
    vdb_op, _ = self.__prepare_vdb_op_and_collection_name(...)
    if hasattr(vdb_op, "delete_by_content_url"):
        return vdb_op.delete_by_content_url(collection_name, source_uris)
    return 0
```

### 3.4 Delta Upsert Flow

```
URL content hash changed since last crawl:
    changed_urls.add(url)
    ↓
_dispatch_batch():
    registry[url].pop("last_ingested")      # atomic: clear before dispatch
    upload_documents(source_uris_to_delete=[url, ...])
    ↓
__run_background_ingest_task():
    vdb_op.delete_by_content_url(collection, [url, ...])   # purge stale chunks
    _run_direct_ingest() or __run_nvingest_batched_ingestion()
    ↓
_harvest_done():
    registry[url]["last_ingested"] = now_iso    # restored only after success
```

If the ingest task fails, `last_ingested` stays cleared — the URL will be re-attempted on the next crawl run.

### 3.5 Redirect Tracking

When a crawled URL returns a 301/302 redirect (detected via `resp.url != url` after `requests.Session.get()`):

1. Final URL is queued into BFS at the same depth (not a new hop)
2. If the old URL was previously ingested, its ES chunks are purged via `delete_by_content_url`
3. Registry entry for the old URL is updated with `redirect_to` field — **not deleted** — so future crawl runs recognize the known redirect without re-purging
4. `redirected_urls` set (separate from `deleted_urls`) ensures the registry entry is preserved

```python
registry[url] = {
    "redirect_to": final_url,
    "last_seen": now_iso,
    "status_code": 301,
}
```

### 3.6 Binary File Pipeline (Phase 3)

Linked binary files (PDF, DOCX, XLSX, PPTX) discovered during BFS are recorded in a manifest CSV during crawl rather than downloaded inline. This prevents mixed `.md` + binary batches that would cause `_all_md_pre_chunked = False` and route everything through nv-ingest.

After HTML BFS drains, Phase 3:
1. Reads the binary manifest
2. Downloads any deferred `inline` entries (files not on NFS) to temp files
3. Ingests NFS-persisted documents (PDFs via nemoretriever-parse, others without)
4. Cleans up temp files

**`skip_phase3` flag** (added in this deployment): set `skip_phase3=True` in the crawl request to complete HTML ingestion and defer binary ingest to a later run. The manifest is still fully populated during BFS.

### 3.7 Cancellation

`POST /cancel?task_id=<id>` sets a `threading.Event` registered in the module-level `_CRAWL_CANCEL` dict. The BFS loop checks the event at the top of each iteration; Phase 3 checks it before starting. Both paths save the URL registry and export artifacts before exiting.

### 3.8 `server.py` — New Endpoints

**File**: `src/nvidia_rag/ingestor_server/server.py`

| Endpoint | Description |
|---|---|
| `POST /crawl` | Start a BFS web crawl; returns `{task_id}` |
| `POST /cancel?task_id=` | Gracefully cancel an active crawl or ingest task |
| `GET /crawl-mode/status` | Current GPU mode (inference vs. ingest) |
| `POST /crawl-mode/enable` | Switch to ingest GPU layout |
| `POST /crawl-mode/disable` | Restore inference GPU layout |

### 3.9 Crawl API Parameters

```json
{
  "start_url": "https://docs.nvidia.com/",
  "collection_name": "nvidia_docs",
  "max_pages": null,
  "max_depth": null,
  "batch_ingest_size": 6,
  "use_nemoretriever_parse": true,
  "force_nemoretriever_parse": true,
  "extract_linked_files": true,
  "skip_phase3": false,
  "use_selenium": true,
  "allowed_url_prefixes": null,
  "blocked_url_patterns": null,
  "use_sitemap": false
}
```

### 3.10 Known Limitation: Backfill for Pre-existing Chunks

Chunks ingested before the `content_url` injection was deployed have no `content_url` field. The first re-ingest of a changed URL produces duplicates; they clean up correctly on the second change.

Retroactive backfill:
```bash
python3 /home/joncoons/claude/rag/scripts/backfill_content_url.py
```

This reads `chunk_doc_names` from the URL registry (populated after the first crawl with new code) and sets `content_url` on matching ES docs via scroll + bulk update.

### 3.11 Files Changed — Section 3

| File | Change |
|---|---|
| `src/nvidia_rag/utils/web_crawler.py` | New — full BFS crawler with delta upsert, redirect tracking, binary manifest, Phase 3, cancellation |
| `src/nvidia_rag/utils/vdb/elasticsearch/elastic_vdb.py` | Added `content_url` injection in `write_to_index()`; added `delete_by_content_url()` |
| `src/nvidia_rag/ingestor_server/main.py` | Added `source_uris_to_delete` param on `upload_documents()`; added `purge_deleted_urls()` |
| `src/nvidia_rag/ingestor_server/server.py` | Added `POST /crawl`, `POST /cancel`, crawl-mode endpoints; `CrawlRequest` model |
| `src/nvidia_rag/utils/configuration.py` | Added `crawler_export_dir`, `audio_repo_dir`, `video_repo_dir`, `max_media_file_mb` |
| `scripts/backfill_content_url.py` | New — retroactive `content_url` backfill for pre-existing chunks |
| `frontend/src/components/drawer/WebCrawlSection.tsx` | New — crawl configuration UI in collection drawer |
| `frontend/src/hooks/useCollectionActions.ts` | Added `handleStartCrawl()` |
