# Operations Runbook — NVIDIA RAG Blueprint with Nemotron Parse Classification
## Local k3s Deployment

This runbook covers the complete launch and shutdown sequence for the NVIDIA RAG Blueprint on the
local k3s cluster, including the nemoretriever-parse document classification router introduced to
route complex data elements (tables, charts, graphs, infographics) through VLM-quality markdown
extraction during ingestion.

---

## Cluster Overview

| Node | GPUs | Virtual Slots | Pinned Workloads |
|---|---|---|---|
| `ubuntu-local-dev` | 2x RTX Pro 6000 Blackwell 96 GB | 8 | LLM NIM, Embedding NIM, Reranking NIM, all nv-ingest NIMs, nemoretriever-parse |
| `ubuntu2` | 1x RTX 5090 32 GB | 4 | Available for overflow |

**VRAM budget on ubuntu-local-dev (worst case, all active simultaneously):**

| Service | Approx VRAM |
|---|---|
| LLM NIM (Nemotron 49B NVFP4) | ~25 GB |
| Embedding NIM (1B) | ~4 GB |
| Reranking NIM (1B) | ~4 GB |
| nv-ingest OCR | ~3 GB |
| nv-ingest Graphic Elements | ~3 GB |
| nv-ingest Page Elements | ~3 GB |
| nv-ingest Table Structure | ~3 GB |
| nemoretriever-parse (VLM) | ~20 GB |
| **Total** | **~65 GB / 96 GB** |

The nemoretriever-parse NIM runs only during active ingestion requests with the toggle enabled.
All inference NIMs use time-sliced GPU slots (factor 4) so they share physical GPU time.

---

## Service Endpoints (post-launch)

| Service | Internal Cluster Address | External (NodePort) |
|---|---|---|
| RAG Server API | `rag-server:8081/v1` | `<node-ip>:8081` |
| Ingestor API | `ingestor-server:8082/v1` | `<node-ip>:8082` |
| Frontend UI | `rag-frontend:3000` | `<node-ip>:3000` |
| LLM NIM | `nim-llm:8000` | — |
| Embedding NIM | `nemoretriever-embedding-ms:8000` | — |
| Reranking NIM | `nemoretriever-ranking-ms:8000` | — |
| nemoretriever-parse | `nemoretriever-parse-ms:8000` | — |
| Elasticsearch | `rag-eck-elasticsearch-es-http:9200` | — |

---

## Prerequisites

These cluster-level resources must exist before running the Helm install. They are created once
and survive across upgrades. Verify they are present before every launch.

```bash
# Verify NGC secrets
kubectl get secret ngc-secret ngc-api -n rag

# Verify ECK CA cert ConfigMap
kubectl get configmap eck-es-ca-cert -n rag

# Verify LLM model cache PV and PVC
kubectl get pv nim-llm-cache-pv
kubectl get pvc nim-llm-cache-pvc -n rag
```

If the ECK CA ConfigMap is missing (e.g. after a cluster wipe), recreate it after Elasticsearch
has started:

```bash
kubectl create configmap eck-es-ca-cert -n rag \
  --from-literal=ca.crt="$(kubectl get secret rag-eck-elasticsearch-es-http-certs-public \
    -n rag -o jsonpath='{.data.ca\.crt}' | base64 -d)"
```

---

## Launch Sequence

### Step 1 — Retrieve the Elasticsearch password

ECK generates and rotates this password automatically. Always fetch it fresh before each deploy.

```bash
ES_PASS=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user \
  -n rag -o jsonpath='{.data.elastic}' | base64 -d)

echo "ES_PASS is set: ${#ES_PASS} chars"   # should print a non-zero length
```

### Step 2 — Helm upgrade/install

Run from the repository root.

```bash
cd /home/joncoons/claude/rag/deploy/helm

helm upgrade --install rag ./nvidia-blueprint-rag \
  -n rag --create-namespace \
  -f values-local.yaml \
  --set envVars.APP_VECTORSTORE_PASSWORD="${ES_PASS}" \
  --set "ingestor-server.envVars.APP_VECTORSTORE_PASSWORD=${ES_PASS}"
```

The `--install` flag makes the command idempotent — safe to run on a running cluster to apply
configuration changes.

**To additionally inject the nemoretriever-parse API key if the NIM is secured:**

```bash
helm upgrade --install rag ./nvidia-blueprint-rag \
  -n rag --create-namespace \
  -f values-local.yaml \
  --set envVars.APP_VECTORSTORE_PASSWORD="${ES_PASS}" \
  --set "ingestor-server.envVars.APP_VECTORSTORE_PASSWORD=${ES_PASS}" \
  --set "ingestor-server.envVars.APP_NEMOPARSE_APIKEY=${NGC_API_KEY}"
```

### Step 3 — Apply the NIMCache PVC patch

This patch must be re-applied after every Helm upgrade that touches the NIMCache resource.
The upstream chart has a `| default true` bug that prevents setting `create: false` via values.

```bash
kubectl patch nimcache nim-llm-cache -n rag --type=merge \
  -p '{"spec":{"storage":{"pvc":{"create":false,"name":"nim-llm-cache-pvc"}}}}'
```

### Step 4 — Wait for Elasticsearch to be ready

ECK takes 1-3 minutes to initialise the Elasticsearch cluster and generate the elastic user
secret. Other pods that depend on Elasticsearch (rag-server, ingestor-server) will crash-loop
until it is healthy.

```bash
# Watch the ECK cluster status
kubectl get elasticsearch -n rag -w

# Ready when: health = green, phase = Ready
# NAME                      HEALTH   NODES   VERSION   PHASE   AGE
# rag-eck-elasticsearch     green    1       8.x.x     Ready   3m
```

### Step 5 — Monitor pod startup

Initial deployment downloads NIMCache models for all enabled NIMs (except the pre-cached LLM).
Allow 60–70 minutes for all pods to reach Running state on a fresh cluster. On subsequent
upgrades where caches are warm, startup is 5–10 minutes.

```bash
# Watch all pods in the rag namespace
kubectl get pods -n rag -w

# Check NIM download progress
kubectl get nimcache -n rag
kubectl get nimservice -n rag

# View events sorted by time (useful for diagnosing stuck pods)
kubectl get events -n rag --sort-by='.lastTimestamp' | tail -30
```

**Expected final pod set (all Running):**

```
rag-server-*                      2/2   Running
ingestor-server-*                 1/1   Running
rag-frontend-*                    1/1   Running
nim-llm-*                         1/1   Running
nemoretriever-embedding-ms-*      1/1   Running
nemoretriever-ranking-ms-*        1/1   Running
nemoretriever-parse-ms-*          1/1   Running     ← new classification NIM
nv-ingest-*                       1/1   Running
rag-eck-elasticsearch-es-*        1/1   Running
rag-minio-*                       1/1   Running
rag-redis-master-*                1/1   Running
```

### Step 6 — Health checks

```bash
# RAG server health
curl -s http://<node-ip>:8081/health | jq .

# Ingestor server health (includes dependency checks)
curl -s http://<node-ip>:8082/health | jq .

# nemoretriever-parse NIM model readiness (internal cluster)
kubectl exec -n rag deployment/ingestor-server -- \
  curl -s http://nemoretriever-parse-ms:8000/v1/health/ready | jq .
```

All endpoints should return `{"message": "Service is up."}` or `{"status": "ready"}`.

---

## Using the Classification Function

### Via the Frontend UI

1. Open the UI at `http://<node-ip>:3000`
2. Open or create a collection
3. Click **Add New Documents** in the collection drawer
4. Select PDF files to upload
5. Toggle **"Nemotron Parse (complex data elements)"** to ON
6. Click **Upload**

When the toggle is on, the ingestor server will:
- Convert each PDF page to a rasterised image (200 DPI)
- Run `detection_only` on every page via nemoretriever-parse
- For any document where a table, chart, graph, or infographic is detected, run
  `markdown_no_bbox` on all pages of that document
- Split the resulting markdown with a header/table/code-fence-aware chunker into
  one or more pre-sized `.md` temp files (≈ 512-token sections); code fences and
  table blocks are treated as atomic units and never bisected
- Submit each chunk file individually to NV-Ingest so the token splitter receives
  semantic units rather than an arbitrary cross-section of the document
- Documents with no complex elements fall through to the standard NV-Ingest pipeline

### Via the API directly

```bash
# Upload with classification routing enabled
curl -X POST http://<node-ip>:8082/documents \
  -F "documents=@/path/to/report.pdf" \
  -F 'data={"collection_name":"my-collection","blocking":false,"use_nemoretriever_parse":true}'
```

Optional lineage fields (both default to safe values if omitted):

```bash
# With explicit batch ID and source system for lineage tracking
curl -X POST http://<node-ip>:8082/documents \
  -F "documents=@/path/to/report.pdf" \
  -F 'data={
    "collection_name":"my-collection",
    "blocking":false,
    "use_nemoretriever_parse":true,
    "upload_batch_id":"your-uuid-here",
    "source_system":"sharepoint"
  }'
```

Check the returned `task_id` against the status endpoint:

```bash
curl -s http://<node-ip>:8082/status?task_id=<task_id> | jq .state
```

### Enable classification server-side by default

To make nemoretriever-parse routing the default for all uploads without requiring the GUI toggle,
set the env var in `values-local.yaml` and redeploy:

```yaml
ingestor-server:
  envVars:
    APP_NEMOPARSE_ENABLED: "true"   # change false → true
```

Then rerun Steps 1–3 of the launch sequence.

### Verify the classification NIM is reachable from the ingestor

```bash
kubectl exec -n rag deployment/ingestor-server -- \
  curl -s -o /dev/null -w "%{http_code}" \
  http://nemoretriever-parse-ms:8000/v1/health/ready

# Expected: 200
```

### Tail ingestor logs during a classified upload

```bash
kubectl logs -n rag -l app=ingestor-server -f | grep -E "nemoretriever|classifier|complex|route"
```

Log lines to expect:
```
Starting classifier pass for: annual_report.pdf
Complex element(s) detected on page 3 of 'annual_report.pdf': {'table', 'chart'}
Routing 'annual_report.pdf' through nemoretriever-parse (12 pages total)
nemoretriever-parse output for 'annual_report.pdf' written to 4 chunk file(s)
nemoretriever-parse: 'annual_report.pdf' → 4 chunk file(s)
```

The number of chunk files depends on document length and section structure. A 12-page
report with mixed prose and tables typically produces 3–6 chunks. A document that fits
within one ≈ 2 048-character section produces a single chunk file (no suffix appended).

---

## Document Lineage Metadata

Every document ingested through the ingestor server — whether via the standard NV-Ingest
pipeline or via the nemoretriever-parse route — now carries the following lineage fields
automatically. No client-side action is required; the ingestor injects them at ingest time.

| Metadata Field | Source | Description |
|---|---|---|
| `content_hash` | Ingestor (auto) | SHA-256 hex digest of the original source file bytes. Identical hash = unchanged content; use for delta-ingest logic. |
| `ingested_at` | Ingestor (auto) | ISO-8601 UTC timestamp of the upload call. |
| `pipeline_type` | Ingestor (auto) | `"nv_ingest"` for the standard pipeline; `"nemoretriever_parse"` for VLM-routed documents. |
| `source_uri` | Ingestor (auto) | Original filename (best proxy for uploaded files). |
| `upload_batch_id` | API field (auto-UUID) | UUID shared by all documents in a single upload request. Auto-generated if not supplied. |
| `source_system` | API field (optional) | Caller-supplied origin label, e.g. `"sharepoint"`, `"s3"`, `"local"`. |
| `document_type` | Ingestor (auto) | Lowercase file extension of the original file, e.g. `"pdf"`, `"docx"`, `"md"`. Falls back to `"unknown"` if no extension. |

### Querying by lineage metadata

The `FilterExpressionGenerator` in the RAG server can translate natural-language queries
into Elasticsearch filter expressions against these fields.  Examples:

```bash
# Find all documents ingested after a specific date
curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quarterly earnings",
    "collection_names": ["finance-docs"],
    "filter": "ingested_at >= \"2025-01-01T00:00:00+00:00\""
  }' | jq .

# Find all documents from a specific upload batch
curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "supply chain risk",
    "collection_names": ["reports"],
    "filter": "upload_batch_id == \"<your-uuid>\""
  }' | jq .

# Find only nemoretriever_parse-processed documents
curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue table",
    "collection_names": ["reports"],
    "filter": "pipeline_type == \"nemoretriever_parse\""
  }' | jq .
```

### Delta ingest pattern

To avoid re-ingesting unchanged documents, compare `content_hash` before uploading:

```bash
# 1. Query the collection for the document's stored hash
STORED_HASH=$(curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"","collection_names":["my-collection"],"filter":"source_uri == \"report.pdf\"","vdb_top_k":1}' \
  | jq -r '.chunks[0].metadata.content_metadata.content_hash // empty')

# 2. Compute hash of the local file
LOCAL_HASH=$(sha256sum /path/to/report.pdf | awk '{print $1}')

# 3. Only upload if changed (or new)
if [ "$LOCAL_HASH" != "$STORED_HASH" ]; then
  curl -X DELETE "http://<node-ip>:8082/documents?collection_name=my-collection&filename=report.pdf"
  curl -X POST http://<node-ip>:8082/documents \
    -F "documents=@/path/to/report.pdf" \
    -F 'data={"collection_name":"my-collection","blocking":false}'
fi
```

---

## Web Crawl Classification Agent

The `scripts/web_crawl_classification_agent.py` script crawls a website domain and
produces ingestion-ready Markdown files with position-aware content merging and optional
VLM classification for complex elements.

### Run the crawl agent

```bash
python3 scripts/web_crawl_classification_agent.py \
  --start-url  https://docs.example.com/ \
  --output-dir ./crawl_output \
  --nemo-parse-url http://nemoretriever-parse-ms:8000/v1/chat/completions \
  --ingestor-url  http://<node-ip>:8082 \
  --collection    web-docs \
  --max-pages     200
```

Omit `--nemo-parse-url` to run in BS4-only mode (no GPU, no complex-element detection).
Omit `--ingestor-url` to write files to disk only without submitting to the pipeline.

### Output structure

```
crawl_output/
  pages/
    docs_example_com_guide.md          (single-chunk page)
    docs_example_com_api_reference_001.md   (multi-chunk page, part 1)
    docs_example_com_api_reference_002.md   (part 2)
  screenshots/
    docs_example_com_guide.png
  crawl_manifest.csv
```

**Manifest CSV columns:**

| Column | Description |
|---|---|
| `url` | Full URL of the crawled page |
| `slug` | Filesystem-safe slug derived from the URL |
| `method` | `bs4_only` or `bs4+vlm` |
| `detected_types` | Pipe-delimited list of detected complex element types |
| `md_path` | Path to the first (or only) chunk file |
| `chunk_count` | Total number of chunk files produced for this page |
| `screenshot_path` | Full-page screenshot path |
| `crawl_depth` | BFS hop distance from the seed URL |
| `crawl_session_id` | UUID shared across all pages in this crawl run |
| `domain` | Netloc of the seed URL |
| `status` | `ok` or `error` |
| `error` | Error message if status is `error` |

### Lineage metadata stored per chunk

When `--ingestor-url` is provided, the agent attaches the following metadata to every
submitted file:

| Field | Value |
|---|---|
| `source_uri` | Original page URL |
| `content_hash` | SHA-256 of the chunk file bytes |
| `ingested_at` | ISO-8601 UTC timestamp of submission |
| `pipeline_type` | `web_crawl_bs4_vlm` or `web_crawl_bs4` |
| `crawl_session_id` | UUID for this crawl run — filter to refresh a full crawl |
| `domain` | Netloc of the seed URL |
| `last_crawled_at` | Same as `ingested_at` for current implementations |
| `crawl_depth` | BFS hop distance from seed URL |
| `document_type` | Always `"md"` for web crawl chunks |
| `page_title` | HTML `<title>` of the crawled page (omitted if blank) |

### Query web crawl content

```bash
# All documents from a specific crawl session
curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication flow",
    "collection_names": ["web-docs"],
    "filter": "crawl_session_id == \"<session-uuid>\""
  }' | jq .

# Only pages with VLM-enhanced extraction (contained tables/charts)
curl -s -X POST http://<node-ip>:8081/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "performance benchmark",
    "collection_names": ["web-docs"],
    "filter": "pipeline_type == \"web_crawl_bs4_vlm\""
  }' | jq .
```

---

## Monitoring and Troubleshooting

### Pod log access

```bash
# Ingestor server (classification and ingestion logs)
kubectl logs -n rag -l app=ingestor-server --tail=100

# nemoretriever-parse NIM logs
kubectl logs -n rag -l app=nemoretriever-parse-ms --tail=100

# RAG server logs
kubectl logs -n rag -l app=rag-server -c nvidia-blueprint-rag --tail=100

# nv-ingest pipeline logs
kubectl logs -n rag -l app=nv-ingest --tail=100
```

### GPU utilisation

```bash
# On ubuntu-local-dev
ssh ubuntu-local-dev 'nvidia-smi'

# Watch continuously
ssh ubuntu-local-dev 'watch -n2 nvidia-smi'
```

### Classification not routing documents

| Symptom | Check | Action |
|---|---|---|
| Toggle has no effect | Ingestor logs: `APP_NEMOPARSE_SERVERURL is not configured` | Verify `APP_NEMOPARSE_SERVERURL` env var is set in ingestor pod |
| Classifier pass runs but nothing is routed | Logs show no complex elements detected | Document may genuinely contain no tables/charts — inspect detection_only output |
| nemoretriever-parse pod not Running | `kubectl get pods -n rag` | Check events and logs; confirm NIMCache download completed |
| 500 error from nemoretriever-parse | NIM logs | Check VRAM headroom with `nvidia-smi`; consider reducing NIM_KVCACHE_PERCENT on other NIMs |

### Elasticsearch connectivity

```bash
# Confirm ingestor can reach Elasticsearch
kubectl exec -n rag deployment/ingestor-server -- \
  curl -s -k -u elastic:${ES_PASS} \
  https://rag-eck-elasticsearch-es-http:9200/_cluster/health | jq .status

# Expected: "green" or "yellow"
```

### Flannel VXLAN recovery (ubuntu2 after node reconnect)

If ubuntu2 rejoins the cluster and inter-node overlay traffic is broken:

```bash
ping 10.42.1.0   # from ubuntu-local-dev host; triggers ARP resolution
# Wait 2-3 minutes; tunnel self-heals
```

---

## Shutdown Sequence

### Graceful shutdown (preserves PVs and cached models)

```bash
helm uninstall rag -n rag
```

This removes all pods, services, and ConfigMaps managed by the chart. It does **not** delete:
- PersistentVolumeClaims (model caches, ingestor data)
- Cluster-scoped resources created outside the chart (PVs, ECK operator, NIM Operator)
- The `eck-es-ca-cert` ConfigMap (created manually)
- NGC secrets

Elasticsearch data stored in the ECK-managed PVC is also preserved.

### Verify full teardown

```bash
kubectl get pods -n rag
# Expected: No resources found in rag namespace.

kubectl get pvc -n rag
# PVCs will still be listed — this is correct
```

### Shutdown nemoretriever-parse only (leave rest of stack running)

To disable the classification NIM without a full redeploy — useful to free VRAM during
non-ingestion workloads:

```bash
kubectl scale deployment nemoretriever-parse-ms -n rag --replicas=0
```

Restore with:

```bash
kubectl scale deployment nemoretriever-parse-ms -n rag --replicas=1
```

### Scale down the LLM NIM to free VRAM (idle cluster)

```bash
kubectl scale deployment nim-llm -n rag --replicas=0

# Restore
kubectl scale deployment nim-llm -n rag --replicas=1
```

Note: After scaling a NIM back up, allow 2–5 minutes for the Triton server to initialise before
sending inference requests.

---

## Upgrade Procedure

When changes are made to `values-local.yaml` or the chart templates:

```bash
cd /home/joncoons/claude/rag/deploy/helm

ES_PASS=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user \
  -n rag -o jsonpath='{.data.elastic}' | base64 -d)

helm upgrade --install rag ./nvidia-blueprint-rag \
  -n rag \
  -f values-local.yaml \
  --set envVars.APP_VECTORSTORE_PASSWORD="${ES_PASS}" \
  --set "ingestor-server.envVars.APP_VECTORSTORE_PASSWORD=${ES_PASS}"

# Always re-apply NIMCache patch after any upgrade
kubectl patch nimcache nim-llm-cache -n rag --type=merge \
  -p '{"spec":{"storage":{"pvc":{"create":false,"name":"nim-llm-cache-pvc"}}}}'
```

After an upgrade that touches the Elasticsearch CR, force-cycle the ES pod to pick up the
refreshed configuration:

```bash
kubectl delete pod -n rag -l elasticsearch.k8s.elastic.co/cluster-name=rag-eck-elasticsearch
```

---

## Collection Metadata Schema

The collection metadata schema tells the ingestor server which fields to index, what
types they carry, and whether the `FilterExpressionGenerator` LLM should reason about
them when a user asks a natural-language filtered query.  The schema is set once at
collection creation time via `POST /collection`.

### Base lineage schema — all collections

These seven fields are injected automatically by the ingestor for every uploaded document.
Declaring them in the schema enables type-safe filtering and LLM-assisted filter
generation against them.

```bash
curl -X POST http://<node-ip>:8082/collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "reports",
    "description": "Financial and technical reports",
    "metadata_schema": [
      {
        "name": "content_hash",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": false,
        "max_length": 64,
        "description": "SHA-256 hex digest of the original source file"
      },
      {
        "name": "ingested_at",
        "type": "datetime",
        "required": false,
        "support_dynamic_filtering": true,
        "description": "ISO-8601 UTC timestamp of when this document was ingested"
      },
      {
        "name": "pipeline_type",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 64,
        "description": "Pipeline that processed the document: nv_ingest | nemoretriever_parse"
      },
      {
        "name": "source_uri",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 1024,
        "description": "Original filename or URL the document was sourced from"
      },
      {
        "name": "upload_batch_id",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": false,
        "max_length": 36,
        "description": "UUID shared by all documents in a single upload request"
      },
      {
        "name": "source_system",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 128,
        "description": "Origin label: sharepoint, s3, local, web_crawl, etc."
      },
      {
        "name": "document_type",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 32,
        "description": "Lowercase file extension of the source file: pdf, docx, md, etc."
      }
    ]
  }'
```

### Web crawl extension — additional fields for crawled collections

Add these six fields to the `metadata_schema` array (in addition to the seven base lineage
fields) when the collection will hold web crawl content.  They are automatically attached
by the web crawl agent on submission.

```bash
curl -X POST http://<node-ip>:8082/collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "web-docs",
    "description": "Crawled web documentation",
    "metadata_schema": [
      {
        "name": "content_hash",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": false,
        "max_length": 64,
        "description": "SHA-256 hex digest of the chunk file"
      },
      {
        "name": "ingested_at",
        "type": "datetime",
        "required": false,
        "support_dynamic_filtering": true,
        "description": "ISO-8601 UTC timestamp of ingest"
      },
      {
        "name": "pipeline_type",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 64,
        "description": "web_crawl_bs4 | web_crawl_bs4_vlm"
      },
      {
        "name": "source_uri",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 1024,
        "description": "Original page URL"
      },
      {
        "name": "upload_batch_id",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": false,
        "max_length": 36,
        "description": "UUID shared by all files in one ingestor submission batch"
      },
      {
        "name": "source_system",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 128,
        "description": "Origin label — typically web_crawl"
      },
      {
        "name": "document_type",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 32,
        "description": "Always md for web crawl chunks"
      },
      {
        "name": "crawl_session_id",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": false,
        "max_length": 36,
        "description": "UUID shared by all pages from one crawl run"
      },
      {
        "name": "domain",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 253,
        "description": "Netloc of the seed URL (e.g. docs.nvidia.com)"
      },
      {
        "name": "crawl_depth",
        "type": "integer",
        "required": false,
        "support_dynamic_filtering": true,
        "description": "BFS hop distance from the seed URL; 0 = seed page"
      },
      {
        "name": "last_crawled_at",
        "type": "datetime",
        "required": false,
        "support_dynamic_filtering": true,
        "description": "ISO-8601 UTC timestamp of when the page was last crawled"
      },
      {
        "name": "page_title",
        "type": "string",
        "required": false,
        "support_dynamic_filtering": true,
        "max_length": 512,
        "description": "HTML <title> of the crawled page; omitted if blank"
      }
    ]
  }'
```

### Field reference

| Field | Type | `support_dynamic_filtering` | Reason |
|---|---|---|---|
| `content_hash` | `string` | `false` | Opaque hex — LLM has no basis for generating a filter value |
| `upload_batch_id` | `string` | `false` | Opaque UUID — same |
| `crawl_session_id` | `string` | `false` | Opaque UUID — same |
| `ingested_at` | `datetime` | `true` | "Show me docs ingested this week" is a valid natural-language query |
| `pipeline_type` | `string` | `true` | "Only VLM-extracted documents" is meaningful |
| `source_uri` | `string` | `true` | "Docs from annual_report.pdf" is meaningful |
| `source_system` | `string` | `true` | "Docs from SharePoint" is meaningful |
| `document_type` | `string` | `true` | "Only PDF documents" or "Only markdown pages" is meaningful |
| `page_title` | `string` | `true` | "Pages about authentication" (title-match) is meaningful (web crawl only) |
| `domain` | `string` | `true` | "Pages from docs.nvidia.com" is meaningful |
| `crawl_depth` | `integer` | `true` | "Top-level pages only (depth ≤ 1)" is meaningful |
| `last_crawled_at` | `datetime` | `true` | "Pages crawled this month" is meaningful |

### Optional catalog fields

The following fields are part of `DocumentCatalogMetadata` and are not auto-injected — they
must be set explicitly via the catalog update API or passed by your upload client.  Declare
them in the schema if you intend to use catalog-level filtering.

| Field | Type | `support_dynamic_filtering` | Description |
|---|---|---|---|
| `description` | `string` | `true` | Free-text description of the document or page |
| `tags` | `array` | `true` | Categorical tags, e.g. `["finance", "q4-2024"]` |

### Filter operators by type

| Type | Valid operators |
|---|---|
| `string` | `==` `!=` `like` `in` `not in` |
| `datetime` | `==` `!=` `>` `>=` `<` `<=` `between` `before` `after` |
| `integer` | `==` `!=` `>` `>=` `<` `<=` `between` `in` `not in` |

Example filter expressions:

```
pipeline_type == "nemoretriever_parse"
ingested_at after "2025-06-01T00:00:00Z"
ingested_at between "2025-01-01" and "2025-12-31"
source_system in ["sharepoint", "s3"]
document_type in ["pdf", "docx"]
crawl_depth <= 2
source_uri like "%annual_report%"
page_title like "%authentication%"
```

---

## Configuration Reference — Nemotron Parse

All settings live in `values-local.yaml` under `ingestor-server.envVars`.

| Environment Variable | Default in values-local.yaml | Description |
|---|---|---|
| `APP_NEMOPARSE_ENABLED` | `"false"` | Server-side default — when `"true"` all uploads use classification routing unless explicitly disabled per-request |
| `APP_NEMOPARSE_SERVERURL` | `http://nemoretriever-parse-ms:8000/v1/chat/completions` | Full URL of the inference endpoint |
| `APP_NEMOPARSE_MODELNAME` | `nvdev/nvidia/nemoretriever-parse` | Model identifier sent in the API payload |
| `APP_NEMOPARSE_APIKEY` | `""` | Bearer token — inject at deploy time via `--set` if the NIM requires authentication |

**Chunking behaviour** (derived from `nv_ingest` config — `APP_NV_INGEST_CHUNK_SIZE` /
`APP_NV_INGEST_CHUNK_OVERLAP`):

The markdown pre-chunker uses the same `chunk_size` and `chunk_overlap` values as the
NV-Ingest text splitter (defaults: 512 tokens / 150 tokens).  The conversion to
characters uses a 4 chars-per-token approximation, giving a soft limit of ≈ 2 048
characters per chunk.  Tables and code fences are always treated as atomic units
regardless of size.

**`POST /documents` API fields for lineage:**

| Field | Default | Description |
|---|---|---|
| `upload_batch_id` | Auto-generated UUID | Shared identifier for all documents in one upload call. Persist and reuse to correlate re-uploads to the same logical batch. |
| `source_system` | `""` (omitted) | Free-text origin label stored as metadata, e.g. `"sharepoint"`, `"s3"`, `"local"`. |

**NIM deployment settings** (under `nv-ingest.nimOperator.nemoretriever_parse`):

| Setting | Value | Rationale |
|---|---|---|
| `NIM_NUM_MODEL_INSTANCES` | `1` | Single instance serialises requests and caps VRAM; ingest is not latency-sensitive |
| `NIM_TRITON_PERFORMANCE_MODE` | `latency` | Optimises single-request throughput |
| `nodeSelector` | `ubuntu-local-dev` | High-VRAM node; NIMCache PVC is local to this node |
| GPU request/limit | `1` | One time-sliced virtual slot |

---

## Quick-Reference Checklist

### Launch

- [ ] `ngc-secret` and `ngc-api` secrets present in `rag` namespace
- [ ] `eck-es-ca-cert` ConfigMap present in `rag` namespace
- [ ] `nim-llm-cache-pv` PV exists and `nim-llm-cache-pvc` PVC is Bound
- [ ] `ES_PASS` fetched from ECK secret
- [ ] `helm upgrade --install` executed with both `APP_VECTORSTORE_PASSWORD` sets
- [ ] NIMCache PVC patch applied
- [ ] Elasticsearch health = green / yellow
- [ ] All pods Running (allow 60-70 min on cold cluster, 5-10 min warm)
- [ ] `GET /health` returns 200 on RAG server and ingestor
- [ ] `nemoretriever-parse-ms` pod Running and `/v1/health/ready` returns 200

### Create collections (first-time or after cluster wipe)

Run once per collection after the ingestor is healthy.  Skip if collections already
exist and PVCs were preserved from a previous deployment.

**Document collection (PDF / file uploads):**

```bash
curl -X POST http://<node-ip>:8082/collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "reports",
    "description": "Financial and technical reports",
    "metadata_schema": [
      {"name":"content_hash",    "type":"string",   "required":false,"support_dynamic_filtering":false,"max_length":64,   "description":"SHA-256 of source file"},
      {"name":"ingested_at",     "type":"datetime", "required":false,"support_dynamic_filtering":true,                    "description":"Ingest timestamp"},
      {"name":"pipeline_type",   "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":64,   "description":"nv_ingest or nemoretriever_parse"},
      {"name":"source_uri",      "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":1024, "description":"Original filename"},
      {"name":"upload_batch_id", "type":"string",   "required":false,"support_dynamic_filtering":false,"max_length":36,   "description":"Upload batch UUID"},
      {"name":"source_system",   "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":128,  "description":"Origin system label"},
      {"name":"document_type",   "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":32,   "description":"Lowercase file extension: pdf, docx, md, etc."}
    ]
  }'
```

**Web crawl collection:**

```bash
curl -X POST http://<node-ip>:8082/collection \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "web-docs",
    "description": "Crawled web documentation",
    "metadata_schema": [
      {"name":"content_hash",     "type":"string",   "required":false,"support_dynamic_filtering":false,"max_length":64,   "description":"SHA-256 of chunk file"},
      {"name":"ingested_at",      "type":"datetime", "required":false,"support_dynamic_filtering":true,                    "description":"Ingest timestamp"},
      {"name":"pipeline_type",    "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":64,   "description":"web_crawl_bs4 or web_crawl_bs4_vlm"},
      {"name":"source_uri",       "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":1024, "description":"Original page URL"},
      {"name":"upload_batch_id",  "type":"string",   "required":false,"support_dynamic_filtering":false,"max_length":36,   "description":"Submission batch UUID"},
      {"name":"source_system",    "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":128,  "description":"Origin label"},
      {"name":"document_type",    "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":32,   "description":"Always md for web crawl chunks"},
      {"name":"crawl_session_id", "type":"string",   "required":false,"support_dynamic_filtering":false,"max_length":36,   "description":"Crawl run UUID"},
      {"name":"domain",           "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":253,  "description":"Seed URL netloc"},
      {"name":"crawl_depth",      "type":"integer",  "required":false,"support_dynamic_filtering":true,                    "description":"BFS hops from seed URL"},
      {"name":"last_crawled_at",  "type":"datetime", "required":false,"support_dynamic_filtering":true,                    "description":"Last crawl timestamp"},
      {"name":"page_title",       "type":"string",   "required":false,"support_dynamic_filtering":true, "max_length":512,  "description":"HTML title of the crawled page"}
    ]
  }'
```

Verify collection was created:

```bash
curl -s http://<node-ip>:8082/collections | jq '.collections[].collection_name'
```

### Shutdown

- [ ] `helm uninstall rag -n rag`
- [ ] Confirm no pods remain in `rag` namespace
- [ ] PVCs remain (expected — preserves caches and data)
