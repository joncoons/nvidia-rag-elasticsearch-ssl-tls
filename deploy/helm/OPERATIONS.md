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
- Replace the PDF with a structured Markdown file before submitting to NV-Ingest
- Documents with no complex elements fall through to the standard NV-Ingest pipeline

### Via the API directly

```bash
# Upload with classification routing enabled
curl -X POST http://<node-ip>:8082/documents \
  -F "documents=@/path/to/report.pdf" \
  -F 'data={"collection_name":"my-collection","blocking":false,"use_nemoretriever_parse":true}'
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
nemoretriever-parse output for 'annual_report.pdf' written to: /tmp/annual_report_nemoparse_XXXX.md
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

## Configuration Reference — Nemotron Parse

All settings live in `values-local.yaml` under `ingestor-server.envVars`.

| Environment Variable | Default in values-local.yaml | Description |
|---|---|---|
| `APP_NEMOPARSE_ENABLED` | `"false"` | Server-side default — when `"true"` all uploads use classification routing unless explicitly disabled per-request |
| `APP_NEMOPARSE_SERVERURL` | `http://nemoretriever-parse-ms:8000/v1/chat/completions` | Full URL of the inference endpoint |
| `APP_NEMOPARSE_MODELNAME` | `nvdev/nvidia/nemoretriever-parse` | Model identifier sent in the API payload |
| `APP_NEMOPARSE_APIKEY` | `""` | Bearer token — inject at deploy time via `--set` if the NIM requires authentication |

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

### Shutdown

- [ ] `helm uninstall rag -n rag`
- [ ] Confirm no pods remain in `rag` namespace
- [ ] PVCs remain (expected — preserves caches and data)
