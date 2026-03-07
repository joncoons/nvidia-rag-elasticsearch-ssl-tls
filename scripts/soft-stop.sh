#!/usr/bin/env bash
# soft-stop.sh
# Scale down all GPU services to free VRAM.
# App layer (rag-server, ingestor-server, rag-frontend, rag-nv-ingest) and
# infrastructure (Elasticsearch, Redis, MinIO) remain running.

set -euo pipefail

NS=rag

# --- GPU services: ubuntu-local-dev ---
echo "[soft-stop] Scaling down ubuntu-local-dev GPU services..."
kubectl scale deployment -n $NS \
  nim-llm \
  nim-vlm \
  nemotron-parse-v12 \
  gpu0-placeholder \
  --replicas=0

# --- GPU services: ubuntu2 ---
echo "[soft-stop] Scaling down ubuntu2 GPU services..."
kubectl scale deployment -n $NS \
  nemoretriever-embedding-ms \
  nemoretriever-ranking-ms \
  nemoretriever-vlm-embedding-ms \
  nemoretriever-graphic-elements-v1 \
  nemoretriever-ocr-v1 \
  nemoretriever-page-elements-v3 \
  nemoretriever-table-structure-v1 \
  audio \
  --replicas=0

echo "[soft-stop] Waiting for GPU pods to terminate (up to 120s)..."
kubectl wait pod -n $NS \
  --for=delete \
  --timeout=120s \
  -l 'app in (nim-llm,nim-vlm,nemotron-parse-v12,gpu0-placeholder,nemoretriever-embedding-ms,nemoretriever-ranking-ms,nemoretriever-vlm-embedding-ms,nemoretriever-graphic-elements-v1,nemoretriever-ocr-v1,nemoretriever-page-elements-v3,nemoretriever-table-structure-v1,audio)' \
  2>/dev/null || true

echo "[soft-stop] Done. App layer and infrastructure are still running."
echo "            Use soft-start.sh or cold-start.sh to bring GPU services back up."
