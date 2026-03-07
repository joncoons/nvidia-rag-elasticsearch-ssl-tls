#!/usr/bin/env bash
# hard-stop.sh
# Scale down all GPU services AND the app layer.
# Infrastructure (Elasticsearch, Redis, MinIO) remains running.
# Use this before a planned maintenance window or node reboot.

set -euo pipefail

NS=rag

# --- GPU services (same as soft-stop) ---
echo "[hard-stop] Scaling down GPU services..."
kubectl scale deployment -n $NS \
  nim-llm \
  nim-vlm \
  nemotron-parse-v12 \
  gpu0-placeholder \
  nemoretriever-embedding-ms \
  nemoretriever-ranking-ms \
  nemoretriever-vlm-embedding-ms \
  nemoretriever-graphic-elements-v1 \
  nemoretriever-ocr-v1 \
  nemoretriever-page-elements-v3 \
  nemoretriever-table-structure-v1 \
  audio \
  --replicas=0

# --- App layer ---
echo "[hard-stop] Scaling down app layer..."
kubectl scale deployment -n $NS \
  rag-server \
  ingestor-server \
  rag-frontend \
  rag-nv-ingest \
  --replicas=0

echo "[hard-stop] Waiting for all non-infrastructure pods to terminate (up to 120s)..."
kubectl wait pod -n $NS \
  --for=delete \
  --timeout=120s \
  -l 'app in (nim-llm,nim-vlm,nemotron-parse-v12,gpu0-placeholder,nemoretriever-embedding-ms,nemoretriever-ranking-ms,nemoretriever-vlm-embedding-ms,nemoretriever-graphic-elements-v1,nemoretriever-ocr-v1,nemoretriever-page-elements-v3,nemoretriever-table-structure-v1,audio,rag-server,ingestor-server,rag-frontend,rag-nv-ingest)' \
  2>/dev/null || true

echo "[hard-stop] Done. Only Elasticsearch, Redis, and MinIO are still running."
echo "            Use cold-start.sh to bring everything back up."
