#!/usr/bin/env bash
# soft-start.sh
# Scale up all GPU services and the app layer.
# Assumes the k3s cluster is already running and both nodes are Ready.
# Re-applies patches that the NIM Operator reconcile loop resets on restart.

set -euo pipefail

NS=rag

# ---------------------------------------------------------------------------
# 1. App layer (no GPU dependency — start early)
# ---------------------------------------------------------------------------
echo "[soft-start] Starting app layer..."
kubectl scale deployment -n $NS \
  rag-server \
  ingestor-server \
  rag-frontend \
  rag-nv-ingest \
  --replicas=1

# ---------------------------------------------------------------------------
# 2. ubuntu2 GPU services (independent of ubuntu-local-dev)
# ---------------------------------------------------------------------------
echo "[soft-start] Starting ubuntu2 GPU services..."
kubectl scale deployment -n $NS \
  nemoretriever-embedding-ms \
  nemoretriever-ranking-ms \
  nemoretriever-vlm-embedding-ms \
  nemoretriever-graphic-elements-v1 \
  nemoretriever-ocr-v1 \
  nemoretriever-page-elements-v3 \
  nemoretriever-table-structure-v1 \
  audio \
  --replicas=1

# ---------------------------------------------------------------------------
# 3. ubuntu-local-dev GPU services
#    Start nim-llm first — its ~90 GB VRAM footprint fills one physical GPU
#    completely, acting as a VRAM barrier that forces nim-vlm and
#    nemotron-parse onto the other GPU.
# ---------------------------------------------------------------------------
echo "[soft-start] Starting nim-llm (ubuntu-local-dev)..."
kubectl scale deployment -n $NS nim-llm --replicas=1

echo "[soft-start] Waiting up to 60s for nim-llm pod to be scheduled..."
kubectl wait pod -n $NS -l app=nim-llm \
  --for=condition=PodScheduled --timeout=60s 2>/dev/null || true

# gpu0-placeholder fills the 3 remaining slots on the nim-llm GPU so that
# nim-vlm and nemotron-parse are forced to the other GPU.
echo "[soft-start] Starting gpu0-placeholder (3 replicas to fill nim-llm GPU slots)..."
kubectl scale deployment -n $NS gpu0-placeholder --replicas=3

echo "[soft-start] Waiting up to 30s for placeholders to be scheduled..."
sleep 10

echo "[soft-start] Starting nim-vlm and nemotron-parse-v12..."
kubectl scale deployment -n $NS nim-vlm --replicas=1
kubectl scale deployment -n $NS nemotron-parse-v12 --replicas=3

# ---------------------------------------------------------------------------
# 4. Re-apply patches that NIM Operator resets on reconcile
# ---------------------------------------------------------------------------
echo "[soft-start] Re-applying NIM Operator patches..."

# nim-llm: NIM Operator sets gpu:4 for 49B model; patch back to 1
kubectl patch nimservice -n $NS nim-llm --type=json -p='[
  {"op":"replace","path":"/spec/resources/limits/nvidia.com~1gpu","value":"1"},
  {"op":"replace","path":"/spec/resources/requests/nvidia.com~1gpu","value":"1"}
]' 2>/dev/null && echo "  [patch] nim-llm gpu:1 applied" || echo "  [warn] nim-llm NIMService patch skipped (may not be needed)"

# embedding-ms: TRT engine compiled for 2048; operator default is 8192
# Use replace if the entry already exists, add otherwise.
_patch_seq_len() {
  local svc=$1 val=$2
  local idx
  idx=$(kubectl get nimservice -n $NS "$svc" -o json 2>/dev/null | \
        python3 -c "
import sys,json
d=json.load(sys.stdin)
env=d.get('spec',{}).get('env',[])
for i,e in enumerate(env):
    if e.get('name')=='NIM_TRITON_MAX_SEQ_LENGTH':
        print(i); sys.exit(0)
print(-1)
" 2>/dev/null)
  if [[ "$idx" -ge 0 ]]; then
    kubectl patch nimservice -n $NS "$svc" --type=json -p="[
      {\"op\":\"replace\",\"path\":\"/spec/env/${idx}/value\",\"value\":\"${val}\"}
    ]" 2>/dev/null
  else
    kubectl patch nimservice -n $NS "$svc" --type=json -p="[
      {\"op\":\"add\",\"path\":\"/spec/env/-\",\"value\":{\"name\":\"NIM_TRITON_MAX_SEQ_LENGTH\",\"value\":\"${val}\"}}
    ]" 2>/dev/null
  fi
}

_patch_seq_len nemoretriever-embedding-ms 2048 && \
  echo "  [patch] embedding-ms NIM_TRITON_MAX_SEQ_LENGTH=2048 applied" || true

_patch_seq_len nemoretriever-ranking-ms 8192 && \
  echo "  [patch] ranking-ms NIM_TRITON_MAX_SEQ_LENGTH=8192 applied" || true

# ---------------------------------------------------------------------------
# 5. Status summary
# ---------------------------------------------------------------------------
echo ""
echo "[soft-start] Startup complete. Current pod status:"
kubectl get pods -n $NS -o wide --no-headers | \
  awk '{printf "  %-55s %-12s %-20s\n", $1, $3, $8}' | sort

echo ""
echo "[soft-start] Note: audio (Parakeet ASR) takes ~15 min for TRT compilation on first start."
echo "             nim-vlm and nim-llm typically take 3-5 min to reach Ready."
