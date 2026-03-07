#!/usr/bin/env bash
# cold-start.sh
# Full cold start after a machine reboot or cluster restart.
# Waits for both k3s nodes to be Ready and Elasticsearch to be healthy,
# then delegates to soft-start.sh.

set -euo pipefail

NS=rag
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Helper: print timestamped message
# ---------------------------------------------------------------------------
log() { echo "[cold-start $(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# 1. Wait for both k3s nodes to be Ready
# ---------------------------------------------------------------------------
log "Waiting for k3s nodes to be Ready..."
for NODE in ubuntu-local-dev ubuntu2; do
  log "  Waiting for node: $NODE"
  until kubectl get node "$NODE" --no-headers 2>/dev/null | grep -q " Ready"; do
    sleep 5
  done
  log "  $NODE is Ready"
done

# ---------------------------------------------------------------------------
# 2. Wait for Elasticsearch to be healthy
#    ECK reports cluster health via the Elasticsearch CR status
# ---------------------------------------------------------------------------
log "Waiting for Elasticsearch to be healthy (up to 5 min)..."
DEADLINE=$(( $(date +%s) + 300 ))
until kubectl get elasticsearch -n $NS rag-eck-elasticsearch \
      -o jsonpath='{.status.health}' 2>/dev/null | grep -q "green\|yellow"; do
  if [[ $(date +%s) -gt $DEADLINE ]]; then
    log "WARNING: Elasticsearch did not become healthy within 5 min — continuing anyway."
    break
  fi
  sleep 10
done
ES_HEALTH=$(kubectl get elasticsearch -n $NS rag-eck-elasticsearch \
            -o jsonpath='{.status.health}' 2>/dev/null || echo "unknown")
log "Elasticsearch health: $ES_HEALTH"

# ---------------------------------------------------------------------------
# 3. Wait for Redis to be ready (rag-nv-ingest dependency)
# ---------------------------------------------------------------------------
log "Waiting for Redis master to be Ready..."
kubectl wait pod -n $NS -l 'app.kubernetes.io/name=redis,app.kubernetes.io/component=master' \
  --for=condition=Ready --timeout=120s 2>/dev/null || \
  log "WARNING: Redis not Ready within 120s — continuing anyway."

# ---------------------------------------------------------------------------
# 4. Wait for MinIO to be ready (ingestor dependency)
# ---------------------------------------------------------------------------
log "Waiting for MinIO to be Ready..."
kubectl wait pod -n $NS -l app=rag-minio \
  --for=condition=Ready --timeout=60s 2>/dev/null || \
  log "WARNING: MinIO not Ready within 60s — continuing anyway."

# ---------------------------------------------------------------------------
# 5. Check for any GPU services stuck in CrashLoopBackOff and delete them
#    so they reschedule cleanly (stale TRT caches sometimes cause this)
# ---------------------------------------------------------------------------
log "Checking for CrashLoopBackOff pods..."
CRASH_PODS=$(kubectl get pods -n $NS --no-headers 2>/dev/null | \
             awk '$4 == "CrashLoopBackOff" {print $1}')
if [[ -n "$CRASH_PODS" ]]; then
  log "Deleting CrashLoopBackOff pods: $CRASH_PODS"
  echo "$CRASH_PODS" | xargs kubectl delete pod -n $NS --grace-period=0
else
  log "No CrashLoopBackOff pods found."
fi

# ---------------------------------------------------------------------------
# 6. Delegate to soft-start.sh for scaling + patching
# ---------------------------------------------------------------------------
log "Cluster is healthy. Delegating to soft-start.sh..."
echo ""
exec "$SCRIPT_DIR/soft-start.sh"
