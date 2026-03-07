#!/usr/bin/env bash
# crawl-mode.sh — Toggle between crawl mode and inference mode
#
# Crawl mode:     nim-llm=0, gpu0-placeholder=0, nemotron-parse=CRAWL_REPLICAS
# Inference mode: nim-llm=1, gpu0-placeholder=PLACEHOLDER_REPLICAS, nemotron-parse=INFERENCE_REPLICAS
#
# GPU placement logic (disable):
#   1. Scale nemotron-parse to 0 — free all GPU slots
#   2. Scale nim-llm to 1 — device-plugin assigns it to GPU0 (first-fit, GPU0 empty)
#   3. Scale gpu0-placeholder to PLACEHOLDER_REPLICAS — fill remaining GPU0 slots
#   4. Scale nemotron-parse to INFERENCE_REPLICAS — forced to GPU1 (GPU0 full + VRAM barrier)

set -euo pipefail

NAMESPACE="${NAMESPACE:-rag}"
NEMOTRON_PARSE_CRAWL_REPLICAS="${NEMOTRON_PARSE_CRAWL_REPLICAS:-7}"
NEMOTRON_PARSE_INFERENCE_REPLICAS="${NEMOTRON_PARSE_INFERENCE_REPLICAS:-3}"
GPU0_PLACEHOLDER_REPLICAS="${GPU0_PLACEHOLDER_REPLICAS:-3}"

usage() {
    echo "Usage: $0 <enable|disable|status>"
    echo ""
    echo "  enable   Switch to crawl mode"
    echo "           nim-llm=0, gpu0-placeholder=0, nemotron-parse=${NEMOTRON_PARSE_CRAWL_REPLICAS}"
    echo "  disable  Restore inference mode"
    echo "           nim-llm=1, gpu0-placeholder=${GPU0_PLACEHOLDER_REPLICAS}, nemotron-parse=${NEMOTRON_PARSE_INFERENCE_REPLICAS}"
    echo "  status   Show current replica counts and pod readiness"
    echo ""
    echo "Overrides (env vars):"
    echo "  NAMESPACE                         (default: rag)"
    echo "  NEMOTRON_PARSE_CRAWL_REPLICAS     (default: 7)"
    echo "  NEMOTRON_PARSE_INFERENCE_REPLICAS (default: 3)"
    echo "  GPU0_PLACEHOLDER_REPLICAS         (default: 3)"
    exit 1
}

wait_rollout() {
    local dep="$1"
    local timeout="${2:-300}"
    echo "  Waiting for ${dep}..."
    kubectl rollout status deployment/"${dep}" -n "${NAMESPACE}" --timeout="${timeout}s"
}

get_replicas() {
    local dep="$1"
    local ready desired
    ready=$(kubectl get deployment "${dep}" -n "${NAMESPACE}" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    desired=$(kubectl get deployment "${dep}" -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    echo "${ready:-0}/${desired:-0}"
}

status() {
    echo "=== Deployment Replicas ==="
    for dep in nim-llm gpu0-placeholder nemotron-parse-v12 nim-vlm; do
        printf "  %-30s %s ready\n" "${dep}" "$(get_replicas "${dep}")"
    done
    echo ""
    echo "=== Pod Status ==="
    kubectl get pods -n "${NAMESPACE}" \
        -l "app in (nim-llm,gpu0-placeholder,nemotron-parse-v12,nim-vlm)" \
        --no-headers 2>/dev/null \
        | awk '{printf "  %-50s %-12s %s\n", $1, $2, $3}' \
        || echo "  (none)"
}

enable_crawl_mode() {
    echo "=== Enabling Crawl Mode ==="

    echo ""
    echo "[1/3] Scaling nim-llm and gpu0-placeholder to 0..."
    kubectl scale deployment nim-llm gpu0-placeholder \
        -n "${NAMESPACE}" --replicas=0

    echo ""
    echo "[2/3] Scaling nemotron-parse-v12 to ${NEMOTRON_PARSE_CRAWL_REPLICAS}..."
    kubectl scale deployment nemotron-parse-v12 \
        -n "${NAMESPACE}" --replicas="${NEMOTRON_PARSE_CRAWL_REPLICAS}"
    wait_rollout nemotron-parse-v12 300

    echo ""
    echo "[3/3] Done."
    echo ""
    status
    echo ""
    echo "Crawl mode active: nim-llm=0, nemotron-parse=${NEMOTRON_PARSE_CRAWL_REPLICAS}"
}

disable_crawl_mode() {
    echo "=== Restoring Inference Mode ==="

    echo ""
    echo "[1/4] Scaling nemotron-parse-v12 to 0 (free GPU slots for nim-llm placement)..."
    kubectl scale deployment nemotron-parse-v12 \
        -n "${NAMESPACE}" --replicas=0
    wait_rollout nemotron-parse-v12 120

    echo ""
    echo "[2/4] Scaling nim-llm to 1 (will land on GPU0, first-fit)..."
    kubectl scale deployment nim-llm \
        -n "${NAMESPACE}" --replicas=1
    wait_rollout nim-llm 600

    echo ""
    echo "[3/4] Scaling gpu0-placeholder to ${GPU0_PLACEHOLDER_REPLICAS} (fill remaining GPU0 slots)..."
    kubectl scale deployment gpu0-placeholder \
        -n "${NAMESPACE}" --replicas="${GPU0_PLACEHOLDER_REPLICAS}"

    echo ""
    echo "[4/4] Scaling nemotron-parse-v12 to ${NEMOTRON_PARSE_INFERENCE_REPLICAS} (GPU1, forced by GPU0 VRAM barrier)..."
    kubectl scale deployment nemotron-parse-v12 \
        -n "${NAMESPACE}" --replicas="${NEMOTRON_PARSE_INFERENCE_REPLICAS}"
    wait_rollout nemotron-parse-v12 300

    echo ""
    status
    echo ""
    echo "Inference mode active: nim-llm=1, nemotron-parse=${NEMOTRON_PARSE_INFERENCE_REPLICAS}"
}

case "${1:-}" in
    enable)  enable_crawl_mode ;;
    disable) disable_crawl_mode ;;
    status)  status ;;
    *)       usage ;;
esac
