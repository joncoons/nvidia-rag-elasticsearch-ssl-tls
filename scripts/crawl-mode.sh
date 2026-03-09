#!/usr/bin/env bash
# crawl-mode.sh — manually switch GPU layout between crawl mode and inference mode.
#
# Crawl mode:     nim-llm=0, gpu0-placeholder=0, nemotron-parse=7
#                 Maximises nemotron-parse throughput; GPU chat is unavailable.
#
# Inference mode: nim-llm=1 (GPU0 exclusive), gpu0-placeholder=3 (fills GPU0),
#                 nemotron-parse=1 (GPU1).
#                 Ordering matters for device-plugin first-fit GPU assignment:
#                   1. nemotron-parse → 0  (free all GPU slots; wait for termination)
#                   2. nim-llm → 1         (first-fit → GPU0)
#                   3. gpu0-placeholder → 3 (fill remaining GPU0 slots)
#                   4. nemotron-parse → 1  (forced to GPU1)
#
# Usage:
#   scripts/crawl-mode.sh enable    # switch to crawl mode
#   scripts/crawl-mode.sh disable   # restore inference mode (default)

set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-rag}"
CRAWL_PARSE_REPLICAS="${K8S_CRAWL_NEMOTRON_REPLICAS:-7}"
INFERENCE_PARSE_REPLICAS="${K8S_INFERENCE_NEMOTRON_REPLICAS:-1}"
PLACEHOLDER_REPLICAS="${K8S_PLACEHOLDER_REPLICAS:-3}"
# TP1 NVFP4 single-GPU profile for RTX PRO 6000 Blackwell (svx1 = single card).
# SHA256 hash of the TP1 NVFP4 vllm profile — confirmed via `docker run list-model-profiles`.
# Use the hash (not the snapshot directory name) as NIM_MODEL_PROFILE value.
NIM_LLM_PROFILE="${NIM_LLM_PROFILE:-e9cc0c5ea49283a493a0b18a05a97eb9b15a82a0d6acbb967e35609ddeb767fa}"

scale() {
    echo "  → scaling ${NAMESPACE}/${1} to ${2} replicas"
    kubectl scale -n "${NAMESPACE}" "deploy/${1}" --replicas="${2}"
}

wait_down() {
    echo "  → waiting for ${NAMESPACE}/${1} pods to terminate..."
    kubectl rollout status -n "${NAMESPACE}" "deploy/${1}" --timeout=120s 2>/dev/null || true
    # Extra safety: wait until ready_replicas == 0
    for _ in $(seq 1 40); do
        ready=$(kubectl get deploy -n "${NAMESPACE}" "${1}" \
                  -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        [ "${ready:-0}" -eq 0 ] && return
        sleep 3
    done
    echo "  WARNING: timeout waiting for ${1} to scale down — proceeding anyway"
}

wait_scheduled() {
    echo "  → waiting for ${NAMESPACE}/${1} pod to be scheduled (GPU slot claimed)..."
    for _ in $(seq 1 40); do
        phase=$(kubectl get pods -n "${NAMESPACE}" -l "app=${1}" \
                  -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
        [ "${phase}" = "Running" ] && return
        sleep 5
    done
    echo "  WARNING: ${1} not yet Running — GPU0 slot may not be claimed; proceeding anyway"
}

case "${1:-disable}" in
    enable|on)
        echo "Enabling crawl mode..."
        scale nim-llm 0
        scale gpu0-placeholder 0
        scale nemotron-parse-v12 "${CRAWL_PARSE_REPLICAS}"
        # nim-vlm stays up in crawl mode (1 slot on GPU0; fits within 8-slot budget)
        scale nim-vlm 1
        echo "Crawl mode active. nim-llm is offline; RAG chat unavailable."
        ;;
    disable|off)
        echo "Disabling crawl mode — restoring inference layout..."
        # Step 1: free all GPU slots
        scale nemotron-parse-v12 0
        wait_down nemotron-parse-v12
        # Step 2: ensure nim-llm uses the correct TP1 NVFP4 profile
        echo "  → setting nim-llm profile to ${NIM_LLM_PROFILE}"
        kubectl set env -n "${NAMESPACE}" deploy/nim-llm \
            NIM_MODEL_PROFILE="${NIM_LLM_PROFILE}"
        # Step 3: nim-llm lands on first available GPU (first-fit)
        scale nim-llm 1
        # Step 3a: wait for nim-llm pod to reach Running phase (GPU0 slot claimed)
        #           before filling remaining slots — prevents placeholder racing nim-llm
        wait_scheduled nim-llm
        # Step 4: fill remaining GPU0 slots so nemotron-parse is forced to GPU1
        scale gpu0-placeholder "${PLACEHOLDER_REPLICAS}"
        # Step 5: single nemotron-parse instance on GPU1
        scale nemotron-parse-v12 "${INFERENCE_PARSE_REPLICAS}"
        echo "Inference mode restored. nim-llm startup may take several minutes."
        ;;
    status)
        kubectl get deploy -n "${NAMESPACE}" nim-llm nim-vlm gpu0-placeholder nemotron-parse-v12 \
            -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas'
        ;;
    *)
        echo "Usage: $0 {enable|disable|status}"
        exit 1
        ;;
esac
