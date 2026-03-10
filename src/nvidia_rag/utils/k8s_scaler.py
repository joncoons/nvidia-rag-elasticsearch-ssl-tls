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
Kubernetes deployment scaler for crawl-mode GPU resource management.

Switches between two GPU layouts:

  Crawl mode      — maximise nemotron-parse throughput for PDF extraction.
                    nim-llm=0, gpu0-placeholder=0, nemotron-parse=CRAWL_REPLICAS

  Inference mode  — standard RAG serving with nim-llm on GPU0.
                    nim-llm=1, gpu0-placeholder=PLACEHOLDER_REPLICAS,
                    nemotron-parse=INFERENCE_REPLICAS

Ordering for inference-mode restore matters for device-plugin first-fit GPU
assignment:
  1. nemotron-parse → 0   free all GPU slots; wait for termination
  2. nim-llm        → 1   lands on GPU0 (first empty GPU)
  3. gpu0-placeholder → N fill remaining GPU0 slots
  4. nemotron-parse → N   forced to GPU1 by GPU0 full + nim-llm VRAM barrier

Fails silently if the Kubernetes API is unavailable (local dev, missing RBAC).
Requires: `kubernetes` Python package and a ServiceAccount with patch/get
permissions on Deployments in the target namespace.
"""

import logging
import os
import threading
import time
from typing import Optional

_NIM_LLM_PROFILE = os.environ.get(
    "NIM_LLM_PROFILE",
    "e9cc0c5ea49283a493a0b18a05a97eb9b15a82a0d6acbb967e35609ddeb767fa",
)

logger = logging.getLogger(__name__)

_NAMESPACE = os.environ.get("K8S_SCALER_NAMESPACE", "rag")
_CRAWL_REPLICAS = int(os.environ.get("K8S_CRAWL_NEMOTRON_REPLICAS", "7"))
_INFERENCE_REPLICAS = int(os.environ.get("K8S_INFERENCE_NEMOTRON_REPLICAS", "1"))
_PLACEHOLDER_REPLICAS = int(os.environ.get("K8S_PLACEHOLDER_REPLICAS", "3"))


def _get_apps_v1():
    """Return a kubernetes AppsV1Api client, or None if unavailable."""
    try:
        from kubernetes import client, config as k8s_config
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        return client.AppsV1Api()
    except Exception as exc:
        logger.warning("k8s client unavailable — deployment scaling disabled: %r", exc)
        return None


def _patch_replicas(apps_v1, name: str, replicas: int) -> bool:
    """Patch a Deployment's replica count. Returns True on success."""
    try:
        apps_v1.patch_namespaced_deployment_scale(
            name=name,
            namespace=_NAMESPACE,
            body={"spec": {"replicas": replicas}},
        )
        logger.info("k8s: scaled %s/%s → %d replicas", _NAMESPACE, name, replicas)
        return True
    except Exception as exc:
        logger.warning("k8s: failed to scale %s/%s: %r", _NAMESPACE, name, exc)
        return False


def _wait_scheduled(name: str, timeout: int = 200) -> None:
    """Poll until at least one pod for the deployment is Running (GPU slot claimed)."""
    try:
        from kubernetes import client, config as k8s_config
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        core_v1 = client.CoreV1Api()
    except Exception as exc:
        logger.warning("k8s: _wait_scheduled unavailable: %r", exc)
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pods = core_v1.list_namespaced_pod(
                _NAMESPACE, label_selector=f"app={name}"
            )
            for pod in pods.items:
                if pod.status and pod.status.phase == "Running":
                    logger.info("k8s: %s/%s pod is Running — GPU slot claimed", _NAMESPACE, name)
                    return
        except Exception:
            pass
        time.sleep(5)
    logger.warning("k8s: timeout waiting for %s/%s to reach Running", _NAMESPACE, name)


def _wait_scaled_down(apps_v1, name: str, timeout: int = 120) -> None:
    """Poll until ready_replicas == 0 or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            dep = apps_v1.read_namespaced_deployment(name, _NAMESPACE)
            if (dep.status.ready_replicas or 0) == 0:
                logger.info("k8s: %s/%s scaled down", _NAMESPACE, name)
                return
        except Exception:
            pass
        time.sleep(3)
    logger.warning("k8s: timeout waiting for %s/%s to scale down", _NAMESPACE, name)


def enable_crawl_mode() -> None:
    """
    Switch to crawl-optimised GPU layout (fire-and-forget, non-blocking).

    nim-llm=0, gpu0-placeholder=0, nemotron-parse=CRAWL_REPLICAS
    New nemotron-parse pods start immediately; no need to wait before crawling.
    """
    apps_v1 = _get_apps_v1()
    if apps_v1 is None:
        return

    logger.info(
        "k8s: enabling crawl mode — nim-llm=0, nemotron-parse=%d, nim-vlm=1", _CRAWL_REPLICAS
    )
    _patch_replicas(apps_v1, "nim-llm", 0)
    _patch_replicas(apps_v1, "gpu0-placeholder", 0)
    _patch_replicas(apps_v1, "nemotron-parse-v12", _CRAWL_REPLICAS)
    # nim-vlm stays up — 1 slot on GPU0 fits within the 8-slot budget (7 parse + 1 vlm)
    _patch_replicas(apps_v1, "nim-vlm", 1)


def _restore_inference_blocking() -> None:
    """
    Restore inference GPU layout with ordered scaling (runs in background thread).

    Ordering is critical for device-plugin first-fit GPU assignment — see module
    docstring.
    """
    apps_v1 = _get_apps_v1()
    if apps_v1 is None:
        return

    logger.info("k8s: restoring inference mode — beginning ordered scale sequence")

    # Step 1: scale nemotron-parse to 0, wait for pods to terminate
    _patch_replicas(apps_v1, "nemotron-parse-v12", 0)
    _wait_scaled_down(apps_v1, "nemotron-parse-v12", timeout=120)

    # Step 2: ensure nim-llm uses the correct single-GPU TP1 NVFP4 profile,
    # then scale to 1. Device-plugin assigns it to the first available GPU.
    try:
        apps_v1.patch_namespaced_deployment(
            name="nim-llm",
            namespace=_NAMESPACE,
            body={"spec": {"template": {"spec": {"containers": [
                {"name": "nim-llm-ctr", "env": [
                    {"name": "NIM_MODEL_PROFILE", "value": _NIM_LLM_PROFILE}
                ]}
            ]}}}},
        )
        logger.info("k8s: nim-llm NIM_MODEL_PROFILE set to %s", _NIM_LLM_PROFILE)
    except Exception as exc:
        logger.warning("k8s: could not patch nim-llm NIM_MODEL_PROFILE: %r", exc)
        # Fallback: use kubectl set env via subprocess
        try:
            import subprocess
            subprocess.run(
                ["kubectl", "set", "env", "-n", _NAMESPACE, "deploy/nim-llm",
                 f"NIM_MODEL_PROFILE={_NIM_LLM_PROFILE}"],
                check=True, capture_output=True,
            )
            logger.info("k8s: nim-llm NIM_MODEL_PROFILE set via kubectl")
        except Exception as exc2:
            logger.warning("k8s: kubectl fallback also failed: %r", exc2)

    _patch_replicas(apps_v1, "nim-llm", 1)

    # Step 3: wait for nim-llm pod to reach Running (GPU slot claimed) before
    # filling remaining slots — prevents gpu0-placeholder racing nim-llm.
    _wait_scheduled("nim-llm")

    # Step 4: fill remaining GPU0 slots with placeholders
    _patch_replicas(apps_v1, "gpu0-placeholder", _PLACEHOLDER_REPLICAS)

    # Step 4: nemotron-parse inference replicas → forced to GPU1
    _patch_replicas(apps_v1, "nemotron-parse-v12", _INFERENCE_REPLICAS)

    logger.info(
        "k8s: inference mode restore initiated — nim-llm startup may take several minutes"
    )


def disable_crawl_mode() -> None:
    """
    Restore inference mode in a background daemon thread (non-blocking).

    The ordered scale sequence runs asynchronously so the crawl task can
    return its result without waiting for nim-llm to finish loading.
    """
    t = threading.Thread(
        target=_restore_inference_blocking,
        daemon=True,
        name="restore-inference-mode",
    )
    t.start()
