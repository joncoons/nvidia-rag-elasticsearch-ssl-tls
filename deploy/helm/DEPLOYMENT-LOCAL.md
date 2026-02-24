# Local k3s Deployment — NVIDIA RAG Blueprint

This document describes the changes made to the upstream
[NVIDIA-AI-Blueprints/rag](https://github.com/NVIDIA-AI-Blueprints/rag) Helm chart to support a
local k3s deployment using ECK-managed Elasticsearch as the vector store, NVIDIA NIM Operator for
GPU-accelerated inference, and time-sliced GPU sharing across two nodes.

---

## Environment

| Component | Details |
|---|---|
| Kubernetes | k3s v1.35.1+k3s1 |
| Nodes | `ubuntu-local-dev` (2x RTX Pro 6000 Blackwell 96 GB), `ubuntu2` (1x RTX 5090 32 GB) |
| GPU sharing | Device plugin time-slicing, factor 4 (8 slots on ubuntu-local-dev, 4 on ubuntu2) |
| Vector store | Elasticsearch 9.x via ECK 3.x (replaces default Milvus) |
| LLM NIM | `nvidia/llama-3.3-nemotron-super-49b-v1.5`, NVFP4, pre-cached on NVMe |

---

## Changes from Upstream

### 1. `deploy/helm/nvidia-blueprint-rag/templates/deployment.yaml`

**What changed:** Added `extraVolumes` and `extraVolumeMounts` support to the rag-server deployment.

**Why:** The upstream template hardcodes the `prompt-volume` as the only volume/volumeMount. To
mount the ECK CA certificate (required for TLS verification against the ECK-managed Elasticsearch)
without forking the entire template, `extraVolumes` / `extraVolumeMounts` list blocks were added,
following standard Helm chart conventions.

```diff
 volumeMounts:
   - name: prompt-volume
     mountPath: /prompt.yaml
     subPath: prompt.yaml
+  {{- if .Values.extraVolumeMounts }}
+  {{- toYaml .Values.extraVolumeMounts | nindent 12 }}
+  {{- end }}
 volumes:
   - name: prompt-volume
     configMap:
       name: {{ include "nvidia-blueprint-rag.fullname" . }}-prompt
       defaultMode: 0555
+  {{- if .Values.extraVolumes }}
+  {{- toYaml .Values.extraVolumes | nindent 8 }}
+  {{- end }}
```

---

### 2. `deploy/helm/nvidia-blueprint-rag/templates/ingestor-server-deployment.yaml`

**What changed:** Added `extraVolumes` and `extraVolumeMounts` support to the ingestor-server
deployment.

**Why:** Same reason as rag-server — the ECK CA cert must be trusted by the ingestor's
`elastic_transport` client. The upstream template only conditionally adds volumes when
`persistence.enabled` is true. The change wraps both persistence and extra-volume logic under
a combined `or` condition so that volumes are rendered whenever either is needed.

```diff
-{{- if $cfg.persistence.enabled }}
+{{- if or $cfg.persistence.enabled $cfg.extraVolumeMounts }}
 volumeMounts:
+  {{- if $cfg.persistence.enabled }}
   - name: ingestor-server-data
     mountPath: ...
+  {{- end }}
+  {{- if $cfg.extraVolumeMounts }}
+  {{- toYaml $cfg.extraVolumeMounts | nindent 12 }}
+  {{- end }}
 {{- end }}
-{{- if $cfg.persistence.enabled }}
+{{- if or $cfg.persistence.enabled $cfg.extraVolumes }}
 volumes:
+  {{- if $cfg.persistence.enabled }}
   - name: ingestor-server-data
     ...
+  {{- end }}
+  {{- if $cfg.extraVolumes }}
+  {{- toYaml $cfg.extraVolumes | nindent 8 }}
+  {{- end }}
 {{- end }}
```

---

### 3. `deploy/helm/values-local.yaml` *(new file)*

A `values-local.yaml` overlay for local k3s deployment. Not present upstream (upstream ships only
`values.yaml` defaults).

Key configuration in this file:

#### Elasticsearch / ECK TLS

ECK 3.x always enables TLS and creates the `elastic` superuser automatically. The upstream chart
assumes Milvus as the vector store; this file switches the stack to Elasticsearch with:

- `APP_VECTORSTORE_URL: https://rag-eck-elasticsearch-es-http:9200`
- `APP_VECTORSTORE_NAME: elasticsearch`
- `APP_VECTORSTORE_USERNAME: elastic`
- `APP_VECTORSTORE_PASSWORD: ""` — populate at deploy time (see command in comment)

**TLS trust:** `elastic_transport` ignores `APP_VECTORSTORE_VERIFY_CERTS`. TLS is established via
`SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` pointing to the ECK CA cert, which is mounted from a
pre-created ConfigMap (`eck-es-ca-cert`).

> **Note on ECK reserved settings:** Do NOT set `xpack.security.enabled` or
> `xpack.security.authc.anonymous.*` in the nodeSet config. These are reserved for internal use by
> ECK 3.x. Setting them causes Elasticsearch to start with security **disabled**.

#### Local NIM Endpoints

The upstream defaults point to NVIDIA's hosted API catalog. These vars route inference to the
local NIM services:

```yaml
APP_LLM_SERVERURL: "nim-llm:8000"
APP_EMBEDDINGS_SERVERURL: "nemoretriever-embedding-ms:8000/v1"
APP_RANKING_SERVERURL: "nemoretriever-ranking-ms:8000"
APP_QUERYREWRITER_SERVERURL: "nim-llm:8000"
APP_FILTEREXPRESSIONGENERATOR_SERVERURL: "nim-llm:8000"
REFLECTION_LLM_SERVERURL: "nim-llm:8000"
```

#### LLM NIM — Pre-cached Nemotron 49B NVFP4

The LLM NIM uses a pre-cached vLLM NVFP4 model on a local NVMe drive, avoiding a re-download
on pod restart:

- Static PV `nim-llm-cache-pv` → hostPath `/mnt/nvme4/nim_cache/nim` on `ubuntu-local-dev`
- `NIM_MODEL_PROFILE` set to the vLLM nvfp4-tp1-pp1 profile hash
- `NIM_RELAX_MEM_CONSTRAINTS=1` — bypasses the pre-launch ~88 GB free-memory gate

> **Known chart bug:** `storage.pvc.create: false` cannot be set via Helm values due to
> `| default true` in `llm-nim.yaml:19`. The NIMCache CR must be patched manually after any
> `helm upgrade` that recreates the NIMCache:
> ```bash
> kubectl patch nimcache nim-llm-cache -n rag --type=merge \
>   -p '{"spec":{"storage":{"pvc":{"create":false,"name":"nim-llm-cache-pvc"}}}}'
> ```

#### GPU Time-slicing

NIM Operator DRA is disabled (`draResources.enabled: false`) because NIM Operator 3.0.2 requires
`resource.k8s.io/v1beta2`, which was removed in Kubernetes 1.35 (promoted to `v1` GA).
Time-sliced `nvidia.com/gpu` via the device plugin is used instead.

---

## Prerequisites (cluster-side, not managed by Helm)

These must exist in the cluster before `helm upgrade --install`:

1. **Secrets** `ngc-secret` and `ngc-api` — NGC pull credentials
2. **ConfigMap** `eck-es-ca-cert` in namespace `rag`:
   ```bash
   kubectl create configmap eck-es-ca-cert -n rag \
     --from-literal=ca.crt="$(kubectl get secret rag-eck-elasticsearch-es-http-certs-public \
       -n rag -o jsonpath='{.data.ca\.crt}' | base64 -d)"
   ```
3. **PersistentVolume** `nim-llm-cache-pv` — static hostPath PV pointing to the pre-cached model
4. **PersistentVolumeClaim** `nim-llm-cache-pvc` — bound to the above PV

---

## Deploy

```bash
cd deploy/helm

# Set the Elasticsearch password from the ECK-managed secret
ES_PASS=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user -n rag \
  -o jsonpath='{.data.elastic}' | base64 -d)

helm upgrade --install rag ./nvidia-blueprint-rag \
  -n rag --create-namespace \
  -f values-local.yaml \
  --set envVars.APP_VECTORSTORE_PASSWORD="${ES_PASS}" \
  --set "ingestor-server.envVars.APP_VECTORSTORE_PASSWORD=${ES_PASS}"

# Re-apply NIMCache PVC patch (workaround for | default true bug)
kubectl patch nimcache nim-llm-cache -n rag --type=merge \
  -p '{"spec":{"storage":{"pvc":{"create":false,"name":"nim-llm-cache-pvc"}}}}'
```
