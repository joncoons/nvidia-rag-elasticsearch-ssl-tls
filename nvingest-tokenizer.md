# nv-ingest Tokenizer Alignment

**Problem**: nv-ingest's `.split()` step uses a tokenizer to count tokens at chunk boundaries. The baseline ships `intfloat/e5-large-unsupervised` as the default — a BERT-family tokenizer downloaded from HuggingFace at image build time. The actual embeddings, however, are produced by `nvidia/llama-3.2-nv-embedqa-1b-v2` (a Llama 3.2 BPE model). These tokenizers have different vocabularies and token-count semantics, so a chunk that measures 512 tokens under E5 may be a different size under the embedding model's tokenizer, causing subtle over/under-chunking at inference time.

---

## How the Tokenizer Is Used

**File**: `src/nvidia_rag/ingestor_server/nvingest.py`, line 165

```python
ingestor = ingestor.split(
    tokenizer=config.nv_ingest.tokenizer,   # ← used for chunk boundary counting only
    chunk_size=split_options.get("chunk_size", config.nv_ingest.chunk_size),
    chunk_overlap=split_options.get("chunk_overlap", config.nv_ingest.chunk_overlap),
    params={"split_source_types": split_source_types},
)
```

The tokenizer string is passed to nv-ingest's split task. It is used **only for counting tokens at chunk boundaries** — it never produces embeddings. Embeddings are produced separately by the `nemoretriever-embedding-ms` NIM.

**Important**: The split task runs inside **Ray workers in the `rag-nv-ingest` pod**, not in the `ingestor-server` pod. Any tokenizer path must be accessible from within `rag-nv-ingest`.

**File**: `src/nvidia_rag/utils/configuration.py`, line 330

```python
tokenizer: str = Field(
    default="intfloat/e5-large-unsupervised",
    env="APP_NVINGEST_TOKENIZER",
    ...
)
```

`APP_NVINGEST_TOKENIZER` overrides the default. It accepts either a HuggingFace model ID or a local filesystem path. When a local path is provided, HuggingFace's `AutoTokenizer.from_pretrained()` loads from disk — **no internet access required**.

**Note on the direct ingest path**: `direct_ingest.py` (used for HTML/web crawler content and nemoretriever-parse pre-chunked `.md` files) uses character-based chunking, not token-based. `APP_NVINGEST_TOKENIZER` has no effect on that path.

---

## Baseline Default

`intfloat/e5-large-unsupervised` is pre-downloaded into the nv-ingest image at build time via `post_build_triggers.py` and baked into the container. This is a BERT-family (WordPiece) tokenizer — its vocabulary and token counts differ from the Llama 3.x BPE tokenizer used by the embedding NIM.

---

## Local Fix: Air-Gapped NFS Tokenizer

### Why not use the NIM cache directly

The embedding NIM's tokenizer is cached by the NIM Operator at:

```
/mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--nvidia--llama-3.2-nv-embedqa-1b-v2/snapshots/tokenizer-4096-f250c002/
├── tokenizer.json           → ../../blobs/836c0dfaca...   (symlink)
├── tokenizer_config.json    → ../../blobs/60f3069e9f...   (symlink)
└── special_tokens_map.json  → ../../blobs/2884f2e038...   (symlink)
```

Two reasons this can't be used directly:

1. **Symlinks require the full tree**: The snapshot files are symlinks pointing to `../../blobs/<hash>`. Mounting only the snapshot subdirectory breaks the symlinks inside the container. Mounting the parent model directory would work, but introduces a second issue:

2. **NIM cache namespace mismatch**: The NIM Operator stores models under `models--nim--nvidia--...` (with an extra `nim` prefix). HuggingFace's cache lookup for `nvidia/llama-3.2-nv-embedqa-1b-v2` expects `models--nvidia--...`. Setting `HUGGINGFACE_HUB_CACHE` to the NIM cache directory will not find the model.

### Solution: Copy resolved files to a dedicated NFS directory

One-time setup — copy the tokenizer files with symlinks resolved to a flat NFS directory:

```bash
mkdir -p /mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2
cp -L /mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--nvidia--llama-3.2-nv-embedqa-1b-v2/snapshots/tokenizer-4096-f250c002/tokenizer.json \
   /mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2/
cp -L /mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--nvidia--llama-3.2-nv-embedqa-1b-v2/snapshots/tokenizer-4096-f250c002/tokenizer_config.json \
   /mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2/
cp -L /mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--nvidia--llama-3.2-nv-embedqa-1b-v2/snapshots/tokenizer-4096-f250c002/special_tokens_map.json \
   /mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2/
```

Result: three real files (no symlinks), ~8.8 MB total. No HuggingFace download, no internet access required.

Two snapshots exist in the NIM cache — `tokenizer-4096-f250c002` was selected as it matches `NIM_TRITON_MAX_SEQ_LENGTH=2048`.

---

## Changes Required

### 1. Add `tiktoken` to the ingestor image

**File**: `Dockerfile.ingestor-patch`

Llama 3.x uses tiktoken as its underlying BPE tokenizer. HuggingFace's `AutoTokenizer` tries the fast path first (`tokenizers` Rust library via `tokenizer.json`); if that fails it falls back requiring tiktoken. Adding `tiktoken` ensures the fallback path is available, though in practice the fast `TokenizersBackend` (Rust) loads successfully.

```dockerfile
# Before
RUN /tmp/uv pip install --python /workspace/.venv/bin/python \
      --no-cache beautifulsoup4 pdf2image lxml kubernetes selenium Pillow && rm /tmp/uv

# After
RUN /tmp/uv pip install --python /workspace/.venv/bin/python \
      --no-cache beautifulsoup4 pdf2image lxml kubernetes selenium Pillow tiktoken && rm /tmp/uv
```

### 2. Mount the tokenizer into both pods

The tokenizer must be mounted in **both** `ingestor-server` (reads the env var from config) and `rag-nv-ingest` (where the Ray split worker actually loads it).

**File**: `deploy/helm/values-local.yaml` — ingestor-server `extraVolumes`

```yaml
- name: embed-tokenizer
  hostPath:
    path: /mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2
    type: Directory
```

**File**: `deploy/helm/values-local.yaml` — ingestor-server `extraVolumeMounts`

```yaml
- name: embed-tokenizer
  mountPath: /embed-tokenizer
  readOnly: true
```

**`rag-nv-ingest` has no `extraVolumes` support in its Helm chart** and `helm upgrade` is broken for nv-ingest (OTEL env var strategic merge patch conflict). Apply directly via kubectl:

```bash
kubectl patch deployment rag-nv-ingest -n rag --type=json -p='[
  {"op":"add","path":"/spec/template/spec/volumes/-","value":{"name":"embed-tokenizer","hostPath":{"path":"/mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2","type":"Directory"}}},
  {"op":"add","path":"/spec/template/spec/containers/0/volumeMounts/-","value":{"name":"embed-tokenizer","mountPath":"/embed-tokenizer","readOnly":true}}
]'
```

This patch must be re-applied after any `helm upgrade` that recreates the `rag-nv-ingest` Deployment.

### 3. Set `APP_NVINGEST_TOKENIZER`

**File**: `deploy/helm/values-local.yaml` — ingestor-server `envVars`

```yaml
# Air-gapped Llama 3.2 BPE tokenizer — matches embedding NIM vocabulary.
# Files copied (symlinks resolved) from NIM cache to NFS flat dir.
# Same path mounted in rag-nv-ingest where Ray split workers run.
APP_NVINGEST_TOKENIZER: "/embed-tokenizer"
```

---

## Summary

| Item | Baseline | Local Deployment |
|---|---|---|
| Tokenizer used for chunking | `intfloat/e5-large-unsupervised` (BERT WordPiece) | `nvidia/llama-3.2-nv-embedqa-1b-v2` tokenizer (Llama 3 BPE) |
| Tokenizer source | Baked into nv-ingest image at build time | NFS flat directory — files copied from NIM cache, symlinks resolved |
| HuggingFace download at runtime | Yes (or pre-baked) | No — fully air-gapped |
| `tiktoken` installed in ingestor | No | Yes (added to `Dockerfile.ingestor-patch`) |
| `APP_NVINGEST_TOKENIZER` | Unset (uses default) | `/embed-tokenizer` |
| Mounted in `ingestor-server` | N/A | Yes — via `extraVolumes` in `values-local.yaml` |
| Mounted in `rag-nv-ingest` | N/A | Yes — via `kubectl patch` (no Helm support) |
| Tokenizer class loaded | BERT fast tokenizer | `TokenizersBackend` (Rust fast path), vocab size 128k |

The net effect is that chunk boundaries are counted using the same tokenizer vocabulary as the model that encodes those chunks — eliminating tokenizer mismatch at the split step.

---

## File Reference

| File | Change |
|---|---|
| `Dockerfile.ingestor-patch` | Added `tiktoken` to pip install |
| `deploy/helm/values-local.yaml` | `APP_NVINGEST_TOKENIZER=/embed-tokenizer`; `embed-tokenizer` hostPath volume + mount for ingestor-server |
| `src/nvidia_rag/utils/configuration.py` | `tokenizer` field reads `APP_NVINGEST_TOKENIZER` (baseline — no change needed) |
| `src/nvidia_rag/ingestor_server/nvingest.py` | Passes `config.nv_ingest.tokenizer` to `.split()` (baseline — no change needed) |

**NFS tokenizer path**: `/mnt/nvme4/hf_models/tokenizers/llama-3.2-nv-embedqa-1b-v2/` (3 files, ~8.8 MB, no symlinks)
