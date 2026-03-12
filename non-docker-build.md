# Building and Deploying Patched Images Without Docker

This document covers how to perform container builds and deployments in a Kubernetes-only environment where the Docker CLI and daemon are not available. It is written in the context of the NVIDIA RAG Blueprint local deployment but applies generally to any k8s workload using patched/overlaid images.

---

## Background: Current Workflow

The local deployment builds three patched images using Docker:

```
docker build -f Dockerfile.<name>-patch -t <image>:latest .
docker save -o /tmp/<image>.tar <image>:latest
sudo k3s ctr images import /tmp/<image>.tar
kubectl delete pod -n rag -l app=<name>
```

Each Dockerfile copies a base NVIDIA image and overlays changed Python source files plus any additional pip packages. The sections below describe how to replicate this without Docker.

---

## Option 1: Kaniko + Local Registry (Full Docker Replacement)

[Kaniko](https://github.com/GoogleContainerTools/kaniko) builds OCI images from a Dockerfile inside a Kubernetes pod, with no Docker daemon required. This is the closest full replacement for `docker build`.

### 1.1 Deploy a Local Container Registry

Deploy a registry inside the cluster once. Images are stored on an NFS PVC so they survive pod restarts.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: registry
  namespace: rag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: registry
  template:
    metadata:
      labels:
        app: registry
    spec:
      containers:
      - name: registry
        image: registry:2
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: data
          mountPath: /var/lib/registry
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: registry-data
---
apiVersion: v1
kind: Service
metadata:
  name: registry
  namespace: rag
spec:
  selector:
    app: registry
  ports:
  - port: 5000
    targetPort: 5000
```

### 1.2 Configure k3s to Trust the Registry

On each k3s node, create or update `/etc/rancher/k3s/registries.yaml`:

```yaml
mirrors:
  "registry.rag.svc.cluster.local:5000":
    endpoint:
      - "http://registry.rag.svc.cluster.local:5000"
```

Restart k3s on each node: `sudo systemctl restart k3s` (or `k3s-agent` on worker nodes).

### 1.3 Build with a Kaniko Pod

Kaniko reads the Dockerfile and build context from a mounted path and pushes the result to the registry. The source tree on NFS can be mounted directly as a hostPath — no file transfer needed.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kaniko-ingestor-build
  namespace: rag
spec:
  restartPolicy: Never
  containers:
  - name: kaniko
    image: gcr.io/kaniko-project/executor:latest
    args:
    - "--dockerfile=/workspace/Dockerfile.ingestor-patch"
    - "--context=dir:///workspace"
    - "--destination=registry.rag.svc.cluster.local:5000/rag-ingestor-local:latest"
    - "--insecure"
    - "--cache=true"
    - "--cache-repo=registry.rag.svc.cluster.local:5000/cache"
    volumeMounts:
    - name: workspace
      mountPath: /workspace
  volumes:
  - name: workspace
    hostPath:
      path: /home/joncoons/claude/rag
      type: Directory
  nodeSelector:
    kubernetes.io/hostname: ubuntu-local-dev
```

Apply and wait:
```bash
kubectl apply -f kaniko-ingestor-build.yaml
kubectl wait --for=condition=complete pod/kaniko-ingestor-build -n rag --timeout=600s
kubectl logs -n rag kaniko-ingestor-build
kubectl delete pod -n rag kaniko-ingestor-build
```

### 1.4 Update the Deployment Image

After a successful Kaniko build, update the deployment to pull from the local registry:

```bash
kubectl set image deployment/ingestor-server -n rag \
  ingestor-server=registry.rag.svc.cluster.local:5000/rag-ingestor-local:latest
```

Or update `values-local.yaml` to reference the registry image and apply via `kubectl patch`.

---

## Option 2: Python Source Overlays Without Rebuilding

For changes to `.py` files only — no new pip packages, no system dependencies — the image rebuild can be skipped entirely. The patched Dockerfiles use a thin overlay pattern (`COPY src/... /workspace/.venv/.../`) that can be replicated at runtime.

### 2.1 `kubectl cp` (Immediate, Ephemeral)

Copy changed files directly into the running pod:

```bash
POD=$(kubectl get pod -n rag -l app=ingestor-server -o name | head -1)

kubectl cp src/nvidia_rag/utils/web_crawler.py -n rag \
  ${POD}:/workspace/.venv/lib/python3.13/site-packages/nvidia_rag/utils/web_crawler.py

kubectl cp src/nvidia_rag/ingestor_server/server.py -n rag \
  ${POD}:/workspace/.venv/lib/python3.13/site-packages/nvidia_rag/ingestor_server/server.py
```

Restart the process (FastAPI reloads on SIGTERM):
```bash
kubectl rollout restart deployment/ingestor-server -n rag
```

**Limitation**: Changes are lost on pod replacement because the base image is unchanged. Use for development iteration only.

### 2.2 ConfigMap Overlay via Init Container (Durable)

Store source files in a ConfigMap and write them to the correct paths at pod startup via an init container. Changes survive pod restarts and are version-controlled in the cluster.

**Create the ConfigMap** from source files:
```bash
kubectl create configmap ingestor-overlays -n rag \
  --from-file=web_crawler.py=src/nvidia_rag/utils/web_crawler.py \
  --from-file=server.py=src/nvidia_rag/ingestor_server/server.py \
  --from-file=main.py=src/nvidia_rag/ingestor_server/main.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Add an init container** to the Deployment:
```yaml
initContainers:
- name: overlay
  image: busybox
  command:
  - sh
  - -c
  - |
    cp /overlays/web_crawler.py /workspace/.venv/lib/python3.13/site-packages/nvidia_rag/utils/
    cp /overlays/server.py      /workspace/.venv/lib/python3.13/site-packages/nvidia_rag/ingestor_server/
    cp /overlays/main.py        /workspace/.venv/lib/python3.13/site-packages/nvidia_rag/ingestor_server/
  volumeMounts:
  - name: overlays
    mountPath: /overlays
  - name: site-packages
    mountPath: /workspace/.venv/lib/python3.13/site-packages
volumes:
- name: overlays
  configMap:
    name: ingestor-overlays
```

Apply via `kubectl patch` on the Deployment, then `kubectl rollout restart`.

**Update cycle** for a source change:
```bash
kubectl create configmap ingestor-overlays -n rag \
  --from-file=web_crawler.py=src/nvidia_rag/utils/web_crawler.py \
  ... \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/ingestor-server -n rag
```

**Limitation**: ConfigMap keys have a 1 MB size limit. Suitable for `.py` source files; not for binary data or large assets.

---

## Option 3: BuildKit Daemon in Cluster

[BuildKit](https://github.com/moby/buildkit) can run as a daemonset and expose a gRPC build endpoint. Client tools (`buildctl`, `nerdctl build`) connect to it remotely.

```bash
# Deploy buildkitd as a daemonset (rootless)
kubectl apply -f https://raw.githubusercontent.com/moby/buildkit/master/examples/kubernetes/daemonset.rootless.yaml

# Build and push to local registry
buildctl --addr tcp://buildkitd.kube-system.svc.cluster.local:1234 build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=. \
  --opt filename=Dockerfile.ingestor-patch \
  --output type=image,name=registry.rag.svc.cluster.local:5000/rag-ingestor-local:latest,push=true,registry.insecure=true
```

This is more complex to set up than Kaniko but allows repeated builds without spawning a new pod each time.

---

## Decision Guide

| Change type | Recommended approach |
|---|---|
| Python source file only — dev iteration | `kubectl cp` + `rollout restart` |
| Python source file — durable | ConfigMap overlay via init container |
| New pip package or system dependency | Kaniko pod + local registry |
| Full image rebuild | Kaniko pod + local registry |
| Frequent rebuilds (CI/CD) | BuildKit daemonset + local registry |

---

## Applying to This Deployment

The three patched images and their primary change types:

| Image | Dockerfile | Typical change | Approach |
|---|---|---|---|
| `rag-ingestor-local` | `Dockerfile.ingestor-patch` | Python overlays | ConfigMap init container for `.py`; Kaniko for new packages |
| `rag-server-local` | `Dockerfile.rag-server-patch` | Python overlays | ConfigMap init container |
| `rag-frontend-local` | `Dockerfile.frontend-patch` | `npm run build` output | Kaniko (binary `dist/` assets can't go in ConfigMap) |

The frontend image is the only one that always requires a full build because the compiled `dist/` output is binary and too large for a ConfigMap. Kaniko or BuildKit are required for frontend changes.
