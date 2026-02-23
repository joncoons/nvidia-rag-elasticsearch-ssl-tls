<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Elasticsearch SSL/TLS Support for NVIDIA RAG Blueprint

This document describes the source-code changes made to add SSL/TLS support for Elasticsearch
connections in the NVIDIA RAG Blueprint, explains why they were necessary, and provides
step-by-step configuration instructions for both Docker Compose and Helm deployments.

:::{tip}
To navigate this page more easily, click the outline button at the top of the page.
:::


## Background and Motivation

The NVIDIA RAG Blueprint ships with a Helm chart (`deploy/helm/nvidia-blueprint-rag/`) that
uses the [Elastic Cloud on Kubernetes (ECK)](https://www.elastic.co/guide/en/cloud-on-kubernetes/current/index.html)
operator to manage Elasticsearch. ECK is capable of automatically generating TLS certificates
for HTTP and inter-node (transport) traffic, and the `values.yaml` already exposes the
relevant X-Pack security knobs (`xpack.security.http.ssl.enabled`,
`xpack.security.transport.ssl.enabled`, `selfSignedCertificate.disabled`).

However, the Python source code that backs both the **Ingestor Server** and the **RAG Server**
had no corresponding SSL/TLS support:

- No SSL configuration fields existed in `VectorStoreConfig`.
- The `Elasticsearch` Python client was always constructed without an `ssl_context`, meaning
  it would reject or silently fail on `https://` endpoints that presented certificates not
  trusted by the system CA bundle.
- The LangChain `ElasticsearchStore` used for retrieval had the same gap.

As a result, enabling TLS in the Helm chart while leaving the source code unchanged would cause
both servers to fail at startup with a connection error.

The changes described below close this gap by:

1. Adding three new environment-variable-backed configuration fields to `VectorStoreConfig`.
2. Adding a helper that builds a properly configured `ssl.SSLContext`.
3. Passing that context to both the raw `Elasticsearch` client and the LangChain
   `ElasticsearchStore`.


## Files Changed

| File | Type of change |
|---|---|
| `src/nvidia_rag/utils/configuration.py` | Added `ssl_enabled`, `ca_certs`, and `verify_certs` fields to `VectorStoreConfig` |
| `src/nvidia_rag/utils/vdb/elasticsearch/elastic_vdb.py` | Added `import ssl`, `_build_ssl_context()` static method, and SSL wiring in `__init__` and `get_langchain_vectorstore()` |


## Detailed Change Description

### 1. `src/nvidia_rag/utils/configuration.py`

**What changed:** Three fields were appended to the `VectorStoreConfig` Pydantic model,
immediately after the existing API key authentication fields.

```python
# SSL/TLS configuration for Elasticsearch
ssl_enabled: bool = Field(
    default=False,
    env="APP_VECTORSTORE_SSL_ENABLED",
    description="Enable SSL/TLS for Elasticsearch connections",
)
ca_certs: str | None = Field(
    default=None,
    env="APP_VECTORSTORE_CA_CERTS",
    description=(
        "Path to a CA bundle PEM file for verifying the Elasticsearch server certificate. "
        "Required when ssl_enabled=True and the server uses a self-signed or private CA cert "
        "(e.g., the ECK-generated CA mounted from the *-es-http-certs-public secret). "
        "Leave unset to use the system CA bundle (suitable for publicly-trusted certs)."
    ),
)
verify_certs: bool = Field(
    default=True,
    env="APP_VECTORSTORE_VERIFY_CERTS",
    description=(
        "Verify the Elasticsearch server certificate when ssl_enabled=True. "
        "Set to False only for development/testing; not recommended in production."
    ),
)
```

**Why:** `VectorStoreConfig` is the single authoritative source of configuration for all vector
store settings. Both the Ingestor Server and the RAG Server construct `NvidiaRAGConfig` (which
embeds `VectorStoreConfig`) from environment variables. Adding the fields here means both
servers pick them up automatically through the existing environment variable loading mechanism
in `_ConfigBase.__init__`, with no further plumbing required.

**New environment variables:**

| Variable | Default | Description |
|---|---|---|
| `APP_VECTORSTORE_SSL_ENABLED` | `false` | Master switch. Set to `true` to activate SSL/TLS. |
| `APP_VECTORSTORE_CA_CERTS` | *(unset)* | Absolute path to a PEM CA bundle file inside the container. Required for self-signed or private CA certificates (e.g. ECK's auto-generated CA). |
| `APP_VECTORSTORE_VERIFY_CERTS` | `true` | Set to `false` to disable certificate verification. **Use only in development.** |


### 2. `src/nvidia_rag/utils/vdb/elasticsearch/elastic_vdb.py`

Three additions were made to this file.

#### 2a. `import ssl`

`ssl` is part of the Python standard library and requires no additional dependencies.

#### 2b. `_build_ssl_context()` static method

A new static method was added to `ElasticVDB`, placed just before the `collection_name`
property. It centralises all SSL context construction logic in one place so both internal
connection sites (see 2c and 2d below) use identical behaviour.

```python
@staticmethod
def _build_ssl_context(
    ca_certs: str | None,
    verify_certs: bool,
) -> ssl.SSLContext:
    """Build an SSLContext for Elasticsearch connections.

    Args:
        ca_certs: Path to a PEM CA bundle file, or None to use the system bundle.
        verify_certs: When False the returned context disables certificate
            verification entirely (development/testing only).

    Returns:
        A configured ssl.SSLContext.
    """
    if not verify_certs:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    if ca_certs:
        return ssl.create_default_context(cafile=ca_certs)

    # Fall back to the system CA bundle (correct for publicly-trusted certs).
    return ssl.create_default_context()
```

The method handles three scenarios in priority order:

| Scenario | `verify_certs` | `ca_certs` | Behaviour |
|---|---|---|---|
| Skip verification (dev/test) | `false` | any | Disables hostname check and cert validation. **Do not use in production.** |
| Private / self-signed CA | `true` | set | Trusts only the specified CA bundle (e.g. the PEM file from ECK's `*-es-http-certs-public` secret). |
| Public CA (default) | `true` | unset | Uses the system CA bundle. Suitable for certificates signed by a well-known CA. |

#### 2c. SSL context wiring in `__init__`

The block that builds `es_conn_params` was extended to conditionally add an `ssl_context` key
before the `Elasticsearch()` constructor is called:

```python
if self.config.vector_store.ssl_enabled:
    es_conn_params["ssl_context"] = self._build_ssl_context(
        ca_certs=self.config.vector_store.ca_certs,
        verify_certs=self.config.vector_store.verify_certs,
    )

self._es_connection = Elasticsearch(**es_conn_params).options(
    request_timeout=int(os.environ.get("ES_REQUEST_TIMEOUT", 600))
)
```

This covers all operations that go through `self._es_connection`: index creation, document
ingestion, document deletion, metadata schema management, document info management, and health
checks.

#### 2d. SSL context wiring in `get_langchain_vectorstore()`

The same context is injected into `ElasticsearchStore` via its `es_params` keyword argument.
Existing auth entries in `es_params` (e.g. bearer token) are preserved via dict unpacking:

```python
if self.config.vector_store.ssl_enabled:
    ssl_ctx = self._build_ssl_context(
        ca_certs=self.config.vector_store.ca_certs,
        verify_certs=self.config.vector_store.verify_certs,
    )
    vectorstore_params["es_params"] = {
        **vectorstore_params.get("es_params", {}),
        "ssl_context": ssl_ctx,
    }
```

This covers all retrieval operations performed through `retrieval_langchain()`.

**Why both sites need the context:**
The raw `Elasticsearch` client (`self._es_connection`) and the LangChain `ElasticsearchStore`
each create their own internal HTTP connections. Providing the context only to one of them
would leave the other attempting plain or unverified TLS connections.


## Unchanged Files

No changes were required to `src/nvidia_rag/utils/vdb/__init__.py` (the VDB factory). The
factory already passes `config=config` through to `ElasticVDB.__init__`, so all three new
config fields are available to the constructor automatically.


## Configuration Guide

### Docker Compose

The Docker Compose deployment does not include ECK, so you must bring your own Elasticsearch
instance with TLS configured. The steps below assume Elasticsearch is already running with a
CA-signed or self-signed certificate.

1. **Obtain the CA certificate.** Export the CA bundle that signed your Elasticsearch server
   certificate as a PEM file and place it somewhere accessible to the containers, for example
   `deploy/compose/certs/es-ca.crt`.

2. **Mount the certificate into the containers.** Edit
   `deploy/compose/docker-compose-rag-server.yaml` and
   `deploy/compose/docker-compose-ingestor-server.yaml` to add a volume mount:

   ```yaml
   volumes:
     - ./certs/es-ca.crt:/etc/es-certs/ca.crt:ro
   ```

3. **Set environment variables.** Add the following to each compose file's `environment`
   section (or export them in your shell before running `docker compose up`):

   ```bash
   export APP_VECTORSTORE_URL="https://elasticsearch:9200"
   export APP_VECTORSTORE_NAME="elasticsearch"
   export APP_VECTORSTORE_SSL_ENABLED="true"
   export APP_VECTORSTORE_CA_CERTS="/etc/es-certs/ca.crt"
   export APP_VECTORSTORE_VERIFY_CERTS="true"
   ```

   If your Elasticsearch instance uses a certificate signed by a public CA (e.g. Let's Encrypt),
   omit `APP_VECTORSTORE_CA_CERTS` and the system bundle will be used automatically.

4. **Start the services:**

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

5. **Verify connectivity:**

   ```bash
   # From inside a running container:
   curl --cacert /etc/es-certs/ca.crt https://elasticsearch:9200/_cluster/health
   ```

:::{note}
Authentication (username/password or API key) is independent of SSL/TLS and works the same way
as documented in [Elasticsearch Authentication](change-vectordb.md#elasticsearch-authentication).
Both can be used together.
:::


### Helm (ECK)

The Helm chart uses ECK to manage Elasticsearch. ECK can automatically generate and rotate
TLS certificates. The steps below enable ECK's self-signed CA and configure the application
pods to trust it.

#### Step 1 — Enable TLS in `values.yaml`

```yaml
eck-elasticsearch:
  enabled: true
  http:
    tls:
      selfSignedCertificate:
        disabled: false        # ECK generates and manages the CA and server certificate
  nodeSets:
  - name: default
    count: 1
    config:
      node.store.allow_mmap: false
      xpack.security.enabled: true
      xpack.security.http.ssl.enabled: true
      xpack.security.transport.ssl.enabled: true
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          readinessProbe:
            exec:
              command:
              - bash
              - -c
              - /mnt/elastic-internal/scripts/readiness-port-script.sh
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 5
            failureThreshold: 3
```

:::{important}
When `xpack.security.enabled: true` is set, replace the default readiness probe with the ECK
built-in script as shown above. The default probe uses an unauthenticated `curl` command that
will fail once security is enabled.
:::

#### Step 2 — Mount the ECK CA certificate into application pods

ECK automatically creates a Kubernetes Secret named
`<release>-eck-elasticsearch-es-http-certs-public` containing `ca.crt` (the public CA
certificate in PEM format). Add a volume and volumeMount to both the RAG Server and Ingestor
Server sections of `values.yaml`:

```yaml
# RAG Server
extraVolumes:
  - name: es-certs
    secret:
      secretName: rag-eck-elasticsearch-es-http-certs-public

extraVolumeMounts:
  - name: es-certs
    mountPath: /etc/es-certs
    readOnly: true

# Ingestor Server
ingestor-server:
  extraVolumes:
    - name: es-certs
      secret:
        secretName: rag-eck-elasticsearch-es-http-certs-public

  extraVolumeMounts:
    - name: es-certs
      mountPath: /etc/es-certs
      readOnly: true
```

:::{note}
The secret name prefix (`rag-`) matches the Helm release name used in the deployment command
(`helm install rag ...`). Adjust the prefix if your release name differs.
:::

#### Step 3 — Set environment variables in `values.yaml`

Update the `envVars` sections for both the RAG Server and Ingestor Server:

```yaml
# RAG Server
envVars:
  APP_VECTORSTORE_URL: "https://rag-eck-elasticsearch-es-http:9200"
  APP_VECTORSTORE_NAME: "elasticsearch"
  APP_VECTORSTORE_SSL_ENABLED: "true"
  APP_VECTORSTORE_CA_CERTS: "/etc/es-certs/ca.crt"
  APP_VECTORSTORE_VERIFY_CERTS: "true"
  # Authentication — retrieve the ECK-generated password (see Step 4)
  APP_VECTORSTORE_USERNAME: "elastic"
  APP_VECTORSTORE_PASSWORD: ""      # fill in after retrieving from secret

# Ingestor Server
ingestor-server:
  envVars:
    APP_VECTORSTORE_URL: "https://rag-eck-elasticsearch-es-http:9200"
    APP_VECTORSTORE_NAME: "elasticsearch"
    APP_VECTORSTORE_SSL_ENABLED: "true"
    APP_VECTORSTORE_CA_CERTS: "/etc/es-certs/ca.crt"
    APP_VECTORSTORE_VERIFY_CERTS: "true"
    APP_VECTORSTORE_USERNAME: "elastic"
    APP_VECTORSTORE_PASSWORD: ""
```

#### Step 4 — Retrieve the ECK-generated password

When `xpack.security.enabled: true`, ECK creates a secret containing the `elastic` superuser
password:

```bash
ES_PASSWORD=$(kubectl get secret rag-eck-elasticsearch-es-elastic-user \
  -n rag -o jsonpath='{.data.elastic}' | base64 -d)
echo "Elasticsearch password: $ES_PASSWORD"
```

Set `APP_VECTORSTORE_PASSWORD` to this value in `values.yaml`, or pass it on the Helm command
line:

```bash
helm upgrade --install rag -n rag deploy/helm/nvidia-blueprint-rag/ \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set "envVars.APP_VECTORSTORE_PASSWORD=$ES_PASSWORD" \
  --set "ingestor-server.envVars.APP_VECTORSTORE_PASSWORD=$ES_PASSWORD" \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

#### Step 5 — Apply the changes

```bash
helm upgrade --install rag -n rag deploy/helm/nvidia-blueprint-rag/ \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml
```

#### Step 6 — Verify

```bash
# Check that pods are running
kubectl get pods -n rag

# Check ingestor-server logs for a successful Elasticsearch connection
kubectl logs -n rag -l app=ingestor-server --tail=30

# Test the HTTPS endpoint from inside the cluster
kubectl exec -n rag rag-eck-elasticsearch-es-default-0 -- \
  curl -s --cacert /usr/share/elasticsearch/config/http-certs/ca.crt \
  -u elastic:$ES_PASSWORD \
  https://localhost:9200/_cluster/health
```

A successful response looks like:

```json
{"cluster_name":"rag-eck-elasticsearch","status":"yellow","..."}
```


## Environment Variable Reference

The following table summarises all environment variables relevant to Elasticsearch SSL/TLS.
Variables marked **new** were added as part of this change.

| Variable | New | Default | Description |
|---|---|---|---|
| `APP_VECTORSTORE_URL` | | `http://localhost:19530` | Full URL of the Elasticsearch endpoint. Use `https://` when SSL is enabled. |
| `APP_VECTORSTORE_NAME` | | `milvus` | Set to `elasticsearch` to select the Elasticsearch backend. |
| `APP_VECTORSTORE_SSL_ENABLED` | **yes** | `false` | Master SSL/TLS switch. |
| `APP_VECTORSTORE_CA_CERTS` | **yes** | *(unset)* | Absolute path inside the container to a PEM CA bundle. Required for self-signed or private CA certs. |
| `APP_VECTORSTORE_VERIFY_CERTS` | **yes** | `true` | Set to `false` to skip certificate verification. Development/testing only. |
| `APP_VECTORSTORE_USERNAME` | | `""` | Elasticsearch username (used when `xpack.security.enabled: true`). |
| `APP_VECTORSTORE_PASSWORD` | | `""` | Elasticsearch password. |
| `APP_VECTORSTORE_APIKEY` | | `""` | Base64-encoded API key (`id:secret`). Takes precedence over username/password. |
| `APP_VECTORSTORE_APIKEY_ID` | | `""` | API key ID (used with `APP_VECTORSTORE_APIKEY_SECRET`). |
| `APP_VECTORSTORE_APIKEY_SECRET` | | `""` | API key secret. |
| `ES_REQUEST_TIMEOUT` | | `600` | Request timeout in seconds for all Elasticsearch operations. |


## Authentication and SSL Interaction

SSL/TLS and authentication are orthogonal features. The following combinations are all valid:

| SSL | Auth | Use case |
|---|---|---|
| Off | Off | Local development with plain HTTP and no security |
| Off | On | Authenticated plain HTTP (not recommended for production) |
| On | Off | Encrypted but unauthenticated (unusual; requires `xpack.security.enabled: false` with HTTP TLS) |
| On | On | **Recommended for production** — encrypted and authenticated |

Authentication priority (highest to lowest) is unchanged from the baseline:
bearer token → API key → username/password.


## Troubleshooting

**Connection refused or SSL handshake error on startup**

- Confirm `APP_VECTORSTORE_URL` starts with `https://` (not `http://`) when SSL is enabled.
- Confirm `APP_VECTORSTORE_SSL_ENABLED` is set to `true`.
- Verify the CA certificate file exists at the path specified in `APP_VECTORSTORE_CA_CERTS`.

**`certificate verify failed` error**

- The certificate presented by Elasticsearch is not trusted by the CA bundle at `APP_VECTORSTORE_CA_CERTS`.
- In ECK deployments, ensure the secret `<release>-eck-elasticsearch-es-http-certs-public` is
  mounted correctly and that the `ca.crt` within it matches the serving certificate.
- Run `openssl s_client -connect <es-host>:9200 -CAfile /etc/es-certs/ca.crt` from inside a
  pod to diagnose certificate chain issues.

**ECK pod stuck in `Pending` or `CrashLoopBackOff`**

- If you enabled `xpack.security.enabled: true` but did not update the readiness probe, the
  pod will fail its health check. Replace the probe as shown in Step 1 above.

**`urllib3` `InsecureRequestWarning` in logs**

- This appears when `APP_VECTORSTORE_VERIFY_CERTS=false`. Switch to a proper CA bundle for
  production workloads.


## Related Documentation

- [Configure Elasticsearch as Your Vector Database](change-vectordb.md)
- [Deploy the RAG Pipeline with Helm](deploy-helm.md)
- [Elasticsearch Authentication](change-vectordb.md#elasticsearch-authentication)
- [Troubleshooting](troubleshooting.md)
- [ECK TLS Configuration Reference](https://www.elastic.co/guide/en/cloud-on-kubernetes/current/k8s-tls-certificates.html)
