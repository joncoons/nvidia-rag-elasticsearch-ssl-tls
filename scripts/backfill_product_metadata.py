#!/usr/bin/env python3
"""
Backfill product_family / product_name on existing ES chunks using content_url.

For each chunk that has a content_url set (populated by backfill_content_url.py
or by a recent crawl), this script applies the same URL→product mapping used by
SimpleWebCrawler._resolve_product_metadata() and bulk-updates the two fields.

Chunks with no content_url are skipped (they cannot be mapped to a product).
Chunks that already have both fields set are skipped unless --force is passed.

Usage (inside ingestor pod or any pod with ES access):

    python3 backfill_product_metadata.py \\
        --index nvidia_docs \\
        --es-url https://rag-eck-elasticsearch-es-http:9200 \\
        --es-user elastic \\
        --es-password <password> \\
        --ca-cert /etc/ssl/eck/ca.crt \\
        [--dry-run] \\
        [--force]

Run from the host (requires port-forward to ES or ingestor):
    kubectl exec -n rag deploy/ingestor-server -- \\
        python3 /path/to/backfill_product_metadata.py --index nvidia_docs ...
"""
import argparse
import os
import ssl
import sys

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


# ---------------------------------------------------------------------------
# Default URL → product mapping (mirrors configuration.CRAWLER_PRODUCT_URL_MAP).
# Running inside the ingestor pod, we can import directly.
# ---------------------------------------------------------------------------
def _load_product_map(override_path: str | None) -> list[tuple[str, str, str | None]]:
    if override_path:
        import json
        try:
            with open(override_path) as f:
                return [(e[0], e[1], e[2] if len(e) > 2 else None) for e in json.load(f)]
        except Exception as e:
            print(f"WARNING: could not load {override_path}: {e}", file=sys.stderr)

    try:
        from nvidia_rag.utils.configuration import CRAWLER_PRODUCT_URL_MAP
        return CRAWLER_PRODUCT_URL_MAP
    except ImportError:
        print("WARNING: nvidia_rag not importable — using empty map", file=sys.stderr)
        return []


def resolve_product(url: str, url_map: list[tuple[str, str, str | None]]) -> dict:
    """Return product_family / product_name for *url*, or empty dict."""
    lower = url.lower()
    for prefix, family, name in url_map:
        if prefix.lower() in lower:
            meta: dict = {"product_family": family}
            if name is not None:
                meta["product_name"] = name
            return meta
    return {}


# ---------------------------------------------------------------------------
# ES scan helpers
# ---------------------------------------------------------------------------
def iter_all_docs(es: Elasticsearch, index: str, batch_size: int = 500):
    """Yield (doc_id, content_url, existing_family, existing_name) for all docs."""
    pit = es.open_point_in_time(index=index, keep_alive="5m")
    pit_id = pit["id"]
    search_after = None

    try:
        while True:
            body: dict = {
                "size": batch_size,
                "query": {"match_all": {}},
                "_source": [
                    "metadata.content_metadata.content_url",
                    "metadata.content_metadata.product_family",
                    "metadata.content_metadata.product_name",
                ],
                "sort": [{"_shard_doc": "asc"}],
                "pit": {"id": pit_id, "keep_alive": "5m"},
            }
            if search_after:
                body["search_after"] = search_after

            resp = es.search(body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                cm = hit["_source"].get("metadata", {}).get("content_metadata", {})
                yield (
                    hit["_id"],
                    cm.get("content_url") or "",
                    cm.get("product_family") or "",
                    cm.get("product_name") or "",
                )

            search_after = hits[-1]["sort"]
    finally:
        try:
            es.close_point_in_time(body={"id": pit_id})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Collection schema update
# ---------------------------------------------------------------------------
def ensure_collection_schema(ingestor_url: str, collection_name: str) -> None:
    """Add product_family and product_name fields to the collection schema if missing."""
    import requests as req  # noqa: PLC0415
    try:
        resp = req.get(f"{ingestor_url}/v1/collections", timeout=10)
        resp.raise_for_status()
        collections = resp.json().get("collections", [])
        col = next((c for c in collections if c["collection_name"] == collection_name), None)
        if col is None:
            print(f"WARNING: collection '{collection_name}' not found — skipping schema update")
            return

        existing_fields = {f["name"] for f in col.get("metadata_schema", [])}
        new_fields = []
        if "product_family" not in existing_fields:
            new_fields.append({
                "name": "product_family",
                "type": "string",
                "required": False,
                "max_length": 128,
                "description": "High-level product family derived from source URL (e.g. CUDA, TensorRT, Jetson)",
            })
        if "product_name" not in existing_fields:
            new_fields.append({
                "name": "product_name",
                "type": "string",
                "required": False,
                "max_length": 256,
                "description": "Specific product name derived from source URL (e.g. CUDA Toolkit, TensorRT)",
            })

        if not new_fields:
            print("Collection schema already has product_family and product_name — no update needed.")
            return

        patch_resp = req.patch(
            f"{ingestor_url}/v1/collection",
            json={
                "collection_name": collection_name,
                "metadata_schema": col.get("metadata_schema", []) + new_fields,
            },
            timeout=10,
        )
        if patch_resp.ok:
            print(f"Collection schema updated — added: {[f['name'] for f in new_fields]}")
        else:
            print(f"WARNING: schema update failed {patch_resp.status_code}: {patch_resp.text[:200]}")
    except Exception as e:
        print(f"WARNING: could not update collection schema: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Backfill product_family/product_name in ES from content_url")
    ap.add_argument("--index", required=True, help="ES index name (== collection name)")
    ap.add_argument("--es-url", default="https://rag-eck-elasticsearch-es-http:9200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", required=True)
    ap.add_argument("--ca-cert", default="/etc/ssl/eck/ca.crt")
    ap.add_argument("--dry-run", action="store_true", help="Print matches without writing to ES")
    ap.add_argument("--force", action="store_true", help="Overwrite chunks that already have product fields set")
    ap.add_argument("--batch-size", type=int, default=200, help="Bulk update batch size")
    ap.add_argument("--product-map", default=None, help="Path to JSON override for URL→product map")
    ap.add_argument("--ingestor-url", default="http://localhost:18084",
                    help="Ingestor base URL for schema update (default: localhost port-forward)")
    ap.add_argument("--skip-schema-update", action="store_true",
                    help="Skip collection schema update (useful if ingestor not reachable)")
    args = ap.parse_args()

    url_map = _load_product_map(args.product_map)
    print(f"Loaded {len(url_map)} URL→product mapping entries")

    # Connect to ES
    ssl_ctx = ssl.create_default_context(cafile=args.ca_cert)
    es = Elasticsearch(
        hosts=[args.es_url],
        basic_auth=(args.es_user, args.es_password),
        ssl_context=ssl_ctx,
        request_timeout=120,
    )
    es.info()
    print(f"Connected to Elasticsearch at {args.es_url}")

    # Update collection schema
    if not args.dry_run and not args.skip_schema_update:
        ensure_collection_schema(args.ingestor_url, args.index)

    # Scan and build updates
    total = 0
    no_url = 0
    already_set = 0
    no_match = 0
    matched = 0
    actions = []

    print(f"Scanning index '{args.index}'...")
    for doc_id, content_url, existing_family, existing_name in iter_all_docs(es, args.index):
        total += 1

        if not content_url:
            no_url += 1
            continue

        if existing_family and not args.force:
            already_set += 1
            continue

        product = resolve_product(content_url, url_map)
        if not product:
            no_match += 1
            continue

        matched += 1
        if args.dry_run:
            print(f"  WOULD UPDATE {doc_id}: url={content_url!r} → {product}")
        else:
            actions.append({
                "_op_type": "update",
                "_index": args.index,
                "_id": doc_id,
                "doc": {"metadata": {"content_metadata": product}},
            })

        if not args.dry_run and len(actions) >= args.batch_size:
            success, errors = bulk(es, actions, raise_on_error=False)
            if errors:
                print(f"  Bulk errors: {errors[:3]}", file=sys.stderr)
            actions.clear()

        if total % 10000 == 0:
            print(f"  Scanned {total:,} | matched {matched:,} | no_url {no_url:,} | "
                  f"already_set {already_set:,} | no_match {no_match:,}")

    # Flush remainder
    if not args.dry_run and actions:
        success, errors = bulk(es, actions, raise_on_error=False)
        if errors:
            print(f"  Bulk errors: {errors[:3]}", file=sys.stderr)

    print(
        f"\nDone."
        f"\n  total scanned : {total:,}"
        f"\n  updated       : {matched:,}"
        f"\n  already set   : {already_set:,}"
        f"\n  no content_url: {no_url:,}"
        f"\n  no map match  : {no_match:,}"
    )

    if not args.dry_run and matched:
        es.indices.refresh(index=args.index)
        print(f"Index '{args.index}' refreshed.")


if __name__ == "__main__":
    main()
