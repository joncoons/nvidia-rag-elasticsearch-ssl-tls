#!/usr/bin/env python3
"""
Backfill content_url on existing ES chunks using the URL registry.

The URL registry stores url → [chunk_doc_names] (basenames of temp/real files).
ES stores source_name as the full path, e.g. /tmp/webcrawl_xxxx.md or the real filename.
This script builds a basename→url lookup, scans all ES docs in the collection,
and bulk-updates content_url for any doc whose source_name basename is in the lookup.

Usage (inside ingestor pod or any pod with ES access):
    python3 backfill_content_url.py \
        --registry /crawl-exports/nvidia_url_registry.json \
        --index nvidia \
        --es-url https://rag-eck-elasticsearch-es-http:9200 \
        --es-user elastic \
        --es-password <password> \
        --ca-cert /etc/ssl/eck/ca.crt \
        [--dry-run]
"""
import argparse
import json
import os
import ssl
import sys
from collections import defaultdict

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def build_basename_to_url(registry_path: str) -> dict[str, str]:
    """Build basename → url lookup from registry."""
    with open(registry_path) as f:
        registry = json.load(f)

    lookup: dict[str, str] = {}
    for url, entry in registry.items():
        for chunk_doc_name in entry.get("chunk_doc_names") or []:
            basename = os.path.basename(chunk_doc_name)
            if basename in lookup and lookup[basename] != url:
                print(f"WARNING: basename collision {basename!r}: {lookup[basename]} vs {url}", file=sys.stderr)
            lookup[basename] = url

    return lookup


def iter_all_docs(es: Elasticsearch, index: str, batch_size: int = 500):
    """Scroll through all docs in index, yielding (doc_id, source_name, current_content_url)."""
    pit = es.open_point_in_time(index=index, keep_alive="5m")
    pit_id = pit["id"]
    search_after = None

    try:
        while True:
            body: dict = {
                "size": batch_size,
                "query": {"match_all": {}},
                "_source": ["metadata.source.source_name", "metadata.content_metadata.content_url"],
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
                m = hit["_source"].get("metadata", {})
                source_name = m.get("source", {}).get("source_name", "") or ""
                content_url = m.get("content_metadata", {}).get("content_url", "") or ""
                yield hit["_id"], source_name, content_url

            search_after = hits[-1]["sort"]
    finally:
        try:
            es.close_point_in_time(body={"id": pit_id})
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Backfill content_url in ES from URL registry")
    ap.add_argument("--registry", required=True, help="Path to url_registry.json")
    ap.add_argument("--index", required=True, help="ES index name (collection)")
    ap.add_argument("--es-url", default="https://rag-eck-elasticsearch-es-http:9200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", required=True)
    ap.add_argument("--ca-cert", default="/etc/ssl/eck/ca.crt")
    ap.add_argument("--dry-run", action="store_true", help="Print matches without writing")
    ap.add_argument("--batch-size", type=int, default=200, help="Bulk update batch size")
    args = ap.parse_args()

    # Build lookup
    print(f"Loading registry from {args.registry}...")
    basename_to_url = build_basename_to_url(args.registry)
    print(f"Loaded {len(basename_to_url)} basename→url mappings")

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

    # Scan docs and build update actions
    total = 0
    matched = 0
    already_set = 0
    actions = []

    print(f"Scanning index '{args.index}'...")
    for doc_id, source_name, current_url in iter_all_docs(es, args.index):
        total += 1
        basename = os.path.basename(source_name)
        url = basename_to_url.get(basename)
        if url:
            if current_url == url:
                already_set += 1
            else:
                matched += 1
                if args.dry_run:
                    print(f"  WOULD UPDATE {doc_id}: source={basename!r} → url={url!r}")
                else:
                    actions.append({
                        "_op_type": "update",
                        "_index": args.index,
                        "_id": doc_id,
                        "doc": {"metadata": {"content_metadata": {"content_url": url}}},
                    })

                # Flush batch
                if not args.dry_run and len(actions) >= args.batch_size:
                    success, errors = bulk(es, actions, raise_on_error=False)
                    if errors:
                        print(f"  Bulk errors: {errors[:3]}", file=sys.stderr)
                    actions.clear()

        if total % 500 == 0:
            print(f"  Scanned {total} docs, matched {matched}, already_set {already_set}...")

    # Flush remaining
    if not args.dry_run and actions:
        success, errors = bulk(es, actions, raise_on_error=False)
        if errors:
            print(f"  Bulk errors: {errors[:3]}", file=sys.stderr)

    print(f"\nDone. total={total} matched={matched} already_set={already_set} skipped={total - matched - already_set}")
    if not args.dry_run and matched:
        es.indices.refresh(index=args.index)
        print(f"Index '{args.index}' refreshed.")


if __name__ == "__main__":
    main()
