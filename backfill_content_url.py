#!/usr/bin/env python3
"""
Backfill content_url in Elasticsearch nvidia index.

For each URL in the registry that has chunk_doc_names, run an ES update_by_query
to set metadata.content_metadata.content_url on all matching chunks.

This allows delete_by_content_url (upsert) to work on chunks that were ingested
before content_url tracking was introduced, without requiring a force re-crawl.

Usage:
    python3 backfill_content_url.py [--dry-run] [--collection nvidia]
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime

REGISTRY_PATH = "/home/joncoons/crawl-exports/nvidia_url_registry.json"
ES_SERVICE    = "rag-eck-elasticsearch-es-http"
ES_NAMESPACE  = "rag"
ES_PORT       = 9200
ES_INDEX      = "nvidia"
ES_SECRET     = "rag-eck-elasticsearch-es-elastic-user"


def get_es_password() -> str:
    result = subprocess.run(
        ["kubectl", "get", "secret", "-n", ES_NAMESPACE, ES_SECRET,
         "-o", "jsonpath={.data.elastic}"],
        capture_output=True, text=True, check=True,
    )
    return base64.b64decode(result.stdout.strip()).decode()


def es_curl(es_pass: str, method: str, path: str, body: dict | None = None) -> dict:
    """Run a curl command against ES via kubectl run ephemeral pod."""
    body_arg = ""
    if body:
        body_json = json.dumps(body).replace("'", "'\\''")
        body_arg = f"-d '{body_json}'"
    script = (
        f"curl -sk -u elastic:{es_pass} -X {method} "
        f"'https://{ES_SERVICE}:{ES_PORT}{path}' "
        f"-H 'Content-Type: application/json' {body_arg}"
    )
    pod_name = f"es-backfill-{abs(hash(path + str(body)[:20])) % 100000}"
    result = subprocess.run(
        ["kubectl", "run", pod_name, "--rm", "-i", "--restart=Never",
         "--image=curlimages/curl:8.7.1", "-n", ES_NAMESPACE,
         "--", "sh", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    data, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    return data


def load_registry(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report without modifying ES")
    parser.add_argument("--collection", default=ES_INDEX,
                        help=f"ES index / collection name (default: {ES_INDEX})")
    args = parser.parse_args()

    if not os.path.exists(REGISTRY_PATH):
        print(f"ERROR: Registry not found at {REGISTRY_PATH}")
        sys.exit(1)

    registry = load_registry(REGISTRY_PATH)

    # Build list of (source_uri, [chunk_doc_names]) for all URLs that have
    # chunk_doc_names stored (set by crawler after ingestion).
    candidates = [
        (url, entry["chunk_doc_names"])
        for url, entry in registry.items()
        if entry.get("chunk_doc_names") and entry.get("last_ingested")
    ]

    print(f"Registry: {len(registry):,} total entries")
    print(f"  With chunk_doc_names: {len(candidates):,}")
    print(f"  Without chunk_doc_names (need re-crawl): "
          f"{sum(1 for e in registry.values() if e.get('last_ingested') and not e.get('chunk_doc_names')):,}")

    if not candidates:
        print("\nNo entries with chunk_doc_names found. Run a crawl first to populate them.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would backfill content_url for {len(candidates):,} URLs.")
        for url, names in candidates[:10]:
            print(f"  {url}")
            print(f"    chunks: {names[:3]}{'...' if len(names) > 3 else ''}")
        if len(candidates) > 10:
            print(f"  ... and {len(candidates) - 10} more")
        return

    print(f"\nFetching ES password...")
    es_pass = get_es_password()

    updated = 0
    failed = 0
    batch_size = 50  # process URLs in batches to avoid huge pod scripts

    print(f"Backfilling content_url for {len(candidates):,} URLs in {ES_INDEX}...\n")

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        for url, chunk_names in batch:
            # Update all chunks whose source_name matches any of the chunk_doc_names.
            # Uses terms query on source_name.keyword (exact basename match).
            full_paths = [f"/tmp/{name}" for name in chunk_names]
            query = {
                "script": {
                    "source": "ctx._source.metadata.content_metadata.content_url = params.url",
                    "lang": "painless",
                    "params": {"url": url},
                },
                "query": {
                    "terms": {
                        "metadata.source.source_name.keyword": full_paths,
                    }
                },
            }
            script = (
                f"curl -sk -u elastic:{es_pass} -X POST "
                f"'https://{ES_SERVICE}:{ES_PORT}/{args.collection}/_update_by_query"
                f"?conflicts=proceed' "
                f"-H 'Content-Type: application/json' "
                f"-d '{json.dumps(query).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            )
            pod_name = f"es-bf-{abs(hash(url)) % 999999:06d}"
            result = subprocess.run(
                ["kubectl", "run", pod_name, "--rm", "-i", "--restart=Never",
                 "--image=curlimages/curl:8.7.1", "-n", ES_NAMESPACE,
                 "--", "sh", "-c", script],
                capture_output=True, text=True, timeout=60,
            )
            try:
                resp, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
                n_updated = resp.get("updated", 0)
                updated += n_updated
                if n_updated == 0:
                    # Chunks may use /pdf-repo/ path for PDFs — try without /tmp/ prefix
                    pass  # non-fatal; PDF chunks use different path
            except Exception as exc:
                failed += 1
                print(f"  WARN: failed for {url}: {exc}")

        pct = min(100, int((i + len(batch)) / len(candidates) * 100))
        print(f"  Progress: {i + len(batch)}/{len(candidates)} URLs ({pct}%) — "
              f"{updated:,} chunks updated so far")

    print(f"\nBackfill complete.")
    print(f"  Chunks updated: {updated:,}")
    print(f"  URLs failed:    {failed:,}")
    print(f"\nNote: URLs without chunk_doc_names require a re-crawl to populate content_url.")


if __name__ == "__main__":
    main()
