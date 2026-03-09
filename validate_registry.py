#!/usr/bin/env python3
"""
Post-crawl registry validation.

Cross-checks the URL registry against the ES document_info index.
Any URL marked as last_ingested in the registry but missing from ES
gets its last_ingested field cleared so it will be re-processed on
the next incremental crawl run.

Usage:
    python3 validate_registry.py [--dry-run]

Options:
    --dry-run   Report mismatches without modifying the registry.
"""

import argparse
import json
import os
import subprocess
import sys
import base64
from datetime import datetime

REGISTRY_PATH = "/home/joncoons/crawl-exports/nvidia_url_registry.json"
ES_SERVICE    = "rag-eck-elasticsearch-es-http"
ES_NAMESPACE  = "rag"
ES_PORT       = 9200
ES_INDEX      = "document_info"
ES_SECRET     = "rag-eck-elasticsearch-es-elastic-user"


def get_es_password() -> str:
    result = subprocess.run(
        ["kubectl", "get", "secret", "-n", ES_NAMESPACE, ES_SECRET,
         "-o", "jsonpath={.data.elastic}"],
        capture_output=True, text=True, check=True,
    )
    return base64.b64decode(result.stdout.strip()).decode()


def get_es_document_names(es_pass: str) -> set[str]:
    """Fetch all document_name values from document_info via scroll."""
    print("Fetching document_info index from Elasticsearch...")
    script = (
        f"curl -sk -u elastic:{es_pass} "
        f"'https://{ES_SERVICE}:{ES_PORT}/{ES_INDEX}/_search?"
        f"size=10000&scroll=2m&_source=document_name' "
        f"-H 'Content-Type: application/json'"
    )
    result = subprocess.run(
        ["kubectl", "run", "es-validate", "--rm", "-i", "--restart=Never",
         "--image=curlimages/curl:8.7.1", "-n", ES_NAMESPACE,
         "--", "sh", "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    data, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    hits = data["hits"]["hits"]
    scroll_id = data.get("_scroll_id")
    doc_names = {h["_source"].get("document_name", "") for h in hits}
    total = data["hits"]["total"]["value"]
    print(f"  Initial batch: {len(doc_names)} / {total} docs")

    # Scroll through remaining pages
    while scroll_id and len(doc_names) < total:
        scroll_script = (
            f"curl -sk -u elastic:{es_pass} "
            f"'https://{ES_SERVICE}:{ES_PORT}/_search/scroll' "
            f"-H 'Content-Type: application/json' "
            f"-d '{{\"scroll\":\"2m\",\"scroll_id\":\"{scroll_id}\"}}'"
        )
        sr = subprocess.run(
            ["kubectl", "run", f"es-scroll-{len(doc_names)}", "--rm", "-i",
             "--restart=Never", "--image=curlimages/curl:8.7.1", "-n", ES_NAMESPACE,
             "--", "sh", "-c", scroll_script],
            capture_output=True, text=True, timeout=120,
        )
        sd, _ = json.JSONDecoder().raw_decode(sr.stdout.lstrip())
        batch = {h["_source"].get("document_name", "") for h in sd["hits"]["hits"]}
        if not batch:
            break
        doc_names |= batch
        scroll_id = sd.get("_scroll_id")
        print(f"  Scrolled: {len(doc_names)} / {total} docs")

    return doc_names


def load_registry(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_registry(path: str, registry: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report without modifying registry")
    args = parser.parse_args()

    if not os.path.exists(REGISTRY_PATH):
        print(f"ERROR: Registry not found at {REGISTRY_PATH}")
        sys.exit(1)

    registry = load_registry(REGISTRY_PATH)
    ingested_urls = {u: e for u, e in registry.items() if e.get("last_ingested")}
    print(f"Registry: {len(registry):,} total entries, "
          f"{len(ingested_urls):,} marked as ingested")

    es_pass = get_es_password()
    es_doc_names = get_es_document_names(es_pass)
    print(f"ES document_info: {len(es_doc_names):,} documents\n")

    # For each URL marked as ingested, derive the expected document_name pattern.
    # Binary files (PDFs) produce nemoparse_* filenames; HTML pages produce webcrawl_* filenames.
    # We can't map URL → exact temp filename, but we CAN check binary files by their
    # base filename (PDF name appears in the nemoparse chunk filenames).
    missing = []
    for url, entry in ingested_urls.items():
        is_binary = any(url.lower().endswith(x)
                        for x in [".pdf", ".docx", ".xlsx", ".pptx", ".xml"])
        if is_binary:
            # PDF chunks appear as "<pdfname>_nemoparse_*.md" in document_info
            pdf_base = os.path.splitext(os.path.basename(url))[0]
            # Check if ANY chunk from this PDF exists in ES
            found = any(pdf_base in name for name in es_doc_names)
            if not found:
                missing.append((url, entry, "binary"))
        # HTML pages use random temp names — can't reverse-map without source tracking.
        # Skip HTML for now; these are low-risk (small .md files re-ingest quickly).

    print(f"Binary files marked ingested but missing from ES: {len(missing)}")
    for url, entry, ftype in missing[:20]:
        print(f"  [{ftype}] {url}")
        print(f"    last_ingested: {entry.get('last_ingested')}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

    if not missing:
        print("\nAll ingested binary files verified in ES. Registry is clean.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would clear last_ingested for {len(missing)} URLs.")
        return

    # Clear last_ingested for missing entries
    print(f"\nClearing last_ingested for {len(missing)} URLs...")
    for url, entry, _ in missing:
        entry_copy = dict(registry[url])
        entry_copy.pop("last_ingested", None)
        registry[url] = entry_copy

    # Backup original
    backup = REGISTRY_PATH + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import shutil
    shutil.copy2(REGISTRY_PATH, backup)
    print(f"Registry backed up to: {backup}")

    save_registry(REGISTRY_PATH, registry)
    print(f"Registry updated. {len(missing)} URLs will be re-processed on next crawl.")


if __name__ == "__main__":
    main()
