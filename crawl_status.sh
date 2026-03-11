#!/usr/bin/env bash
# crawl_status.sh — show live crawl progress

ES_PASS=$(kubectl get secret -n rag rag-eck-elasticsearch-es-elastic-user -o jsonpath='{.data.elastic}' | base64 -d)
INGESTOR_POD=$(kubectl get pod -n rag -l app=ingestor-server -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

echo "=== Registry ==="
python3 - <<'EOF'
import json, os, datetime
path = '/home/joncoons/crawl-exports/nvidia_docs_url_registry.json'
if not os.path.exists(path):
    print('Registry not yet flushed')
else:
    modified = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    with open(path) as f:
        reg = json.load(f)
    html  = [u for u in reg if not any(u.lower().endswith(x) for x in ['.pdf','.docx','.xlsx','.pptx','.xml'])]
    files = [u for u in reg if     any(u.lower().endswith(x) for x in ['.pdf','.docx','.xlsx','.pptx','.xml'])]
    ingested = sum(1 for v in reg.values() if v.get('last_ingested'))
    pending  = sum(1 for v in reg.values() if not v.get('last_ingested'))
    print(f'Last flushed : {modified}')
    print(f'Total URLs   : {len(reg)} ({len(html)} pages, {len(files)} binary files)')
    print(f'Ingested     : {ingested}')
    print(f'Pending      : {pending}')
EOF

echo ""
echo "=== ES chunk count ==="
kubectl exec -n rag -c elasticsearch rag-eck-elasticsearch-es-default-0 -- \
  curl -sk -u "elastic:${ES_PASS}" "https://localhost:9200/nvidia_docs/_count" 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'nvidia_docs chunks: {d[\"count\"]}')" 2>/dev/null || echo "ES query failed"

echo ""
echo "=== Recent activity ==="
if [ -n "$INGESTOR_POD" ]; then
    kubectl logs -n rag "$INGESTOR_POD" --tail=200 2>/dev/null \
      | grep -E "Crawling|Collected|Dispatching|Ingest batch.*complete|batch.*fail|timed out|BFS complete|Sub-batch|nemoretriever-parse output|ERROR" \
      | grep -v "elastic_transport" \
      | tail -15
else
    echo "Ingestor pod not found"
fi

echo ""
echo "=== Error matrix ==="
cat /home/joncoons/crawl-exports/nvidia_docs_error_matrix.csv 2>/dev/null | head -20 || echo "No error matrix yet"
