#!/usr/bin/env bash
ES_PASS=$(kubectl get secret -n rag rag-eck-elasticsearch-es-elastic-user -o jsonpath='{.data.elastic}' | base64 -d)

kubectl run es-count --rm -i --restart=Never --image=curlimages/curl:8.7.1 -n rag \
  -- sh -c "curl -sk -u elastic:${ES_PASS} https://rag-eck-elasticsearch-es-http:9200/_cat/indices?v" 2>/dev/null &

python3 - <<'EOF'
import json, os, datetime
path = '/home/joncoons/crawl-exports/nvidia_url_registry.json'
if not os.path.exists(path):
    print('Registry not yet flushed')
else:
    modified = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    with open(path) as f:
        reg = json.load(f)
    html  = [u for u in reg if not any(u.lower().endswith(x) for x in ['.pdf','.docx','.xlsx','.pptx','.xml'])]
    files = [u for u in reg if     any(u.lower().endswith(x) for x in ['.pdf','.docx','.xlsx','.pptx','.xml'])]
    print(f'Registry last flushed: {modified}')
    print(f'Total: {len(reg)} ({len(html)} pages, {len(files)} binary files)')
EOF

wait
