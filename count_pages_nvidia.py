#!/usr/bin/env python3                                                                                                   
"""                                                                                                                      
Count expected crawlable pages under https://www.nvidia.com/en-us/                                                       
Run: python3 count_nvidia_pages.py                                                                                       
"""                                                                                                                      
import requests                                                                                                          
import xml.etree.ElementTree as ET                                                                                       
from urllib.parse import urlparse                                                                                        
from collections import deque                                                                                            
                                                                                                                        
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-counter/1.0)"}                                                    
BASE = "https://www.nvidia.com"                                                                                          
NETLOC = "www.nvidia.com"                                                                                                
PATH_PREFIX = "/en-us/"                                                                                                  
                                                                                                                        
def get_sitemaps(base_url):                                                                                              
    sitemaps = []                                                                                                        
    try:
        r = requests.get(f"{base_url}/robots.txt", headers=HEADERS, timeout=10)
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemaps.append(line.split(":", 1)[1].strip())
    except Exception:
        pass
    if not sitemaps:
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/en-us/sitemap.xml"]:
            sitemaps.append(f"{base_url}{path}")
    return sitemaps

def expand_sitemap(url, visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return set()
    visited.add(url)
    pages = set()
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return pages
        root = ET.fromstring(r.content)
        ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        def tag(t): return f"{{{ns}}}{t}" if ns else t

        for loc in root.findall(f".//{tag('sitemap')}/{tag('loc')}"):
            pages |= expand_sitemap(loc.text.strip(), visited)

        for loc in root.findall(f".//{tag('url')}/{tag('loc')}"):
            u = loc.text.strip()
            parsed = urlparse(u)
            if parsed.netloc == NETLOC and parsed.path.startswith(PATH_PREFIX):
                pages.add(u)

        print(f"  {url} → {len(pages):,} /en-us/ pages so far")
    except Exception as e:
        print(f"  SKIP {url}: {e}")
    return pages

if __name__ == "__main__":
    print(f"Counting pages under {BASE}{PATH_PREFIX} via sitemap...")
    all_pages = set()
    for sm in get_sitemaps(BASE):
        all_pages |= expand_sitemap(sm)

    if all_pages:
        print(f"\nTotal /en-us/ pages in sitemap: {len(all_pages):,}")
    else:
        print("No sitemap found — falling back to BFS (depth 3)...")
        from bs4 import BeautifulSoup
        seen = set()
        queue = deque([(BASE + PATH_PREFIX, 0)])
        while queue:
            url, depth = queue.popleft()
            if url in seen or depth > 3:
                continue
            seen.add(url)
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/en-us/"):
                        href = BASE + href
                    parsed = urlparse(href)
                    if parsed.netloc == NETLOC and parsed.path.startswith(PATH_PREFIX) \
                            and href not in seen:
                        queue.append((href, depth + 1))
            except Exception:
                pass
        print(f"Unique /en-us/ URLs found: {len(seen):,}")
