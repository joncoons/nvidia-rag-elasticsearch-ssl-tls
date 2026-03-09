 #!/usr/bin/env python3                                                                                                   
"""                                                                                                                      
Count expected crawlable pages on nvidia.com via sitemap(s).                                                             
Falls back to a shallow BFS link count if no sitemap found.                                                              
Run: python3 count_nvidia_pages.py                                                                                       
"""                                                                                                                      
import requests                                                                                                          
import xml.etree.ElementTree as ET                                                                                       
from urllib.parse import urlparse                                                                                        
from collections import deque                                                                                            
                                                                                                                        
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-counter/1.0)"}                                                    
BASE = "https://www.nvidia.com"                                                                                          
NETLOC = "www.nvidia.com"                                                                                                
                                                                                                                        
def get_sitemaps(base_url):                                                                                              
    """Try robots.txt then common paths to find sitemap URLs."""                                                         
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
    """Recursively expand sitemap index → sitemaps → URLs. Returns set of page URLs."""
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

        # Sitemap index — recurse
        for loc in root.findall(f".//{tag('sitemap')}/{tag('loc')}"):
            pages |= expand_sitemap(loc.text.strip(), visited)

        # URL set — collect locs
        for loc in root.findall(f".//{tag('url')}/{tag('loc')}"):
            u = loc.text.strip()
            parsed = urlparse(u)
            if parsed.netloc == NETLOC or parsed.netloc.endswith("." + NETLOC):
                pages.add(u)

        print(f"  {url} → {len(pages):,} pages so far")
    except Exception as e:
        print(f"  SKIP {url}: {e}")
    return pages

if __name__ == "__main__":
    print("Fetching sitemaps...")
    all_pages = set()
    for sm in get_sitemaps(BASE):
        all_pages |= expand_sitemap(sm)

    if all_pages:
        print(f"\nTotal pages in sitemap: {len(all_pages):,}")
    else:
        print("No sitemap found — falling back to shallow BFS (depth 2)...")
        from bs4 import BeautifulSoup
        seen = set()
        queue = deque([(BASE + "/en-us/", 0)])
        while queue:
            url, depth = queue.popleft()
            if url in seen or depth > 2:
                continue
            seen.add(url)
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/"):
                        href = BASE + href
                    parsed = urlparse(href)
                    if (parsed.netloc == NETLOC or parsed.netloc.endswith("." + NETLOC)) \
                            and href not in seen:
                        queue.append((href, depth + 1))
            except Exception:
                pass
        print(f"Unique URLs found (depth ≤ 2): {len(seen):,}")

