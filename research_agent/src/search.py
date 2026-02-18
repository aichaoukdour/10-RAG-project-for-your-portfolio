import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time

from .config import TIMEOUT, SEARCH_RESULTS

def unwrap_ddg(url):
    """If DuckDuckGo returns a redirect wrapper, extract the real URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except Exception:
        pass
    return url

def search_web(query, max_results=SEARCH_RESULTS):
    """Search the web and return a list of URLs."""
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                url = r.get("href") or r.get("url")
                if not url:
                    continue
                url = unwrap_ddg(url)
                urls.append(url)
    except Exception as e:
        print(f"Search failed: {e}")
    return urls

def fetch_text(url, timeout=TIMEOUT):
    """Fetch and clean text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if r.status_code != 200:
            return ""
        
        ct = r.headers.get("content-type", "")
        if "html" not in ct.lower():
            return ""
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Remove noisy tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside", "form"]):
            tag.extract()
            
        # Extract paragraph text
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        
        if text.strip():
            return re.sub(r"\s+", " ", text).strip()
            
        # Fallback to meta description
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        
        if soup.title and soup.title.string:
            return soup.title.string.strip()
            
    except Exception as e:
        # Fail silently but could log error if needed
        return ""
    return ""
