from typing import List, Dict, Optional
import requests
from langchain.tools import tool
from ddgs import DDGS
from modules.pageindex import PageIndex


@tool
def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo. Returns a list of results with title, href, and body."""
    ddgs = DDGS()
    results = ddgs.text(query, max_results=max_results)
    return results


@tool
def extract_url_content(
    url: str,
    page_index: Optional[PageIndex] = None,
    timeout: int = 30,
    max_chars: int = 1500,
) -> Dict[str, Optional[str]]:
    """Extract content from a URL using Jina AI Reader, with PageIndex cache support."""
    cache_hit = False
    content = None
    title = None

    if page_index is not None:
        entry = page_index.get(url)
        if entry is not None:
            content = entry.content
            title = entry.title
            cache_hit = True

    if not cache_hit:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, timeout=timeout)
            if response.status_code == 200:
                content = response.text[:max_chars]
                # Try to extract title from first line
                lines = content.split("\n")
                if lines:
                    title = lines[0].strip()[:200]

                if page_index is not None:
                    page_index.set(url, content, title=title, source="jina")
        except Exception:
            content = None

    return {
        "url": url,
        "title": title,
        "content": content,
        "cache_hit": cache_hit,
    }


@tool
def news_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search for news using DuckDuckGo News. Returns news articles with title, href, and body."""
    ddgs = DDGS()
    results = ddgs.news(query, max_results=max_results)
    return results
