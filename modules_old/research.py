import requests
from typing import Dict, List, Any, Optional
from ddgs import DDGS


class ResearchPipeline:
    """Research pipeline with web search and content extraction."""
    
    def __init__(self, client, model: str = "nvidia/nemotron-3-nano-30b-a3b:free"):
        self.client = client
        self.model = model
        self.ddgs = DDGS()
    
    def search(
        self, 
        query: str, 
        max_results: int = 3
    ) -> List[Dict[str, str]]:
        """Perform web search using DuckDuckGo."""
        results = self.ddgs.text(query, max_results=max_results)
        return results
    
    def extract_content(
        self, 
        url: str, 
        timeout: int = 30,
        max_chars: int = 1500
    ) -> Optional[str]:
        """Extract content from URL using Jina AI Reader."""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, timeout=timeout)
            if response.status_code == 200:
                return response.text[:max_chars]
        except Exception:
            pass
        return None
    
    def research(
        self, 
        query: str, 
        max_results: int = 3,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute full research pipeline: search → extract → summarize."""
        
        import time
        start_time = time.time()
        
        # Step 1: Search
        search_results = self.search(query, max_results)
        
        # Step 2: Extract content
        contexts = []
        for result in search_results:
            content = self.extract_content(result['href'])
            if content:
                contexts.append({
                    "source": result['href'],
                    "title": result.get('title', 'Unknown'),
                    "content": content
                })
        
        # Step 3: Send to LLM
        context_text = "\n\n".join([
            f"Source: {c['source']}\n{c['content']}"
            for c in contexts
        ])
        
        prompt = f"""Based on the following search results, answer the question.

Question: {query}

Search Results:
{context_text}

Provide a comprehensive answer citing the sources."""
        
        default_system = "You are a research assistant. Synthesize information from multiple sources."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message or default_system},
                {"role": "user", "content": prompt}
            ]
        )
        
        elapsed = time.time() - start_time
        
        return {
            "query": query,
            "content": response.choices[0].message.content,
            "sources": [c['source'] for c in contexts],
            "tokens": response.usage.total_tokens,
            "time": round(elapsed, 2),
            "num_sources": len(contexts)
        }
