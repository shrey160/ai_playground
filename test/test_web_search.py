import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing free web search tools for LLMs...")
print("=" * 60)

# Test 1: DuckDuckGo Search
print("\n1. Testing DuckDuckGo Search (free, no API key)...")
try:
    from ddgs import DDGS
    
    ddgs = DDGS()
    results = ddgs.text("python programming", max_results=2)
    
    print(f"[PASS] Success! Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['href']}")
        print(f"   Snippet: {result['body'][:100]}...")
except Exception as e:
    print(f"[FAIL] Error: {e}")

# Test 2: Jina AI Reader
print("\n2. Testing Jina AI Reader (free, no API key)...")
try:
    import requests
    
    url = "https://r.jina.ai/https://example.com"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        print(f"[PASS] Success! Extracted {len(response.text)} characters")
        print(f"   Preview: {response.text[:150]}...")
    else:
        print(f"[FAIL] Error: Status code {response.status_code}")
except Exception as e:
    print(f"[FAIL] Error: {e}")

# Test 3: DDGS Extract
print("\n3. Testing DDGS Built-in Extract (free, no API key)...")
try:
    from ddgs import DDGS
    
    ddgs = DDGS()
    result = ddgs.extract("https://example.com", fmt="text_markdown")
    
    if 'content' in result:
        print(f"[PASS] Success! Extracted {len(result['content'])} characters")
        print(f"   Preview: {result['content'][:150]}...")
    else:
        print(f"[WARN] Unexpected response format: {result}")
except Exception as e:
    print(f"[FAIL] Error: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("\nYou can now use these free tools in your notebook:")
print("- web_search() - Search the web")
print("- extract_with_jina() - Extract clean content from URLs")
print("- extract_with_ddgs() - Alternative URL extraction")
print("- search_news() - Search news articles")
print("- search_images() - Search images")