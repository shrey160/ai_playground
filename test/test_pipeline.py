import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS
import requests

# Load environment variables
load_dotenv()

# Initialize OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

print("Testing Complete Pipeline: Search -> Extract -> LLM")
print("=" * 70)

# Step 1: Search
print("\n[1] Searching for 'latest AI news'...")
ddgs = DDGS()
results = ddgs.text("latest AI news", max_results=2)
print(f"Found {len(results)} results")

# Step 2: Extract content from first result
if results:
    url = results[0]['href']
    print(f"\n[2] Extracting content from: {url}")
    
    try:
        # Use Jina AI Reader
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url, timeout=30)
        content = response.text if response.status_code == 200 else results[0]['body']
        print(f"Extracted {len(content)} characters")
    except Exception as e:
        content = results[0]['body']
        print(f"Using snippet instead: {e}")
    
    # Step 3: Send to LLM
    print(f"\n[3] Sending to OpenRouter LLM...")
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Summarize the provided web content in 3 bullet points."
            },
            {
                "role": "user",
                "content": f"Content:\n{content[:2000]}\n\nSummarize this in 3 bullet points:"
            }
        ]
        
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=messages,
            extra_body={"reasoning": {"enabled": True}}
        )
        
        print("\n" + "=" * 70)
        print("LLM RESPONSE:")
        print("=" * 70)
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        print("Make sure your OPENROUTER_API_KEY is set in .env")

print("\n" + "=" * 70)
print("[DONE] Pipeline test complete!")
print("\nYou can now use web_search.ipynb for interactive use.")