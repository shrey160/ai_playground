"""Example usage of the LangGraph-based AI Research Pipeline."""

import sys
sys.path.insert(0, r'C:\Shrey_Projs\AI_ML_Projs\test_openrouter_nvidia')

from modules import LangGraphApp

# Initialize app
app = LangGraphApp()

print("=" * 70)
print("LangGraph AI Research Pipeline - Examples")
print("=" * 70)

# Example 1: Simple query
print("\n[Example 1] Simple Query")
print("-" * 70)
result = app.invoke("What is the capital of France?")
print(f"Content: {result['content']}")
print(f"Model: {result['model']}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print(f"Type: {result['classification']['task_type']}")
print(f"Quality: {result['validation']['quality_score']}/10")

# Example 2: Research query
print("\n[Example 2] Research Query")
print("-" * 70)
result = app.invoke("What are the latest developments in quantum computing?")
print(f"Content: {result['content'][:300]}...")
print(f"Sources: {len(result['sources'])} sources")
print(f"Time: {result['time']}s")
print(f"Type: {result['classification']['task_type']}")
if result['cache_stats']:
    print(f"Cache: {result['cache_stats']['hits']} hits, {result['cache_stats']['misses']} misses")

# Show cache stats
print("\n[Cache Statistics]")
print("-" * 70)
stats = app.get_cache_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"DB size: {stats['db_size_mb']} MB")

print("\n" + "=" * 70)
print("Done! Check the outputs above.")
