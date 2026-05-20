"""Example usage of the Model Routing System."""

import sys
sys.path.insert(0, 'C:\\Shrey_Projs\\AI_ML_Projs\\test_openrouter_nvidia')

from modules import ModelRouter, ResearchPipeline

# Initialize router
router = ModelRouter()

# Example 1: Simple query (routes to worker)
print("=" * 70)
print("Example 1: Simple Query")
print("=" * 70)
result = router.route("What is the capital of France?")
print(f"Content: {result['content']}")
print(f"Model: {result['model']}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print()

# Example 2: Complex query (routes to orchestrator)
print("=" * 70)
print("Example 2: Complex Query")
print("=" * 70)
result = router.route(
    "Explain the theory of relativity and its implications for space travel",
    validate=True
)
print(f"Content: {result['content'][:200]}...")
print(f"Model: {result['model']}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print(f"Quality: {result.get('validation', {}).get('quality_score', 'N/A')}/10")
print()

# Example 3: Research query (routes to search + worker)
print("=" * 70)
print("Example 3: Research Query")
print("=" * 70)
research = ResearchPipeline(router.client)
result = router.route(
    "What are the latest developments in quantum computing?",
    research_fn=research.research
)
print(f"Content: {result['content'][:200]}...")
print(f"Sources: {result.get('sources', [])}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print()

print("Done! Check the outputs above.")
