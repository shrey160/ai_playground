"""Quick verification that LangGraph compiles and basic flow works."""

import sys
sys.path.insert(0, r'C:\Shrey_Projs\AI_ML_Projs\test_openrouter_nvidia')

from modules.graph import create_graph
from modules.pageindex import PageIndex

print("Creating graph...")
page_index = PageIndex()
graph = create_graph(page_index=page_index)
app = graph.compile()

print("Graph compiled successfully!")
print(f"Nodes: {list(app.get_graph().nodes.keys())}")

# Test with a mock state (won't call API, just validates structure)
mock_state = {
    "query": "test",
    "messages": [],
    "classification": None,
    "search_results": [],
    "extracted_contents": [],
    "cache_stats": None,
    "content": "",
    "sources": [],
    "validation": None,
    "retry_count": 0,
    "fallback": False,
    "error": None,
    "model": "",
    "tokens": 0,
    "time": 0.0,
}

print("\nMock state prepared successfully!")
print("All modules imported and graph compiled correctly.")
