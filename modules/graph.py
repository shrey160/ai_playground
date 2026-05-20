from typing import Dict, Any, Optional
import time
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from modules.nodes import (
    classify_node,
    route_node,
    simple_node,
    complex_node,
    validate_node,
    should_retry,
    fallback_node,
    build_result,
)
from modules.pageindex import PageIndex
from modules.schemas import (
    TaskClassification,
    ValidationResult,
    CacheStats,
)


class GraphState(TypedDict):
    query: str
    messages: list
    classification: Optional[TaskClassification]
    search_results: list
    extracted_contents: list
    cache_stats: Optional[CacheStats]
    content: str
    sources: list
    validation: Optional[ValidationResult]
    retry_count: int
    fallback: bool
    error: Optional[str]
    model: str
    tokens: int
    time: float


def create_graph(page_index: Optional[PageIndex] = None) -> StateGraph:
    """Create and configure the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    # Create research node with bound page_index
    def research_with_cache(state: Dict[str, Any]) -> Dict[str, Any]:
        """Research node with injected page_index."""
        from modules.nodes import research_node
        state["page_index"] = page_index
        return research_node(state)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("simple", simple_node)
    workflow.add_node("complex", complex_node)
    workflow.add_node("research", research_with_cache)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("build_result", build_result)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add conditional edges from classify
    workflow.add_conditional_edges(
        "classify",
        route_node,
        {
            "simple": "simple",
            "complex": "complex",
            "research": "research",
        },
    )

    # All execution paths go to validation
    workflow.add_edge("simple", "validate")
    workflow.add_edge("complex", "validate")
    workflow.add_edge("research", "validate")

    # Validation check
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "fallback": "fallback",
            "end": "build_result",
        },
    )

    # Fallback goes back to validation
    workflow.add_edge("fallback", "validate")

    # Build result is the end
    workflow.add_edge("build_result", END)

    return workflow


class LangGraphApp:
    """Main application class for running the LangGraph pipeline."""

    def __init__(
        self,
        page_index: Optional[PageIndex] = None,
    ):
        self.page_index = page_index or PageIndex()
        graph = create_graph(page_index=self.page_index)
        self.app = graph.compile()

    def invoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        messages: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline for a single query."""
        start = time.time()

        state = {
            "query": query,
            "messages": messages or [],
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

        result = self.app.invoke(state)
        total_time = time.time() - start

        return {
            "content": result.get("content", ""),
            "sources": result.get("sources", []),
            "model": result.get("model", ""),
            "tokens": result.get("tokens", 0),
            "time": round(total_time, 2),
            "classification": (
                result["classification"].model_dump()
                if result.get("classification")
                else None
            ),
            "cache_stats": (
                result["cache_stats"].model_dump()
                if result.get("cache_stats")
                else None
            ),
            "validation": (
                result["validation"].model_dump()
                if result.get("validation")
                else None
            ),
            "fallback": result.get("fallback", False),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return self.page_index.get_stats()

    def clear_cache(self) -> int:
        """Clear all cached entries."""
        return self.page_index.clear()
