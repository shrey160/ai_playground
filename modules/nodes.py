from typing import Dict, Any, List
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from modules.schemas import (
    TaskClassification,
    ValidationResult,
    CacheStats,
    ResearchContext,
)
from modules.models import get_worker_model, get_orchestrator_model
from modules.pageindex import PageIndex
from modules.tools import web_search, extract_url_content


SYSTEM_SIMPLE = "You are a helpful assistant. Provide clear, concise answers."
SYSTEM_COMPLEX = "You are an expert assistant. Provide thorough, well-reasoned responses."
SYSTEM_RESEARCH = "You are a research assistant. Synthesize information from sources and cite them."


def classify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the query type using structured output."""
    query = state["query"]
    model = get_orchestrator_model(temperature=0.0)

    classifier = model.with_structured_output(TaskClassification)

    prompt = f"""Analyze the following user query and classify it.

Query: {query}

Consider:
- Does this require current/real-time information? (research)
- Is this a straightforward factual question? (simple)
- Does this require multi-step reasoning, analysis, or planning? (complex)

Respond with the task classification."""

    result = classifier.invoke([HumanMessage(content=prompt)])
    return {"classification": result}


def route_node(state: Dict[str, Any]) -> str:
    """Route to the appropriate node based on classification."""
    classification = state["classification"]
    if classification.task_type == "research" or classification.needs_search:
        return "research"
    elif classification.task_type == "complex":
        return "complex"
    else:
        return "simple"


def simple_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle simple queries with the worker model."""
    query = state["query"]
    messages = state.get("messages", [])
    model = get_worker_model()

    start = time.time()
    response = model.invoke(
        [SystemMessage(content=SYSTEM_SIMPLE)]
        + messages
        + [HumanMessage(content=query)]
    )
    elapsed = time.time() - start

    return {
        "content": response.content,
        "model": model.model_name,
        "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
        "time": elapsed,
    }


def complex_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle complex queries with the orchestrator model."""
    query = state["query"]
    messages = state.get("messages", [])
    model = get_orchestrator_model()

    start = time.time()
    response = model.invoke(
        [SystemMessage(content=SYSTEM_COMPLEX)]
        + messages
        + [HumanMessage(content=query)]
    )
    elapsed = time.time() - start

    return {
        "content": response.content,
        "model": model.model_name,
        "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
        "time": elapsed,
    }


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle research queries: search, extract, synthesize."""
    query = state["query"]
    page_index = state.get("page_index") or PageIndex()
    model = get_worker_model()

    start = time.time()

    # Step 1: Search
    search_results = web_search.invoke({"query": query, "max_results": 3})

    # Step 2: Extract content (with cache)
    contexts = []
    cache_hits = 0
    cache_misses = 0

    for result in search_results:
        url = result.get("href")
        if not url:
            continue

        extracted = extract_url_content.invoke(
            {"url": url, "page_index": page_index, "max_chars": 1500}
        )

        if extracted.get("cache_hit"):
            cache_hits += 1
        else:
            cache_misses += 1

        if extracted.get("content"):
            contexts.append(
                ResearchContext(
                    source=url,
                    title=extracted.get("title"),
                    content=extracted["content"],
                    from_cache=extracted.get("cache_hit", False),
                )
            )

    # Step 3: Synthesize with LLM
    context_text = "\n\n".join(
        [
            f"Source: {c.source}\n{c.content}"
            for c in contexts
        ]
    )

    prompt = f"""Based on the following search results, answer the question comprehensively and cite sources.

Question: {query}

Search Results:
{context_text}

Provide a comprehensive answer citing the sources using [Source: URL] format."""

    response = model.invoke([SystemMessage(content=SYSTEM_RESEARCH), HumanMessage(content=prompt)])
    elapsed = time.time() - start

    total = cache_hits + cache_misses
    hit_rate = cache_hits / total if total > 0 else 0.0

    return {
        "content": response.content,
        "model": model.model_name,
        "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
        "time": elapsed,
        "sources": [c.source for c in contexts],
        "extracted_contents": [c.model_dump() for c in contexts],
        "cache_stats": CacheStats(
            hits=cache_hits,
            misses=cache_misses,
            hit_rate=round(hit_rate, 2),
            entries_total=page_index.get_stats()["total_entries"] if page_index else 0,
        ),
    }


def validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output quality using structured output."""
    query = state["query"]
    content = state.get("content", "")

    if not content:
        return {
            "validation": ValidationResult(
                quality_score=1,
                is_accurate=False,
                is_complete=False,
                issues=["No content generated"],
                needs_fallback=True,
                feedback="Failed to generate content",
            )
        }

    model = get_orchestrator_model(temperature=0.0)
    validator = model.with_structured_output(ValidationResult)

    prompt = f"""Evaluate the quality of this AI response.

Original Query: {query}

Response:
{content}

Assess accuracy, completeness, and overall quality. Be strict but fair."""

    result = validator.invoke([HumanMessage(content=prompt)])
    return {"validation": result}


def should_retry(state: Dict[str, Any]) -> str:
    """Determine if we should retry with a stronger model."""
    validation = state.get("validation")
    retry_count = state.get("retry_count", 0)
    max_retries = 1

    if validation and validation.needs_fallback and retry_count < max_retries:
        return "fallback"
    return "end"


def fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retry with orchestrator model."""
    query = state["query"]
    messages = state.get("messages", [])
    model = get_orchestrator_model()

    start = time.time()
    response = model.invoke(
        [SystemMessage(content=SYSTEM_COMPLEX)]
        + messages
        + [HumanMessage(content=query)]
    )
    elapsed = time.time() - start

    return {
        "content": response.content,
        "model": model.model_name,
        "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
        "time": state.get("time", 0) + elapsed,
        "fallback": True,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def build_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the final result."""
    return state
