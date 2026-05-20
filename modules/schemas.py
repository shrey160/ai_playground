from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TaskClassification(BaseModel):
    task_type: Literal["simple", "complex", "research"] = Field(
        description="Type of task: simple (factual Q&A), complex (multi-step reasoning), or research (requires current information)"
    )
    complexity_score: int = Field(
        ge=1,
        le=10,
        description="Complexity score from 1 (very simple) to 10 (very complex)"
    )
    needs_search: bool = Field(
        description="Whether the query requires web search for current/real-time information"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


class ValidationResult(BaseModel):
    quality_score: int = Field(
        ge=1,
        le=10,
        description="Quality score from 1 (poor) to 10 (excellent)"
    )
    is_accurate: bool = Field(
        description="Whether the response appears factually accurate"
    )
    is_complete: bool = Field(
        description="Whether the response fully answers the query"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of any issues or concerns with the response"
    )
    needs_fallback: bool = Field(
        description="Whether a fallback to a stronger model is needed"
    )
    feedback: str = Field(
        description="Brief assessment of the response quality"
    )


class CacheStats(BaseModel):
    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    hit_rate: float = Field(default=0.0, description="Cache hit rate (0.0 to 1.0)")
    entries_total: int = Field(default=0, description="Total entries in cache")


class ResearchContext(BaseModel):
    source: str = Field(description="Source URL")
    title: Optional[str] = Field(default=None, description="Page title")
    content: str = Field(description="Extracted content")
    from_cache: bool = Field(default=False, description="Whether content came from cache")


class GraphResult(BaseModel):
    content: str = Field(description="Final response content")
    sources: List[str] = Field(default_factory=list, description="List of source URLs")
    model: str = Field(description="Model used for final generation")
    tokens: int = Field(default=0, description="Total tokens used")
    time: float = Field(description="Total execution time in seconds")
    classification: Optional[TaskClassification] = Field(default=None, description="Task classification")
    cache_stats: Optional[CacheStats] = Field(default=None, description="Cache statistics")
    validation: Optional[ValidationResult] = Field(default=None, description="Validation result")
    fallback: bool = Field(default=False, description="Whether fallback was used")
