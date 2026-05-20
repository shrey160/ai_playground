"""
LangGraph-based AI Research Pipeline

Orchestrator-based routing with PageIndex cache and LangGraph state management.
"""

from .router import ModelRouter
from .research import ResearchPipeline
from .pageindex import PageIndex, CacheEntry
from .graph import LangGraphApp, create_graph
from .schemas import (
    TaskClassification,
    ValidationResult,
    CacheStats,
    ResearchContext,
    GraphResult,
)
from .models import get_base_model, get_orchestrator_model, get_worker_model
from .config import load_config, set_provider, get_config

__version__ = "0.3.0"
__all__ = [
    # Config
    "load_config",
    "set_provider",
    "get_config",
    # Legacy
    "ModelRouter",
    "ResearchPipeline",
    # New LangGraph
    "LangGraphApp",
    "create_graph",
    # Cache
    "PageIndex",
    "CacheEntry",
    # Schemas
    "TaskClassification",
    "ValidationResult",
    "CacheStats",
    "ResearchContext",
    "GraphResult",
    # Models
    "get_base_model",
    "get_orchestrator_model",
    "get_worker_model",
]
