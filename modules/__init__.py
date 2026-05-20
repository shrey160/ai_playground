"""
Model Routing System

Orchestrator-based routing with worker models for efficient task execution.
"""

from .router import ModelRouter
from .research import ResearchPipeline

__version__ = "0.1.0"
__all__ = ["ModelRouter", "ResearchPipeline"]
