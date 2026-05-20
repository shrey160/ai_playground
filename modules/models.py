import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from modules.config import get_config

load_dotenv()


def get_base_model(model_alias: Optional[str] = None, temperature: float = 0.1) -> ChatOpenAI:
    """Get a base LangChain chat model using the current provider config."""
    config = get_config()
    model_name = config.get_model(model_alias or "worker")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=config.api_key,
        openai_api_base=config.base_url,
    )


def get_orchestrator_model(temperature: float = 0.1) -> ChatOpenAI:
    """Get the orchestrator model (larger, more capable)."""
    return get_base_model("orchestrator", temperature)


def get_worker_model(temperature: float = 0.1) -> ChatOpenAI:
    """Get the worker model (faster, cheaper)."""
    return get_base_model("worker", temperature)
