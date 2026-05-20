# Progress Report

## Completed

### API Setup
- OpenRouter API integration with free nvidia/nemotron model
- NVIDIA NIM API integration (https://integrate.api.nvidia.com/v1)
- Environment variables in `.env` (API keys protected per AGENTS.md)
- `.gitignore` updated to exclude `.env`, `env/`, and `data/`

### Config-Driven Provider System (New)
- **`config.yaml`**: Central configuration file at project root
  - Defines providers with base_url, api_key_env, and model aliases
  - No secrets stored in config (references env vars)
  - Default provider: NVIDIA NIM
  
- **`modules/config.py`**: Singleton config manager
  - `load_config()`: Load YAML config once at startup
  - `set_provider()`: Switch provider (e.g., "nvidia" → "openrouter")
  - `get_config()`: Access current provider settings
  - Model aliases: `orchestrator`, `worker` (resolved per provider)

- **Updated modules** to use config system:
  - `modules/models.py`: ChatOpenAI models use config for URL/key/model
  - `modules/router.py`: OpenAI client uses config
  - `modules/research.py`: OpenAI client uses config
  - `modules/nodes.py`: LangChain models use config
  - `modules/__init__.py`: Exports config utilities

- **`main.py`**: CLI with `--provider` flag

- **`notebooks/test_config_providers.ipynb`**: Demo notebook

### Web Search Tools (Free, No API Keys)
- **DuckDuckGo Search** (`ddgs`) - web search, news, images
- **Jina AI Reader** - extract clean markdown from URLs
- **DDGS Extract** - built-in URL content extraction

### LangGraph Architecture
- **LangChain Integration**: Replaced raw OpenAI client with LangChain ChatOpenAI
- **LangGraph State Machine**: Graph-based routing with typed state management
- **Structured Output**: Pydantic models replacing fragile JSON parsing
- **PageIndex Cache**: SQLite-based persistent cache for extracted web content
- **Tool Wrappers**: DDGS search and Jina extraction as LangChain @tool functions
- **Validation + Fallback**: Quality check with automatic retry using orchestrator model

### Notebooks
- `openrouter_api.ipynb` - OpenRouter LLM reasoning demo
- `web_search.ipynb` - Full pipeline: search → extract → LLM
- `model_routing_robust.ipynb` - Robust routing WITHOUT structured output (plain-text parsing)
- `03_langgraph_pipeline.ipynb` - New LangGraph architecture demo with cache comparison
- `test_gemma_4_31b.ipynb` - Test google/gemma-4-31b-it:free via OpenRouter
- `test_nvidia_nim_nemotron_reasoning.ipynb` - Test NVIDIA NIM reasoning model
- `test_config_providers.ipynb` - Config-driven provider switching demo

### Test Scripts
- `test_api.py` - API connectivity test
- `test_pipeline.py` - End-to-end pipeline test
- `test_web_search.py` - Web scraping tools test
- `test_router.py` - Model router example usage
- `test_imports.py` - LangGraph module import verification
- `test_graph_compile.py` - Graph structure validation

### Model Routing System
- **Orchestrator** (configurable alias) - Task classifier & validator
- **Workers** (configurable alias) - Fast task execution
- **Routing Strategy**: Plain-text parsing (NO JSON) with YES/NO classification
- **Task Decomposition**: Breaks complex tasks into numbered subtasks for workers
- **Integration**: Existing web search pipeline as worker tool

## Architecture

### Provider Configuration
```yaml
# config.yaml
providers:
  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    models:
      orchestrator: "nvidia/nemotron-3-super-120b-a12b:free"
      worker: "nvidia/nemotron-3-nano-30b-a3b:free"

  nvidia:
    name: "NVIDIA NIM"
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key_env: "NVIDIA_NIM_API_KEY"
    models:
      orchestrator: "nvidia/nemotron-3-super-120b-a12b"
      worker: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

default_provider: "nvidia"
```

### Config Usage
```python
from modules import load_config, set_provider, get_config

# Setup once at startup
load_config("config.yaml")
set_provider("nvidia")  # or "openrouter"

# Access anywhere
config = get_config()
model = config.get_model("worker")      # Provider-specific model
api_key = config.api_key                 # From env var
base_url = config.base_url               # Provider endpoint
```

### Legacy Pipeline (Robust)
```
User Query
    |
    v
[Task Analyzer] (Orchestrator - configurable)
    Can a worker handle this? -> YES or NO
    |
    YES -> [Worker] executes directly
    NO  -> [Decomposer] breaks into numbered subtasks
              |
              v
          [Worker Pool] executes each subtask
              |
              v
          [Synthesizer] combines results
    |
    v
Return Result
```

### New LangGraph Pipeline
```
User Query
    |
    v
[Classify Node] (Orchestrator - structured output)
    |
    |-- simple --> [Worker LLM] --> [Validate]
    |-- complex --> [Orchestrator LLM] --> [Validate]
    |-- research --> [Search] --> [Extract*] --> [LLM Synthesize] --> [Validate]
                                                      |
                                               [*Cache Check]
                                                      |
                                             hit --> use cached
                                             miss --> Jina AI --> store in cache
    |
    v
[Result]
```

### Key Features
| Feature | Tool | Cost |
|---------|------|------|
| Web Search | `DDGS.text()` | Free |
| URL Extract | `r.jina.ai/{url}` | Free |
| URL Extract | `DDGS.extract()` | Free |
| News Search | `DDGS.news()` | Free |
| Image Search | `DDGS.images()` | Free |
| LLM | OpenRouter free tier / NVIDIA NIM | Free* |
| Cache | SQLite PageIndex | Free |
| Config | YAML + env vars | Free |

*Free tiers may have rate limits

## Project Structure
```
test_openrouter_nvidia/
├── .env                          # API keys (gitignored)
├── .gitignore                    # Excludes .env, env/, data/
├── AGENTS.md                     # Agent instructions
├── config.yaml                   # Provider configuration (new)
├── main.py                       # Entry point with --provider flag
├── pyproject.toml                # Project dependencies
├── README.md                     # Project docs
├── uv.lock                       # Lock file
│
├── data/                         # PageIndex cache database
│   └── .gitkeep
│
├── modules/                      # Reusable modules
│   ├── __init__.py              # Package exports (v0.3.0)
│   ├── config.py                # Config loader + singleton (new)
│   ├── router.py                # ModelRouter class (uses config)
│   ├── research.py              # ResearchPipeline class (uses config)
│   ├── schemas.py               # Pydantic structured output models
│   ├── models.py                # LangChain model setup (uses config)
│   ├── pageindex.py             # SQLite cache implementation
│   ├── tools.py                 # LangChain @tool wrappers
│   ├── nodes.py                 # Graph node functions (uses config)
│   └── graph.py                 # LangGraph builder + LangGraphApp
│
├── notebooks/                    # Jupyter notebooks
│   ├── openrouter_api.ipynb     # LLM reasoning demo
│   ├── web_search.ipynb         # Search pipeline
│   ├── model_routing_robust.ipynb  # Robust routing (NO JSON)
│   ├── 03_langgraph_pipeline.ipynb  # LangGraph + PageIndex cache demo
│   ├── test_gemma_4_31b.ipynb    # Test Gemma model via OpenRouter
│   ├── test_nvidia_nim_nemotron_reasoning.ipynb  # Test NVIDIA NIM
│   └── test_config_providers.ipynb  # Config provider demo (new)
│
├── test/                        # Test scripts
│   ├── test_api.py             # API connectivity
│   ├── test_pipeline.py        # End-to-end pipeline
│   ├── test_web_search.py      # Web scraping tools
│   ├── test_router.py          # Router usage example
│   ├── test_imports.py         # Module import verification
│   └── test_graph_compile.py   # Graph structure validation
│
└── context/
    └── progress.md             # This file
```

## Next Steps
- [x] Implement model routing system (orchestrator + workers)
- [x] Create router module with task classification
- [x] Add validation layer with quality threshold
- [x] Implement robust routing WITHOUT structured output (plain-text parsing)
- [x] Build LangGraph architecture with state management
- [x] Create PageIndex cache tier with SQLite backend
- [x] Implement structured output with Pydantic models
- [x] Wrap search/extract as LangChain Tools
- [x] Config-driven provider system (YAML + env vars)
- [ ] Add automatic failover between providers (future)
- [ ] Add runtime provider switching (future)
- [ ] Add streaming support for real-time responses
- [ ] Add Tavily/Firecrawl integration (optional, requires API keys)
- [ ] Create agent workflow with persistent memory
- [ ] Add background cache refresh
- [ ] Build web UI for interactive use

## Environment
- Python 3.10
- uv package manager
- Virtual env: `env/`
- LangGraph, LangChain, LangChain-OpenAI
- PyYAML (for config)
