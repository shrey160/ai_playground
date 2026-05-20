# Research Pipeline with OpenRouter & NVIDIA NIM

An intelligent research pipeline that routes queries between LLM providers (OpenRouter, NVIDIA NIM) with web search, content extraction, and quality validation.

## Features

- **Multi-Provider Support**: Switch between OpenRouter and NVIDIA NIM via config
- **Intelligent Routing**: Orchestrator classifies tasks as simple/complex/research
- **Web Research**: DuckDuckGo search + Jina AI content extraction
- **Quality Validation**: Automatic validation with fallback to stronger models
- **Caching**: SQLite-based PageIndex cache for extracted content
- **LangGraph Architecture**: State-machine based pipeline execution
- **Streamlit UI**: Interactive web interface in `prototypes/research_app/`

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set API keys** in `.env`:
```
NVIDIA_NIM_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

3. **Run the Streamlit app**:
```bash
cd prototypes/research_app
streamlit run app.py
```

Or use the CLI:
```bash
python main.py --provider nvidia "What is quantum computing?"
```

## Architecture

```
User Query
    |
    v
[Classify] (Orchestrator model)
    |
    |-- simple --> [Worker LLM] --> [Validate]
    |-- complex --> [Orchestrator LLM] --> [Validate]
    |-- research --> [Search] --> [Extract] --> [LLM Synthesize] --> [Validate]
    |
    v
[Result]
```

## Project Structure

```
├── config.yaml              # Provider configuration
├── main.py                  # CLI entry point
├── modules/                 # Core modules
│   ├── config.py           # Config loader (singleton)
│   ├── router.py           # ModelRouter (classification + routing)
│   ├── research.py         # ResearchPipeline (search + extract + synthesize)
│   ├── models.py           # LangChain model setup
│   ├── nodes.py            # LangGraph nodes
│   ├── graph.py            # LangGraph builder
│   ├── schemas.py          # Pydantic models
│   ├── tools.py            # Web search/extract tools
│   └── pageindex.py        # SQLite cache
├── notebooks/               # Jupyter notebooks
│   ├── test_config_providers.ipynb
│   ├── 03_langgraph_pipeline.ipynb
│   └── ...
├── prototypes/              # UI prototypes
│   └── research_app/       # Streamlit app
│       ├── app.py
│       └── README.md
└── test/                    # Test scripts
```

## Configuration

Edit `config.yaml` to change providers/models:

```yaml
providers:
  nvidia:
    name: "NVIDIA NIM"
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key_env: "NVIDIA_NIM_API_KEY"
    models:
      orchestrator: "nvidia/nemotron-3-super-120b-a12b"
      worker: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
  
  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    models:
      orchestrator: "nvidia/nemotron-3-super-120b-a12b:free"
      worker: "nvidia/nemotron-3-nano-30b-a3b:free"

default_provider: "nvidia"
```

## Notebooks

- `test_config_providers.ipynb` - Provider switching demo
- `03_langgraph_pipeline.ipynb` - LangGraph pipeline with cache
- `test_gemma_4_31b.ipynb` - Test Gemma model
- `test_nvidia_nim_nemotron_reasoning.ipynb` - Test NVIDIA reasoning

## Environment

- Python 3.10
- LangGraph, LangChain, LangChain-OpenAI
- Streamlit (for UI)

## License

MIT
