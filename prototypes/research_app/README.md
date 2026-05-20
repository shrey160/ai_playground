# Research Pipeline Prototype

Streamlit-based web UI for the Research Pipeline with OpenRouter and NVIDIA NIM.

## Quick Start

1. **Install dependencies** (from project root):
```bash
pip install -r ../../requirements.txt
```

2. **Run the app**:
```bash
streamlit run app.py
```

3. **Open your browser** at `http://localhost:8501`

## Features

- **Provider Selection**: Switch between NVIDIA NIM and OpenRouter in the sidebar
- **Query Classification**: Automatic task classification (simple/complex/research) with reasoning
- **Pipeline Visualization**: Visual step-by-step execution tracker showing:
  1. Classify - Task type detection
  2. Execute/Research - Model execution or web search
  3. Validate - Quality check
  4. Fallback - Retry if needed
- **Metrics Dashboard**: Time, tokens, quality score, source count
- **Web Research**: DuckDuckGo search with content extraction and caching
- **Quality Validation**: Automatic output validation with fallback to stronger models
- **Query History**: Track and reload previous queries
- **Settings**: Configure max search results, validation toggle, quality threshold

## Architecture

The app demonstrates the full pipeline:

```
User Query
    |
    v
[Classify] → Task type, complexity score, needs search
    |
    |-- simple --> [Worker LLM]
    |-- complex --> [Orchestrator LLM with reasoning]
    |-- research --> [Search] → [Extract] → [LLM Synthesize]
    |
    v
[Validate] → Quality score, accuracy, completeness
    |
    |-- pass --> Return result
    |-- fail --> [Fallback] with stronger model
```

## Configuration

- Edit `../../config.yaml` to change models and providers
- Set API keys in `.env` at project root:
  - `NVIDIA_NIM_API_KEY` for NVIDIA
  - `OPENROUTER_API_KEY` for OpenRouter

## Project Context

This is a prototype UI for the research pipeline located at the project root. The pipeline uses:
- `modules/config.py` - Config-driven provider system
- `modules/router.py` - Task classification and routing
- `modules/research.py` - Web search and content extraction
- `modules/models.py` - LangChain model initialization

See the main project README for full documentation.
