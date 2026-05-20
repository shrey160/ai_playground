"""
Streamlit Prototype for Research Pipeline

Run with: streamlit run prototypes/research_app/app.py
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Research Pipeline Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #fafafa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .pipeline-step {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2ecc71;
    }
    .pipeline-step.active {
        background-color: #d5f5e3;
        border-left-color: #27ae60;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "config_loaded" not in st.session_state:
    st.session_state.config_loaded = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = {}


def load_configuration():
    """Load config and return status."""
    try:
        from modules.config import load_config, get_config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        load_config(config_path)
        st.session_state.config_loaded = True
        return get_config()
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None


def check_api_keys(provider: str) -> Dict[str, bool]:
    """Check if required API keys are set."""
    keys = {}
    if provider in ["nvidia", "all"]:
        keys["NVIDIA_NIM_API_KEY"] = bool(os.getenv("NVIDIA_NIM_API_KEY"))
    if provider in ["openrouter", "all"]:
        keys["OPENROUTER_API_KEY"] = bool(os.getenv("OPENROUTER_API_KEY"))
    return keys


def run_research_pipeline(query: str, max_results: int, provider: str, 
                         validate: bool = True, threshold: int = 7) -> Dict[str, Any]:
    """Execute the research pipeline."""
    from modules.config import set_provider, get_config
    from modules.research import ResearchPipeline
    from modules.router import ModelRouter
    
    # Set provider
    set_provider(provider)
    config = get_config()
    
    # Initialize components
    router = ModelRouter()
    research = ResearchPipeline()
    
    pipeline_steps = {}
    start_time = time.time()
    
    # Step 1: Classify
    step_start = time.time()
    classification = router.classify_task(query)
    pipeline_steps["classify"] = {
        "status": "completed",
        "time": round(time.time() - step_start, 2),
        "data": classification
    }
    
    # Step 2: Route & Execute
    task_type = classification.get("task_type", "simple")
    needs_search = classification.get("needs_search", False)
    
    if task_type == "research" or needs_search:
        step_start = time.time()
        result = research.research(query, max_results=max_results)
        pipeline_steps["research"] = {
            "status": "completed",
            "time": round(time.time() - step_start, 2),
            "data": {
                "num_sources": result.get("num_sources", 0),
                "sources": result.get("sources", [])
            }
        }
        pipeline_steps["execute"] = {
            "status": "completed",
            "time": 0,
            "data": {"model": result.get("model", config.get_model("worker"))}
        }
    else:
        step_start = time.time()
        result = router.execute_task(
            query, 
            task_type, 
            reasoning=(task_type == "complex")
        )
        pipeline_steps["execute"] = {
            "status": "completed",
            "time": round(time.time() - step_start, 2),
            "data": {"model": result.get("model", "unknown"), "task_type": task_type}
        }
    
    # Step 3: Validate (optional)
    if validate and "content" in result:
        step_start = time.time()
        validation = router.validate_output(query, result["content"], threshold)
        pipeline_steps["validate"] = {
            "status": "completed",
            "time": round(time.time() - step_start, 2),
            "data": validation
        }
        result["validation"] = validation
        
        # Fallback if needed
        if validation.get("needs_fallback", False):
            step_start = time.time()
            fallback = router.execute_task(query, "complex", reasoning=True)
            result["fallback_response"] = fallback["content"]
            result["fallback"] = True
            pipeline_steps["fallback"] = {
                "status": "completed",
                "time": round(time.time() - step_start, 2),
                "data": {"model": fallback.get("model", "unknown")}
            }
    
    total_time = round(time.time() - start_time, 2)
    result["total_time"] = total_time
    result["query"] = query
    result["provider"] = provider
    result["pipeline_steps"] = pipeline_steps
    
    return result


def display_pipeline_steps(steps: Dict[str, Any]):
    """Display pipeline steps visualization."""
    st.subheader("🔍 Pipeline Execution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "classify" in steps:
            st.markdown("""
            <div class="pipeline-step active">
                <strong>1. Classify</strong><br>
                <span style="color: green;">✓ Completed</span><br>
                <small>{}s</small>
            </div>
            """.format(steps["classify"]["time"]), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-step">
                <strong>1. Classify</strong><br>
                <span style="color: gray;">⏳ Waiting...</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if "research" in steps:
            st.markdown("""
            <div class="pipeline-step active">
                <strong>2. Research</strong><br>
                <span style="color: green;">✓ Completed</span><br>
                <small>{}s | {} sources</small>
            </div>
            """.format(
                steps["research"]["time"],
                steps["research"]["data"]["num_sources"]
            ), unsafe_allow_html=True)
        elif "execute" in steps:
            st.markdown("""
            <div class="pipeline-step active">
                <strong>2. Execute</strong><br>
                <span style="color: green;">✓ Completed</span><br>
                <small>{}s | {}</small>
            </div>
            """.format(
                steps["execute"]["time"],
                steps["execute"]["data"].get("task_type", "")
            ), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-step">
                <strong>2. Execute/Research</strong><br>
                <span style="color: gray;">⏳ Waiting...</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if "validate" in steps:
            score = steps["validate"]["data"].get("quality_score", 0)
            color = "green" if score >= 7 else "orange" if score >= 5 else "red"
            st.markdown("""
            <div class="pipeline-step active">
                <strong>3. Validate</strong><br>
                <span style="color: {};">✓ Score: {}/10</span><br>
                <small>{}s</small>
            </div>
            """.format(color, score, steps["validate"]["time"]), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-step">
                <strong>3. Validate</strong><br>
                <span style="color: gray;">⏳ Waiting...</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if "fallback" in steps:
            st.markdown("""
            <div class="pipeline-step active">
                <strong>4. Fallback</strong><br>
                <span style="color: orange;">⚠ Triggered</span><br>
                <small>{}s</small>
            </div>
            """.format(steps["fallback"]["time"]), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-step">
                <strong>4. Fallback</strong><br>
                <span style="color: gray;">Not needed</span>
            </div>
            """, unsafe_allow_html=True)


def display_metrics(result: Dict[str, Any]):
    """Display metrics dashboard."""
    st.subheader("📊 Metrics")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            label="Total Time",
            value=f"{result.get('total_time', result.get('time', 0)):.2f}s"
        )
    
    with cols[1]:
        st.metric(
            label="Tokens Used",
            value=result.get("tokens", 0)
        )
    
    with cols[2]:
        validation = result.get("validation", {})
        score = validation.get("quality_score", "N/A")
        st.metric(
            label="Quality Score",
            value=f"{score}/10" if isinstance(score, int) else score
        )
    
    with cols[3]:
        sources = result.get("sources", [])
        st.metric(
            label="Sources",
            value=len(sources)
        )


def display_sources(result: Dict[str, Any]):
    """Display source cards."""
    sources = result.get("sources", [])
    if not sources:
        st.info("No web sources were used for this query.")
        return
    
    st.subheader(f"📚 Sources ({len(sources)})")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source[:60]}..."):
            st.markdown(f"**URL:** [{source}]({source})")
            
            # Try to get extracted content if available
            extracted = result.get("extracted_contents", [])
            for ext in extracted:
                if ext.get("source") == source:
                    st.markdown(f"**Title:** {ext.get('title', 'Unknown')}")
                    st.markdown(f"**From Cache:** {'Yes' if ext.get('from_cache') else 'No'}")
                    with st.container():
                        st.text_area("Extracted Content", ext.get("content", ""), 
                                   height=150, disabled=True, key=f"source_{i}")
                    break


def display_classification(result: Dict[str, Any]):
    """Display classification details."""
    steps = result.get("pipeline_steps", {})
    classify_data = steps.get("classify", {}).get("data", {})
    
    if not classify_data:
        return
    
    st.subheader("🎯 Classification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = classify_data.get("task_type", "unknown")
        emoji = {"simple": "✨", "complex": "🧠", "research": "🔬"}.get(task_type, "❓")
        st.markdown(f"**Type:** {emoji} {task_type.title()}")
    
    with col2:
        score = classify_data.get("complexity_score", 0)
        st.markdown(f"**Complexity:** {'🔴' if score > 7 else '🟡' if score > 4 else '🟢'} {score}/10")
    
    with col3:
        needs_search = classify_data.get("needs_search", False)
        st.markdown(f"**Needs Search:** {'✅ Yes' if needs_search else '❌ No'}")
    
    st.markdown(f"**Reasoning:** {classify_data.get('reasoning', 'No reasoning provided')}")


def display_validation(result: Dict[str, Any]):
    """Display validation results."""
    validation = result.get("validation")
    if not validation:
        st.info("Validation was not performed.")
        return
    
    st.subheader("✅ Validation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = validation.get("quality_score", 0)
        color = "green" if score >= 7 else "orange" if score >= 5 else "red"
        st.markdown(f"**Quality Score:** <span style='color: {color}; font-size: 1.5rem;'>{score}/10</span>", 
                   unsafe_allow_html=True)
    
    with col2:
        accurate = validation.get("is_accurate", False)
        st.markdown(f"**Accurate:** {'✅ Yes' if accurate else '❌ No'}")
    
    with col3:
        complete = validation.get("is_complete", False)
        st.markdown(f"**Complete:** {'✅ Yes' if complete else '❌ No'}")
    
    issues = validation.get("issues", [])
    if issues:
        st.markdown("**Issues:**")
        for issue in issues:
            st.markdown(f"- ⚠️ {issue}")
    
    st.markdown(f"**Feedback:** {validation.get('feedback', 'No feedback')}")
    
    if validation.get("needs_fallback", False):
        st.warning("⚠️ Fallback was triggered - response regenerated with stronger model")
        if "fallback_response" in result:
            with st.expander("View Fallback Response"):
                st.markdown(result["fallback_response"])


# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # Load config
    config = load_configuration()
    
    if config:
        st.success("✅ Config loaded")
        
        # Provider selection
        providers = config.list_providers()
        selected_provider = st.selectbox(
            "Select Provider",
            providers,
            index=providers.index(config.current_provider) if config.current_provider in providers else 0
        )
        
        # API Key status
        st.subheader("🔑 API Keys")
        api_keys = check_api_keys("all")
        for key_name, is_set in api_keys.items():
            status = "✅ Set" if is_set else "❌ Missing"
            st.markdown(f"**{key_name}:** {status}")
        
        if not all(api_keys.values()):
            st.warning("Some API keys are missing. Check your .env file.")
        
        # Model info
        st.subheader("🤖 Models")
        try:
            st.markdown(f"**Orchestrator:** `{config.get_model('orchestrator')}`")
            st.markdown(f"**Worker:** `{config.get_model('worker')}`")
        except Exception as e:
            st.error(f"Error loading models: {e}")
    else:
        st.error("❌ Failed to load config")
        selected_provider = "nvidia"
    
    st.divider()
    
    # Settings
    st.subheader("🔧 Settings")
    max_results = st.slider("Max Search Results", 1, 10, 3)
    validate_output = st.toggle("Validate Output", value=True)
    quality_threshold = st.slider("Quality Threshold", 1, 10, 7)
    
    st.divider()
    
    # Query History
    st.subheader("📝 History")
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.expander(f"{item['query'][:40]}..."):
                st.markdown(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Provider:** {item['provider']}")
                st.markdown(f"**Type:** {item.get('task_type', 'unknown')}")
                if st.button("Load Result", key=f"history_{i}"):
                    st.session_state.last_result = item["result"]
                    st.rerun()
    else:
        st.info("No queries yet")
    
    st.divider()
    st.caption("Research Pipeline v0.3.0")


# Main content
st.markdown('<div class="main-header">🔬 Research Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent query routing with web research, content extraction, and quality validation</div>', unsafe_allow_html=True)

# Query input
st.subheader("💬 Enter Your Query")
query = st.text_area(
    "What would you like to research?",
    placeholder="e.g., What are the latest developments in quantum computing in 2025?",
    height=100
)

col1, col2 = st.columns([1, 6])
with col1:
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=False):
        st.session_state.query_history = []
        st.session_state.last_result = None
        st.rerun()

# Execute pipeline
if run_button and query:
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Running research pipeline..."):
            try:
                result = run_research_pipeline(
                    query=query,
                    max_results=max_results,
                    provider=selected_provider,
                    validate=validate_output,
                    threshold=quality_threshold
                )
                
                st.session_state.last_result = result
                
                # Add to history
                steps = result.get("pipeline_steps", {})
                classify_data = steps.get("classify", {}).get("data", {})
                st.session_state.query_history.append({
                    "query": query,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "provider": selected_provider,
                    "task_type": classify_data.get("task_type", "unknown"),
                    "result": result
                })
                
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.exception(e)

# Display results
if st.session_state.last_result:
    result = st.session_state.last_result
    
    st.divider()
    
    # Pipeline visualization
    display_pipeline_steps(result.get("pipeline_steps", {}))
    
    st.divider()
    
    # Metrics
    display_metrics(result)
    
    st.divider()
    
    # Results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Answer", "📚 Sources", "📊 Details", "🔧 Raw JSON"])
    
    with tab1:
        st.markdown("### Response")
        content = result.get("content", "No content generated")
        st.markdown(content)
        
        if result.get("fallback") and "fallback_response" in result:
            st.divider()
            st.warning("This is the fallback response (original failed validation)")
    
    with tab2:
        display_sources(result)
    
    with tab3:
        display_classification(result)
        st.divider()
        display_validation(result)
        
        # Cache stats if available
        cache_stats = result.get("cache_stats")
        if cache_stats:
            st.divider()
            st.subheader("💾 Cache Statistics")
            st.markdown(f"**Hits:** {cache_stats.hits}")
            st.markdown(f"**Misses:** {cache_stats.misses}")
            st.markdown(f"**Hit Rate:** {cache_stats.hit_rate:.1%}")
            st.markdown(f"**Total Entries:** {cache_stats.entries_total}")
    
    with tab4:
        st.json(result)

# Footer
st.divider()
st.caption("""
Built with Streamlit | Research Pipeline with OpenRouter & NVIDIA NIM | 
[Documentation](https://github.com/anomalyco/opencode) | 
Issues? Check your API keys and config.yaml
""")
