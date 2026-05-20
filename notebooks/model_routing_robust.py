#!/usr/bin/env python
# coding: utf-8

# # Model Routing System - Usage & Robust Fallbacks
# 
# This notebook demonstrates the model routing system with two approaches:
# 
# 1. **Standard Router** (`modules/router.py`) - Uses structured JSON output for classification
# 2. **Robust Fallback** - Plain-text parsing when JSON fails or models struggle with formatting
# 
# ## Architecture
# 
# ```
# User Query
#     |
#     v
# [Task Classifier] (Orchestrator - super-120b)
#     |
#     simple  -> [Worker] executes directly
#     complex -> [Orchestrator] with reasoning
#     research -> [Search] + [Worker] synthesis
#     |
#     v
# [Validator] Quality check + fallback if needed
#     |
#     v
# Return Result
# ```
# 
# **Models Used:**
# - Orchestrator: `nvidia/nemotron-3-super-120b-a12b:free`
# - Worker: `nvidia/nemotron-3-nano-30b-a3b:free`

# In[ ]:


import os
import sys
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load project modules
sys.path.insert(0, os.path.dirname(os.getcwd()))
from modules import ModelRouter, ResearchPipeline

# Load environment variables
load_dotenv()

# Verify API key is set
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Initialize router (uses modules/router.py)
router = ModelRouter()
research = ResearchPipeline(router.client)

print("Model Routing System Initialized")
print(f"  Orchestrator: {router.ORCHESTRATOR_MODEL}")
print(f"  Worker: {router.WORKER_MODEL}")
print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")


# ## 1. Standard Router - JSON-Based Classification
# 
# The `ModelRouter` class uses structured JSON output for task classification. This is the primary approach.

# In[ ]:


# Test 1: Simple query (should route to worker)
print("Test 1: Simple Query\n" + "="*70)
result = router.route("What is the capital of France?", validate=False)

print(f"Query: What is the capital of France?")
print(f"Classification: {result['classification']['task_type']}")
print(f"Complexity: {result['classification']['complexity_score']}/10")
print(f"Model Used: {result['model']}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print(f"Answer: {result['content']}")


# In[ ]:


# Test 2: Complex query with validation (should route to orchestrator)
print("Test 2: Complex Query with Validation\n" + "="*70)
result = router.route(
    "Explain quantum entanglement and its applications in quantum computing",
    validate=True,
    quality_threshold=7
)

print(f"Query: Explain quantum entanglement...")
print(f"Classification: {result['classification']['task_type']}")
print(f"Model Used: {result['model']}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print(f"Quality Score: {result.get('validation', {}).get('quality_score', 'N/A')}/10")
print(f"Fallback Used: {result.get('fallback', False)}")
print(f"\nAnswer (first 300 chars):\n{result['content'][:300]}...")


# In[ ]:


# Test 3: Research query (search + synthesis)
print("Test 3: Research Query\n" + "="*70)
result = router.route(
    "What are the latest developments in AI in 2024?",
    research_fn=research.research,
    validate=False
)

print(f"Query: What are the latest developments in AI in 2024?")
print(f"Classification: {result['classification']['task_type']}")
print(f"Sources: {result.get('num_sources', 0)}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time']}s")
print(f"\nSources:")
for source in result.get('sources', [])[:3]:
    print(f"  - {source}")
print(f"\nAnswer (first 400 chars):\n{result['content'][:400]}...")


# ## 2. Robust Fallback - Plain-Text Parsing
# 
# When JSON parsing fails or models produce malformed output, use these robust functions that parse plain text.
# 
# ### 2a. Task Analyzer (YES/NO Classification)

# In[ ]:


def analyze_task_robust(query: str) -> Tuple[bool, str, float]:
    """
    Ask orchestrator if worker can handle this task using plain-text parsing.
    Returns: (can_worker_handle, reasoning, time_taken)
    """
    
    prompt = f"""You are a task analyzer. Decide if a smaller AI model (worker) can handle this task.

Task: {query}

Can a small AI model handle this task completely and accurately?

Reply with ONLY ONE WORD at the start of your response:
- YES - if it is simple (factual Q&A, basic math, definitions, single-step)
- NO - if it is complex (multi-step reasoning, analysis, coding, creative writing)

After your YES/NO answer, briefly explain why."""
    
    start = time.time()
    response = router.client.chat.completions.create(
        model=router.ORCHESTRATOR_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start
    
    content = response.choices[0].message.content.strip()
    first_word = content.split()[0].upper() if content else "NO"
    
    can_handle = first_word == "YES"
    
    return can_handle, content, elapsed

# Test robust analyzer
test_queries = [
    "What is the capital of France?",
    "Calculate 15 * 23",
    "Write a Python function to implement quicksort",
    "Explain the theory of relativity and its implications",
    "What is machine learning?",
]

print("Robust Task Analyzer (Plain-Text Parsing)\n" + "="*70)
for query in test_queries:
    can_handle, reasoning, elapsed = analyze_task_robust(query)
    decision = "WORKER" if can_handle else "ORCHESTRATOR"
    print(f"\nQuery: {query}")
    print(f"  Decision: {decision} ({elapsed:.2f}s)")
    print(f"  Reasoning: {reasoning[:100]}...")
    print("-"*70)


# ### 2b. Task Decomposer (Numbered List Parsing)

# In[ ]:


def decompose_task_robust(query: str) -> Tuple[List[str], str, float]:
    """
    Break complex task into numbered subtasks using regex parsing.
    Returns: (subtasks, full_response, time_taken)
    """
    
    prompt = f"""You are a task decomposer. Break this complex task into 2-4 simple subtasks.

Complex Task: {query}

Instructions:
1. Break into clear, numbered steps (1., 2., 3., etc.)
2. Each step should be simple and self-contained
3. A small AI model should handle each step independently
4. Keep each step to 1-2 sentences maximum

Example:
Task: Write a Python web scraper
1. Import required libraries (requests, BeautifulSoup)
2. Write a function to fetch HTML from a URL
3. Write a function to parse and extract data

Now break this task into steps:"""
    
    start = time.time()
    response = router.client.chat.completions.create(
        model=router.ORCHESTRATOR_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start
    
    content = response.choices[0].message.content
    
    # Parse numbered steps with multiple regex patterns
    patterns = [
        r'^\s*\d+[\.\)]\s*(.+)$',
        r'^\s*Step\s+\d+[\.:]?\s*(.+)$',
    ]
    
    subtasks = []
    for line in content.split('\n'):
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                subtasks.append(match.group(1).strip())
                break
    
    return subtasks, content, elapsed

# Test decomposition
complex_query = "Explain the theory of relativity and its implications for modern physics"

print("Robust Task Decomposer\n" + "="*70)
print(f"Query: {complex_query}\n")

subtasks, full_response, elapsed = decompose_task_robust(complex_query)

print(f"Decomposition ({elapsed:.2f}s):\n")
print("Full Response:")
print(full_response)
print("\nParsed Subtasks:")
for i, subtask in enumerate(subtasks, 1):
    print(f"  {i}. {subtask}")

if not subtasks:
    print("  WARNING: No subtasks parsed - using fallback")


# ### 2c. Worker Pool Execution

# In[ ]:


def execute_with_worker(task: str, system_message: Optional[str] = None) -> Dict:
    """Execute a single task with the worker model."""
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": task})
    
    start = time.time()
    response = router.client.chat.completions.create(
        model=router.WORKER_MODEL,
        messages=messages
    )
    elapsed = time.time() - start
    
    return {
        "task": task,
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "time": round(elapsed, 2)
    }

def execute_batch(tasks: List[str], max_workers: int = 3) -> List[Dict]:
    """Execute multiple tasks in parallel."""
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_with_worker, task): i
                  for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                results.append((idx, {"error": str(e)}))
    
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

# Test worker execution
print("Worker Execution\n" + "="*70)

# Single task
simple_task = "What is the capital of Japan?"
print(f"Single Task: {simple_task}")
result = execute_with_worker(simple_task)
print(f"  Result: {result['content']}")
print(f"  Time: {result['time']}s, Tokens: {result['tokens']}")
print()

# Batch execution
batch_tasks = [
    "What is 2+2?",
    "Define photosynthesis",
    "Who wrote Romeo and Juliet?"
]
print(f"Batch Tasks ({len(batch_tasks)} tasks):")
batch_results = execute_batch(batch_tasks)
for i, res in enumerate(batch_results, 1):
    print(f"  {i}. {res['task']}")
    print(f"     -> {res['content'][:60]}... ({res['time']}s)")


# ### 2d. Result Synthesizer

# In[ ]:


def synthesize_results_robust(query: str, subtask_results: List[Dict]) -> Dict:
    """Combine subtask results into a coherent final answer."""
    
    context = "\n\n".join([
        f"Subtask {i+1}: {res['task']}\nResult: {res['content']}"
        for i, res in enumerate(subtask_results)
    ])
    
    prompt = f"""You are a result synthesizer. Combine the following subtask results into a single, coherent answer.

Original Question: {query}

Subtask Results:
{context}

Instructions:
1. Synthesize into a well-structured, comprehensive answer
2. Maintain logical flow between sections
3. Remove redundancy
4. Ensure the answer directly addresses the original question"""
    
    start = time.time()
    response = router.client.chat.completions.create(
        model=router.ORCHESTRATOR_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start
    
    return {
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "time": round(elapsed, 2),
        "num_subtasks": len(subtask_results)
    }

# Test synthesis
print("Result Synthesizer\n" + "="*70)

sample_subtasks = [
    {"task": "Explain special relativity", "content": "Special relativity states that the laws of physics are the same for all observers in uniform motion. It introduced E=mc2."},
    {"task": "Explain general relativity", "content": "General relativity extends this to include gravity, describing it as curvature of spacetime caused by mass and energy."},
    {"task": "List modern physics implications", "content": "Applications include GPS satellite corrections, black hole understanding, and cosmological models."}
]

original_query = "Explain the theory of relativity and its implications"
result = synthesize_results_robust(original_query, sample_subtasks)

print(f"Original Query: {original_query}\n")
print(f"Synthesized Answer ({result['time']}s, {result['tokens']} tokens):\n")
print(result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"])


# ## 3. Complete Robust Pipeline
# 
# Putting it all together: analyze -> route -> (decompose -> execute -> synthesize) -> return

# In[ ]:


def route_and_execute_robust(query: str, verbose: bool = True) -> Dict:
    """
    Complete robust routing pipeline using plain-text parsing.
    Falls back to standard router if plain-text approach fails.
    """
    
    trace = {"query": query, "steps": []}
    total_start = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ROBUST ROUTING PIPELINE")
        print(f"Query: {query}")
        print(f"{'='*80}")
    
    # Step 1: Analyze with robust parser
    if verbose:
        print("\n[Step 1] Analyzing task (plain-text)...")
    
    try:
        can_handle, reasoning, analysis_time = analyze_task_robust(query)
        trace["steps"].append({
            "step": "analyze_robust",
            "can_worker_handle": can_handle,
            "reasoning": reasoning,
            "time": analysis_time
        })
        
        if verbose:
            decision = "WORKER can handle" if can_handle else "Needs decomposition"
            print(f"  {decision} ({analysis_time:.2f}s)")
            print(f"  Reasoning: {reasoning[:120]}...")
    
    except Exception as e:
        # Fallback to standard router
        if verbose:
            print(f"  Robust analyzer failed: {e}")
            print("  Falling back to standard router...")
        
        result = router.route(query, validate=False)
        result["approach"] = "fallback_standard"
        result["total_time"] = round(time.time() - total_start, 2)
        return result
    
    # Step 2: Route and Execute
    if can_handle:
        # Simple task - worker handles directly
        if verbose:
            print("\n[Step 2] Executing with WORKER...")
        
        result = execute_with_worker(query)
        trace["steps"].append({
            "step": "execute_worker",
            "model": router.WORKER_MODEL,
            "time": result["time"],
            "tokens": result["tokens"]
        })
        
        final_result = {
            "content": result["content"],
            "model_used": router.WORKER_MODEL,
            "tokens": result["tokens"],
            "approach": "direct_worker"
        }
        
        if verbose:
            print(f"  Complete ({result['time']}s, {result['tokens']} tokens)")
    
    else:
        # Complex task - decompose and execute
        if verbose:
            print("\n[Step 2] Decomposing task...")
        
        subtasks, decomp_response, decomp_time = decompose_task_robust(query)
        trace["steps"].append({
            "step": "decompose",
            "num_subtasks": len(subtasks),
            "time": decomp_time
        })
        
        if verbose:
            print(f"  Found {len(subtasks)} subtasks ({decomp_time:.2f}s)")
            for i, subtask in enumerate(subtasks, 1):
                print(f"    {i}. {subtask[:80]}...")
        
        # Fallback: if no subtasks parsed, use original query
        if not subtasks:
            if verbose:
                print("  WARNING: No subtasks parsed, using fallback...")
            subtasks = [query]
        
        # Execute subtasks with worker pool
        if verbose:
            print(f"\n[Step 3] Executing {len(subtasks)} subtasks with WORKER POOL...")
        
        subtask_results = execute_batch(subtasks)
        trace["steps"].append({
            "step": "execute_subtasks",
            "num_subtasks": len(subtasks),
            "total_time": sum(r.get("time", 0) for r in subtask_results),
            "total_tokens": sum(r.get("tokens", 0) for r in subtask_results)
        })
        
        if verbose:
            for i, res in enumerate(subtask_results):
                status = "OK" if "error" not in res else "ERROR"
                print(f"  {status} Subtask {i+1}: {res.get('time', 0)}s, {res.get('tokens', 0)} tokens")
        
        # Synthesize results
        if verbose:
            print("\n[Step 4] Synthesizing results with ORCHESTRATOR...")
        
        synthesis = synthesize_results_robust(query, subtask_results)
        trace["steps"].append({
            "step": "synthesize",
            "time": synthesis["time"],
            "tokens": synthesis["tokens"]
        })
        
        if verbose:
            print(f"  Complete ({synthesis['time']}s, {synthesis['tokens']} tokens)")
        
        final_result = {
            "content": synthesis["content"],
            "model_used": f"{router.ORCHESTRATOR_MODEL} (planning) + {router.WORKER_MODEL} (execution)",
            "tokens": sum(r.get("tokens", 0) for r in subtask_results) + synthesis["tokens"],
            "approach": "decomposed",
            "num_subtasks": len(subtasks),
            "subtask_results": subtask_results
        }
    
    total_time = time.time() - total_start
    final_result["total_time"] = round(total_time, 2)
    final_result["trace"] = trace
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FINAL RESULT ({total_time:.2f}s total)")
        print(f"{'='*80}")
        print(final_result["content"])
        print(f"\n{'='*80}\n")
    
    return final_result

# Test with both simple and complex queries
test_cases = [
    "What is the capital of France?",
    "Explain the theory of relativity and its implications for modern physics",
]

for query in test_cases:
    route_and_execute_robust(query)


# ## 4. Research Integration with Robust Router
# 
# Combine the robust router with the research pipeline for tasks requiring web search.

# In[ ]:


def research_with_robust_router(query: str, max_results: int = 3) -> Dict:
    """
    Research pipeline that uses the robust router for task analysis,
    then searches the web and synthesizes results.
    """
    
    print(f"\n{'='*80}")
    print(f"RESEARCH WITH ROBUST ROUTER")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    # Step 1: Analyze if research is needed
    print("\n[Step 1] Analyzing if research is needed...")
    can_handle, reasoning, _ = analyze_task_robust(query)
    
    # Research queries typically need search regardless
    print(f"  Task analysis: {'Simple' if can_handle else 'Complex'} task")
    
    # Step 2: Search
    print(f"\n[Step 2] Searching web (max {max_results} results)...")
    search_results = research.search(query, max_results=max_results)
    print(f"  Found {len(search_results)} results")
    
    # Step 3: Extract content
    print("\n[Step 3] Extracting content from sources...")
    contexts = []
    for result in search_results:
        content = research.extract_content(result['href'])
        if content:
            contexts.append({
                "source": result['href'],
                "title": result.get('title', 'Unknown'),
                "content": content
            })
            print(f"  Extracted: {result['href'][:60]}...")
    
    print(f"  Successfully extracted {len(contexts)} sources")
    
    # Step 4: Route content synthesis
    if can_handle and len(contexts) <= 2:
        # Simple task with few sources - worker can handle
        print("\n[Step 4] Synthesizing with WORKER (simple task)...")
        context_text = "\n\n".join([f"Source: {c['source']}\n{c['content']}" for c in contexts])
        
        prompt = f"""Based on these search results, answer the question concisely.

Question: {query}

Search Results:
{context_text}

Provide a clear, factual answer."""
        
        result = execute_with_worker(prompt)
        model_used = router.WORKER_MODEL
        
    else:
        # Complex task or many sources - use orchestrator
        print("\n[Step 4] Synthesizing with ORCHESTRATOR (complex task)...")
        
        # Decompose research synthesis
        subtasks = [
            f"Summarize key findings about: {query}",
            f"Analyze implications and trends for: {query}",
        ]
        
        # Add source summaries as subtasks
        for ctx in contexts[:3]:
            subtasks.append(f"Summarize this source: {ctx['title']}")
        
        subtask_results = execute_batch(subtasks)
        result = synthesize_results_robust(query, subtask_results)
        model_used = router.ORCHESTRATOR_MODEL
    
    print(f"  Complete ({result['time']}s, {result['tokens']} tokens)")
    
    final_result = {
        "query": query,
        "content": result["content"],
        "model_used": model_used,
        "sources": [c["source"] for c in contexts],
        "num_sources": len(contexts),
        "tokens": result["tokens"],
        "time": result["time"],
    }
    
    print(f"\n{'='*80}")
    print(f"RESEARCH RESULT")
    print(f"{'='*80}")
    print(final_result["content"])
    print(f"\n{'='*80}\n")
    
    return final_result

# Test research integration
research_query = "What are the latest breakthroughs in renewable energy 2024?"
research_result = research_with_robust_router(research_query, max_results=3)


# ## 5. Performance Comparison
# 
# Compare the standard JSON-based router vs. the robust plain-text approach.

# In[ ]:


import pandas as pd

def compare_approaches(queries: List[str]) -> pd.DataFrame:
    """Compare standard vs robust routing approaches."""
    
    results = []
    
    for query in queries:
        print(f"Testing: {query[:50]}...")
        
        # Standard approach
        try:
            start = time.time()
            std_result = router.route(query, validate=False)
            std_time = time.time() - start
            std_approach = "standard"
        except Exception as e:
            std_time = -1
            std_approach = f"error: {str(e)[:30]}"
        
        # Robust approach
        try:
            start = time.time()
            rob_result = route_and_execute_robust(query, verbose=False)
            rob_time = time.time() - start
            rob_approach = rob_result.get("approach", "unknown")
        except Exception as e:
            rob_time = -1
            rob_approach = f"error: {str(e)[:30]}"
        
        results.append({
            "Query": query[:40] + "..." if len(query) > 40 else query,
            "Standard (s)": round(std_time, 2) if std_time > 0 else "FAIL",
            "Robust (s)": round(rob_time, 2) if rob_time > 0 else "FAIL",
            "Std Model": std_result.get("model", "N/A")[:20] if std_time > 0 else "N/A",
            "Rob Approach": rob_approach[:20] if rob_time > 0 else "N/A",
        })
    
    return pd.DataFrame(results)

# Run comparison
comparison_queries = [
    "What is 2+2?",
    "Capital of Japan?",
    "Explain machine learning",
    "Write a Python hello world",
]

print("Performance Comparison: Standard vs Robust\n" + "="*70)
df = compare_approaches(comparison_queries)
print("\n")
print(df.to_string(index=False))


# ## 6. Error Handling & Fallbacks
# 
# Demonstrate how the system handles failures gracefully.

# In[ ]:


def robust_route_with_fallbacks(query: str) -> Dict:
    """
    Route with multiple fallback layers:
    1. Try standard JSON-based router
    2. Fallback to robust plain-text router
    3. Fallback to direct orchestrator execution
    4. Final fallback: return error with suggestions
    """
    
    errors = []
    
    # Layer 1: Standard router
    try:
        print("[Layer 1] Trying standard router...")
        result = router.route(query, validate=False)
        result["routing_layer"] = "standard"
        print("  Success!")
        return result
    except Exception as e:
        errors.append(f"Standard router: {str(e)}")
        print(f"  Failed: {str(e)[:60]}...")
    
    # Layer 2: Robust router
    try:
        print("[Layer 2] Trying robust router...")
        result = route_and_execute_robust(query, verbose=False)
        result["routing_layer"] = "robust"
        print("  Success!")
        return result
    except Exception as e:
        errors.append(f"Robust router: {str(e)}")
        print(f"  Failed: {str(e)[:60]}...")
    
    # Layer 3: Direct orchestrator
    try:
        print("[Layer 3] Trying direct orchestrator...")
        start = time.time()
        response = router.client.chat.completions.create(
            model=router.ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": query}]
        )
        elapsed = time.time() - start
        print("  Success!")
        return {
            "content": response.choices[0].message.content,
            "model": router.ORCHESTRATOR_MODEL,
            "tokens": response.usage.total_tokens,
            "time": round(elapsed, 2),
            "routing_layer": "direct_orchestrator"
        }
    except Exception as e:
        errors.append(f"Direct orchestrator: {str(e)}")
        print(f"  Failed: {str(e)[:60]}...")
    
    # Layer 4: All failed
    print("[Layer 4] All routing methods failed")
    return {
        "error": "All routing methods failed",
        "errors": errors,
        "routing_layer": "failed",
        "suggestion": "Check API connectivity and model availability"
    }

# Test fallback chain
print("Testing Fallback Chain\n" + "="*70)
fallback_result = robust_route_with_fallbacks("What is the speed of light?")
print(f"\nFinal routing layer: {fallback_result.get('routing_layer', 'unknown')}")
print(f"Model used: {fallback_result.get('model', fallback_result.get('model_used', 'N/A'))}")
print(f"\nAnswer: {fallback_result.get('content', 'N/A')[:200]}...")


# ## Summary
# 
# This notebook demonstrated two complementary approaches to model routing:
# 
# ### Standard Router (`modules/router.py`)
# - **Pros**: Clean JSON classification, validation scores, automatic fallback
# - **Cons**: Requires models to produce valid JSON, can fail with formatting issues
# - **Best for**: Stable production use with well-behaved models
# 
# ### Robust Router (Plain-Text Parsing)
# - **Pros**: No JSON parsing errors, transparent execution, easy to debug, parallel worker pool
# - **Cons**: Less structured, requires regex parsing, more verbose prompts
# - **Best for**: Fallback when JSON fails, debugging, or models that struggle with structured output
# 
# ### Key Features Demonstrated
# | Feature | Implementation |
# |---------|---------------|
# | Task Classification | YES/NO parsing + JSON fallback |
# | Task Decomposition | Numbered list regex parsing |
# | Parallel Execution | ThreadPoolExecutor worker pool |
# | Result Synthesis | Orchestrator combines subtask outputs |
# | Research Integration | DDGS search + Jina AI extraction |
# | Error Handling | 4-layer fallback chain |
# | Performance Tracking | Timing + token usage for each step |
# 
# ### Next Steps
# - Convert robust router to reusable module (`modules/router_robust.py`)
# - Add caching for repeated subtasks
# - Implement streaming for real-time responses
# - Add memory/context management across sessions
