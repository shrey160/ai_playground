import os
import json
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ModelRouter:
    """Orchestrator-based model routing system."""
    
    ORCHESTRATOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    WORKER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    
    def classify_task(self, query: str) -> Dict[str, Any]:
        """Classify task type using orchestrator."""
        system_prompt = """You are a task classifier. Analyze the user query and respond with a JSON object containing:
{
  "task_type": "simple" | "complex" | "research",
  "complexity_score": 1-10,
  "needs_search": true | false,
  "reasoning": "brief explanation"
}

Rules:
- simple: factual Q&A, definitions, basic math, direct answers
- complex: multi-step reasoning, analysis, planning, coding, creative writing
- research: requires current information, news, facts not in training data
- needs_search: true if query asks about recent events or specific real-time data"""
        
        response = self.client.chat.completions.create(
            model=self.ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this query: {query}"}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result["tokens_used"] = response.usage.total_tokens
        return result
    
    def execute_task(
        self, 
        query: str, 
        task_type: str = "simple",
        reasoning: bool = False,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute task with appropriate model based on type."""
        
        model = self.WORKER_MODEL if task_type == "simple" else self.ORCHESTRATOR_MODEL
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        kwargs = {
            "model": model,
            "messages": messages
        }
        
        if reasoning and model == self.ORCHESTRATOR_MODEL:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        
        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - start_time
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens": response.usage.total_tokens,
            "time": round(elapsed, 2)
        }
    
    def validate_output(
        self, 
        query: str, 
        response: str,
        threshold: int = 7
    ) -> Dict[str, Any]:
        """Validate output quality using orchestrator."""
        
        validation_prompt = f"""Evaluate the quality of this AI response.

Original Query: {query}

Response:
{response}

Respond with JSON:
{{
  "quality_score": 1-10,
  "is_accurate": true | false,
  "is_complete": true | false,
  "issues": ["list any issues"],
  "needs_fallback": true | false,
  "feedback": "brief assessment"
}}"""
        
        val_response = self.client.chat.completions.create(
            model=self.ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a quality validator. Be strict but fair."},
                {"role": "user", "content": validation_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        validation = json.loads(val_response.choices[0].message.content)
        validation["passed"] = validation["quality_score"] >= threshold
        return validation
    
    def route(
        self, 
        query: str,
        validate: bool = True,
        quality_threshold: int = 7,
        research_fn = None
    ) -> Dict[str, Any]:
        """Complete routing pipeline: classify → route → execute → validate."""
        
        # Step 1: Classify
        classification = self.classify_task(query)
        task_type = classification["task_type"]
        needs_search = classification["needs_search"]
        
        # Step 2: Route and execute
        if task_type == "research" or needs_search:
            if research_fn is None:
                raise ValueError("Research tasks require research_fn parameter")
            result = research_fn(query)
            result["classification"] = classification
        else:
            result = self.execute_task(
                query, 
                task_type, 
                reasoning=(task_type == "complex")
            )
            result["classification"] = classification
        
        # Step 3: Validate (optional)
        if validate and "content" in result:
            content = result["content"]
            validation = self.validate_output(query, content, quality_threshold)
            result["validation"] = validation
            
            # Fallback if needed
            if validation.get("needs_fallback", False):
                fallback = self.execute_task(query, "complex", reasoning=True)
                result["fallback_response"] = fallback["content"]
                result["fallback"] = True
            else:
                result["fallback"] = False
        
        return result
