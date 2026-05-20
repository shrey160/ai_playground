"""Main entry point for the test-openrouter-nvidia application.

Usage:
    python main.py                    # Uses default provider (nvidia)
    python main.py --provider openrouter  # Uses OpenRouter
"""

import argparse
from modules import load_config, set_provider, get_config, ModelRouter, LangGraphApp


def main():
    parser = argparse.ArgumentParser(description="AI Research Pipeline")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "nvidia"],
        default="nvidia",
        help="LLM provider to use (default: nvidia)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    load_config(args.config)
    
    # Set provider
    set_provider(args.provider)
    config = get_config()
    
    print(f"✓ Using provider: {config.current_provider}")
    print(f"  Name: {config.provider.name}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Orchestrator: {config.get_model('orchestrator')}")
    print(f"  Worker: {config.get_model('worker')}")
    print()

    # Example: Simple query with router
    print("Testing ModelRouter...")
    router = ModelRouter()
    result = router.route("What is the capital of France?")
    print(f"Response: {result['content']}")
    print(f"Model: {result['model']}")
    print()

    # Example: LangGraph pipeline
    print("Testing LangGraphApp...")
    app = LangGraphApp()
    result = app.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {result['content'][:200]}...")
    print(f"Model: {result['model']}")
    print(f"Time: {result['time']}s")


if __name__ == "__main__":
    main()
