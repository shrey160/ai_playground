import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    name: str
    base_url: str
    api_key_env: str
    models: Dict[str, str]
    
    @property
    def api_key(self) -> str:
        """Get API key from environment variable."""
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"Environment variable '{self.api_key_env}' not set. "
                f"Required for provider '{self.name}'."
            )
        return key


class Config:
    """Singleton configuration manager."""
    _instance: Optional["Config"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._providers: Dict[str, ProviderConfig] = {}
        self._current_provider: Optional[str] = None
        self._config_path: Optional[str] = None
        self._initialized = True
    
    def load(self, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        self._config_path = config_path
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Load providers
        for key, provider_data in data.get("providers", {}).items():
            self._providers[key] = ProviderConfig(
                name=provider_data["name"],
                base_url=provider_data["base_url"],
                api_key_env=provider_data["api_key_env"],
                models=provider_data["models"],
            )
        
        # Set default provider
        default = data.get("default_provider")
        if default and default in self._providers:
            self._current_provider = default
        elif self._providers:
            self._current_provider = next(iter(self._providers))
        
        return self
    
    @property
    def current_provider(self) -> str:
        """Get current provider name."""
        if self._current_provider is None:
            raise RuntimeError("No provider configured. Call load() first.")
        return self._current_provider
    
    @current_provider.setter
    def current_provider(self, value: str) -> None:
        """Set current provider."""
        if value not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"Unknown provider '{value}'. Available: {available}"
            )
        self._current_provider = value
    
    @property
    def provider(self) -> ProviderConfig:
        """Get current provider configuration."""
        if self._current_provider is None:
            raise RuntimeError("No provider configured. Call load() first.")
        return self._providers[self._current_provider]
    
    @property
    def base_url(self) -> str:
        """Get current provider base URL."""
        return self.provider.base_url
    
    @property
    def api_key(self) -> str:
        """Get current provider API key."""
        return self.provider.api_key
    
    def get_model(self, alias: str) -> str:
        """Get model name by alias for current provider."""
        if alias not in self.provider.models:
            available = list(self.provider.models.keys())
            raise ValueError(
                f"Unknown model alias '{alias}' for provider '{self.current_provider}'. "
                f"Available: {available}"
            )
        return self.provider.models[alias]
    
    def list_providers(self) -> list:
        """List all available provider names."""
        return list(self._providers.keys())
    
    def get_provider_config(self, name: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        if name not in self._providers:
            raise ValueError(f"Unknown provider '{name}'")
        return self._providers[name]


# Global singleton instance
_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file.
    
    Usage:
        from modules.config import load_config
        config = load_config()  # Loads config.yaml, defaults to NVIDIA
    """
    global _config
    _config = Config()
    return _config.load(config_path)


def set_provider(provider: str) -> None:
    """Set the active provider.
    
    Usage:
        from modules.config import set_provider
        set_provider("openrouter")  # Switch to OpenRouter
    """
    global _config
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    _config.current_provider = provider


def get_config() -> Config:
    """Get the current configuration.
    
    Usage:
        from modules.config import get_config
        config = get_config()
        model = config.get_model("worker")
        api_key = config.api_key
    """
    global _config
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
