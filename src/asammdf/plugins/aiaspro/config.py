"""Configuration management for AIASPRO plugin"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for when pydantic is not installed
    logging.warning("pydantic not available, using simple config")
    BaseModel = object
    Field = lambda **kwargs: None

logger = logging.getLogger("asammdf.plugins.aiaspro.config")


class LLMConfig:
    """LLM configuration"""
    
    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider", "openai")
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.api_key = kwargs.get("api_key", None)
        self.endpoint = kwargs.get("endpoint", None)
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 4096)
    
    def dict(self):
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def update(self, data: dict):
        """Update from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class AIASPROConfig:
    """Main AIASPRO configuration"""
    
    def __init__(self):
        """Initialize configuration with defaults"""
        self.llm = LLMConfig()
        self.enable_auto_analysis = True
        self.cache_results = True
        self.user_requirements = ""
        self.theme = "auto"  # auto, light, dark
        self.log_level = "INFO"
        
        # Paths
        self.config_dir = Path.home() / ".asammdf" / "plugins" / "aiaspro"
        self.config_file = self.config_dir / "config.json"
        self.cache_dir = self.config_dir / "cache"
        
    def load(self, config_path: Optional[Path] = None) -> None:
        """Load configuration from file
        
        Args:
            config_path: Optional custom config path
        """
        if config_path:
            self.config_file = config_path
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._update_from_dict(data)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        else:
            logger.info("No config file found, using defaults")
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file
        
        Args:
            config_path: Optional custom config path
        """
        if config_path:
            self.config_file = config_path
        
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            data = self._to_dict()
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary
        
        Args:
            data: Configuration dictionary
        """
        # Update LLM config
        if "llm" in data and isinstance(data["llm"], dict):
            self.llm.update(data["llm"])
        
        # Update other settings
        for key in ["enable_auto_analysis", "cache_results", "user_requirements", "theme", "log_level"]:
            if key in data:
                setattr(self, key, data[key])
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary
        
        Returns:
            Configuration dictionary
        """
        return {
            "llm": self.llm.dict(),
            "enable_auto_analysis": self.enable_auto_analysis,
            "cache_results": self.cache_results,
            "user_requirements": self.user_requirements,
            "theme": self.theme,
            "log_level": self.log_level
        }
    
    def validate(self) -> bool:
        """Validate configuration
        
        Returns:
            True if configuration is valid
        """
        # Check required fields
        if not self.llm.provider:
            logger.error("LLM provider is required")
            return False
        
        if not self.llm.model:
            logger.error("LLM model is required")
            return False
        
        # Check API key for cloud providers
        if self.llm.provider in ["openai", "azure"] and not self.llm.api_key:
            logger.warning(f"API key not set for {self.llm.provider}")
            # Not a hard error, user might set it later
        
        return True
    
    def get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize key for filesystem
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        return self.cache_dir / f"{safe_key}.json"
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            logger.info("Cleared cache directory")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"AIASPROConfig(provider={self.llm.provider}, model={self.llm.model}, auto_analysis={self.enable_auto_analysis})"


# If pydantic is available, create proper models
if BaseModel != object:
    class LLMConfigModel(BaseModel):
        """LLM configuration model with validation"""
        provider: str = Field(default="openai", description="LLM provider")
        model: str = Field(default="gpt-4o-mini", description="Model name")
        api_key: Optional[str] = Field(default=None, description="API key")
        endpoint: Optional[str] = Field(default=None, description="Custom endpoint")
        temperature: float = Field(default=0.1, ge=0, le=2)
        max_tokens: int = Field(default=4096, ge=1)

    class AIASPROConfigModel(BaseModel):
        """Main AIASPRO configuration model with validation"""
        llm: LLMConfigModel = Field(default_factory=LLMConfigModel)
        enable_auto_analysis: bool = Field(default=True)
        cache_results: bool = Field(default=True)
        user_requirements: str = Field(default="")
        theme: str = Field(default="auto", pattern="^(auto|light|dark)$")
        log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")