"""Dependency injection container for AIASPRO agents"""

import logging
from typing import Any, Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("asammdf.plugins.aiaspro.core")


@dataclass
class AIASPRODependencies:
    """Dependency injection container for AIASPRO agents
    
    This class holds all the services and components that agents need access to.
    """
    
    # Core asammdf dependencies
    mdf: Optional[Any] = None
    file_widget: Optional[Any] = None 
    main_window: Optional[Any] = None
    
    # File context
    current_file_name: Optional[str] = None
    available_channels: List[str] = field(default_factory=list)
    
    # Services (will be initialized later)
    mdf_data_service: Optional[Any] = None
    plotting_service: Optional[Any] = None
    analytics_service: Optional[Any] = None
    requirements_service: Optional[Any] = None
    
    # Configuration
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "provider": "openai",
        "model": "gpt-4o-mini", 
        "api_key": None,
        "temperature": 0.1,
        "max_tokens": 4096
    })
    
    # Additional fields
    user_settings: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize dependencies after dataclass creation"""
        logger.info("Dependencies container initialized")
    
    def update_for_file(self, file_widget):
        """Update dependencies when a new file is loaded
        
        Args:
            file_widget: The asammdf FileWidget instance
        """
        self.file_widget = file_widget
        self.mdf = file_widget.mdf if file_widget else None
        
        # Update file info
        if file_widget and hasattr(file_widget, 'file_name'):
            self.current_file_name = file_widget.file_name
        else:
            self.current_file_name = None
        
        # Update available channels
        self._update_available_channels()
        
        # Reinitialize services with new MDF
        if self.mdf:
            self._initialize_services()
        
        logger.info(f"Dependencies updated for file: {self.current_file_name}")
    
    def _update_available_channels(self):
        """Update the list of available channels"""
        self.available_channels = []
        
        if self.mdf and hasattr(self.mdf, 'channels_db'):
            try:
                self.available_channels = list(self.mdf.channels_db)
                logger.debug(f"Updated available channels: {len(self.available_channels)} channels")
            except Exception as e:
                logger.warning(f"Error updating channels: {e}")
    
    def _initialize_services(self):
        """Initialize services that depend on MDF data"""
        if not self.mdf:
            return
        
        try:
            # Only initialize services if MDF is available
            # Import services locally to avoid circular imports
            from ..services.mdf_data_service import MDFDataService
            self.mdf_data_service = MDFDataService(self.mdf)
            
            logger.debug("MDF data service initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize MDF data service: {e}")
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context for agents
        
        Returns:
            Dictionary with current state context
        """
        return {
            "has_file": self.mdf is not None,
            "file_name": self.current_file_name,
            "channel_count": len(self.available_channels),
            "available_channels": self.available_channels[:50],  # Limit for context
            "llm_provider": self.llm_config.get("provider", "openai"),
            "session_id": self.session_id
        }
    
    def is_ready(self) -> bool:
        """Check if dependencies are ready for AI operations
        
        Returns:
            True if basic dependencies are available
        """
        # Check if we have an API key configured
        api_key = self.llm_config.get("api_key")
        if not api_key or api_key == "your-api-key-here":
            return False
        
        # Check if we have a valid LLM provider
        provider = self.llm_config.get("provider")
        if provider not in ["openai", "azure", "local"]:
            return False
        
        return True
    
    def get_llm_config_for_agent(self) -> str:
        """Get LLM configuration string for PydanticAI
        
        Returns:
            LLM configuration string (e.g., "openai:gpt-4o-mini")
        """
        provider = self.llm_config.get("provider", "openai")
        model = self.llm_config.get("model", "gpt-4o-mini")
        
        return f"{provider}:{model}"
    
    def set_session_id(self, session_id: str):
        """Set session ID for tracking
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        logger.debug(f"Session ID set: {session_id}")
    
    def clear_file_context(self):
        """Clear file-specific context"""
        self.file_widget = None
        self.mdf = None
        self.current_file_name = None
        self.available_channels = []
        self.mdf_data_service = None
        
        logger.info("File context cleared")
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"AIASPRODependencies("
                f"file={self.current_file_name}, "
                f"channels={len(self.available_channels)}, "
                f"llm={self.llm_config.get('provider')}:{self.llm_config.get('model')}, "
                f"ready={self.is_ready()})")


# Legacy pydantic model (not needed with dataclass but kept for compatibility)
try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda **kwargs: None

if HAS_PYDANTIC:
    class AIASPRODependenciesModel(BaseModel):
        """Pydantic model for dependency validation"""
        
        class Config:
            arbitrary_types_allowed = True
        
        # Core dependencies
        mdf: Optional[Any] = Field(default=None, description="MDF object")
        file_widget: Optional[Any] = Field(default=None, description="FileWidget instance")
        main_window: Optional[Any] = Field(default=None, description="Main window instance")
        
        # Services
        mdf_data_service: Optional[Any] = Field(default=None, description="MDF data service")
        plotting_service: Optional[Any] = Field(default=None, description="Plotting service")
        analytics_service: Optional[Any] = Field(default=None, description="Analytics service")
        requirements_service: Optional[Any] = Field(default=None, description="Requirements service")
        
        # Configuration
        llm_config: Dict[str, Any] = Field(default_factory=lambda: {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": None,
            "temperature": 0.1,
            "max_tokens": 4096
        })
        user_settings: Dict[str, Any] = Field(default_factory=dict)
        
        # State
        current_file_name: Optional[str] = Field(default=None)
        available_channels: list = Field(default_factory=list)
        session_id: Optional[str] = Field(default=None)
        
        def is_ready(self) -> bool:
            """Check if dependencies are ready"""
            api_key = self.llm_config.get("api_key")
            if not api_key or api_key == "your-api-key-here":
                return False
            
            provider = self.llm_config.get("provider")
            if provider not in ["openai", "azure", "local"]:
                return False
            
            return True


def create_dependencies(**kwargs) -> AIASPRODependencies:
    """Factory function to create dependencies container
    
    Args:
        **kwargs: Dependency values
        
    Returns:
        Dependencies container instance
    """
    return AIASPRODependencies(**kwargs)


def load_dependencies_from_config(config_path: Optional[Path] = None) -> AIASPRODependencies:
    """Load dependencies from configuration file
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Dependencies container with loaded configuration
    """
    try:
        # Import config locally to avoid circular imports
        import sys
        from pathlib import Path
        
        # Add the config module to path
        config_file = Path(__file__).parent.parent / "config.py"
        if config_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("aiaspro_config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Create config instance and load
            config = config_module.AIASPROConfig()
            config.load(config_path)
            
            # Create dependencies with loaded config
            return create_dependencies(
                llm_config=config.llm.dict(),
                user_settings={
                    "enable_auto_analysis": config.enable_auto_analysis,
                    "cache_results": config.cache_results,
                    "user_requirements": config.user_requirements,
                    "theme": config.theme,
                    "log_level": config.log_level
                }
            )
    except Exception as e:
        logger.warning(f"Could not load dependencies from config: {e}")
    
    # Return default dependencies if loading fails
    return create_dependencies()