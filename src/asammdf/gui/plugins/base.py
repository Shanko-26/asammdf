"""Base plugin interface for asammdf plugins"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from PySide6 import QtCore, QtWidgets


class BasePlugin(ABC):
    """Base interface for all asammdf plugins
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self):
        """Initialize base plugin properties"""
        self.name = "BasePlugin"
        self.version = "0.0.0"
        self.description = ""
        self.author = ""
        self.enabled = False
        self.main_window = None
        
    @abstractmethod
    def initialize(self, main_window: QtWidgets.QMainWindow) -> bool:
        """Initialize the plugin with main window reference
        
        Args:
            main_window: The asammdf MainWindow instance
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_menu_items(self) -> Dict[str, QtWidgets.QAction]:
        """Create menu items to be added to main window
        
        Returns:
            Dict mapping action names to QAction objects
        """
        pass
    
    @abstractmethod
    def create_widgets(self) -> Dict[str, QtWidgets.QWidget]:
        """Create widgets that can be added to MDI area
        
        Returns:
            Dict mapping widget names to QWidget objects
        """
        pass
    
    def create_toolbar_actions(self) -> List[QtWidgets.QAction]:
        """Create toolbar actions (optional)
        
        Returns:
            List of QAction objects for toolbar
        """
        return []
    
    def create_context_menu_actions(self, context: str) -> List[QtWidgets.QAction]:
        """Create context menu actions for specific contexts (optional)
        
        Args:
            context: Context identifier (e.g., "channel_tree", "plot_area")
            
        Returns:
            List of QAction objects for context menu
        """
        return []
    
    def on_file_loaded(self, file_widget: Any) -> None:
        """Called when a new file is loaded (optional)
        
        Args:
            file_widget: The FileWidget instance with the loaded MDF
        """
        pass
    
    def on_file_closed(self, file_widget: Any) -> None:
        """Called when a file is closed (optional)
        
        Args:
            file_widget: The FileWidget instance being closed
        """
        pass
    
    def get_settings_widget(self) -> Optional[QtWidgets.QWidget]:
        """Get settings widget for plugin configuration (optional)
        
        Returns:
            QWidget for plugin settings or None
        """
        return None
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of plugin
        
        This method is called when the plugin is being unloaded or
        when asammdf is closing. Clean up resources here.
        """
        pass
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information
        
        Returns:
            Dict with plugin metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "enabled": self.enabled
        }