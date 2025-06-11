"""Plugin manager for asammdf"""

import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from PySide6 import QtCore, QtWidgets, QtGui

from .base import BasePlugin

logger = logging.getLogger("asammdf.plugins")


class PluginManager(QtCore.QObject):
    """Manages plugin discovery, loading, and lifecycle
    
    The PluginManager is responsible for:
    - Discovering available plugins
    - Loading and initializing plugins
    - Managing plugin lifecycle
    - Integrating plugins with the main window
    """
    
    # Signals
    plugin_loaded = QtCore.Signal(str)  # Plugin name
    plugin_unloaded = QtCore.Signal(str)  # Plugin name
    plugin_error = QtCore.Signal(str, str)  # Plugin name, error message
    
    def __init__(self, main_window: QtWidgets.QMainWindow):
        """Initialize plugin manager
        
        Args:
            main_window: The asammdf MainWindow instance
        """
        super().__init__()
        self.main_window = main_window
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_errors: Dict[str, str] = {}
        
        # Plugin search paths
        self.plugin_paths = [
            Path(__file__).parent.parent.parent / "plugins",  # Built-in plugins
            Path.home() / ".asammdf" / "plugins",  # User plugins
        ]
        
        # Ensure plugin directories exist
        for path in self.plugin_paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for path in self.plugin_paths:
            if not path.exists():
                continue
                
            for plugin_dir in path.iterdir():
                if not plugin_dir.is_dir():
                    continue
                
                # Check for manifest file
                manifest_path = plugin_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            if self._validate_manifest(manifest):
                                discovered.append(plugin_dir.name)
                    except Exception as e:
                        logger.warning(f"Invalid manifest in {plugin_dir}: {e}")
                        continue
                
                # Also check for plugin.py (legacy support)
                elif (plugin_dir / "plugin.py").exists():
                    discovered.append(plugin_dir.name)
        
        return list(set(discovered))  # Remove duplicates
    
    def _validate_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Validate plugin manifest
        
        Args:
            manifest: Plugin manifest dictionary
            
        Returns:
            bool: True if manifest is valid
        """
        required_fields = ["name", "version", "entry_point"]
        return all(field in manifest for field in required_fields)
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            bool: True if plugin loaded successfully
        """
        if plugin_name in self.plugins:
            logger.info(f"Plugin {plugin_name} is already loaded")
            return True
        
        try:
            # Find plugin path
            plugin_path = self._find_plugin_path(plugin_name)
            if not plugin_path:
                raise ValueError(f"Plugin {plugin_name} not found")
            
            # Load manifest if available
            manifest_path = plugin_path / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    entry_point = manifest.get("entry_point")
            else:
                # Legacy: assume entry point is plugin.PluginNamePlugin
                entry_point = f"asammdf.plugins.{plugin_name}.plugin:{plugin_name.upper()}Plugin"
            
            # Parse entry point
            module_path, class_name = entry_point.rsplit(":", 1)
            
            # Import module
            module = importlib.import_module(module_path)
            
            # Get plugin class
            plugin_class = getattr(module, class_name)
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Verify it's a valid plugin
            if not isinstance(plugin_instance, BasePlugin):
                raise TypeError(f"{class_name} is not a valid plugin (must inherit from BasePlugin)")
            
            # Initialize plugin
            if plugin_instance.initialize(self.main_window):
                self.plugins[plugin_name] = plugin_instance
                plugin_instance.enabled = True
                self._integrate_plugin(plugin_instance)
                
                logger.info(f"Successfully loaded plugin: {plugin_name}")
                self.plugin_loaded.emit(plugin_name)
                return True
            else:
                raise RuntimeError(f"Plugin {plugin_name} initialization failed")
                
        except Exception as e:
            error_msg = f"Failed to load plugin {plugin_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.plugin_errors[plugin_name] = error_msg
            self.plugin_error.emit(plugin_name, error_msg)
            return False
    
    def _find_plugin_path(self, plugin_name: str) -> Optional[Path]:
        """Find the path to a plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Path to plugin directory or None if not found
        """
        for path in self.plugin_paths:
            plugin_dir = path / plugin_name
            if plugin_dir.exists() and plugin_dir.is_dir():
                return plugin_dir
        return None
    
    def _integrate_plugin(self, plugin: BasePlugin) -> None:
        """Integrate plugin with main window
        
        Args:
            plugin: Plugin instance to integrate
        """
        # Add menu items
        menu_items = plugin.create_menu_items()
        if menu_items:
            # Find or create plugin menu
            plugin_menu = self._get_plugin_menu(plugin.name)
            for action_name, action in menu_items.items():
                plugin_menu.addAction(action)
        
        # Add toolbar actions
        toolbar_actions = plugin.create_toolbar_actions()
        if toolbar_actions and hasattr(self.main_window, 'toolbar'):
            self.main_window.toolbar.addSeparator()
            for action in toolbar_actions:
                self.main_window.toolbar.addAction(action)
        
        # Connect to file events if main window supports them
        if hasattr(self.main_window, 'file_loaded'):
            self.main_window.file_loaded.connect(
                lambda fw: plugin.on_file_loaded(fw)
            )
        
        if hasattr(self.main_window, 'file_closed'):
            self.main_window.file_closed.connect(
                lambda fw: plugin.on_file_closed(fw)
            )
    
    def _get_plugin_menu(self, plugin_name: str) -> QtWidgets.QMenu:
        """Get or create menu for plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            QMenu for the plugin
        """
        # Check if plugin has its own top-level menu
        menubar = self.main_window.menuBar()
        for action in menubar.actions():
            if action.text() == plugin_name:
                return action.menu()
        
        # Create new menu
        return menubar.addMenu(plugin_name)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if plugin unloaded successfully
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} is not loaded")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            
            # Call shutdown
            plugin.shutdown()
            
            # Remove from active plugins
            del self.plugins[plugin_name]
            
            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            self.plugin_unloaded.emit(plugin_name)
            return True
            
        except Exception as e:
            error_msg = f"Failed to unload plugin {plugin_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.plugin_error.emit(plugin_name, error_msg)
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            bool: True if plugin reloaded successfully
        """
        # Unload if loaded
        if plugin_name in self.plugins:
            if not self.unload_plugin(plugin_name):
                return False
        
        # Reload module
        try:
            plugin_path = self._find_plugin_path(plugin_name)
            if plugin_path:
                # Clear module cache for this plugin
                modules_to_remove = []
                for module_name in sys.modules:
                    if module_name.startswith(f"asammdf.plugins.{plugin_name}"):
                        modules_to_remove.append(module_name)
                
                for module_name in modules_to_remove:
                    del sys.modules[module_name]
        except Exception as e:
            logger.warning(f"Error clearing module cache: {e}")
        
        # Load plugin
        return self.load_plugin(plugin_name)
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugins
        
        Returns:
            List of loaded plugin names
        """
        return list(self.plugins.keys())
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin instance
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not loaded
        """
        return self.plugins.get(plugin_name)
    
    def shutdown_all(self) -> None:
        """Shutdown all plugins
        
        Called when asammdf is closing
        """
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)