"""AI Assistant Pro (AIASPRO) plugin for asammdf"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6 import QtCore, QtWidgets, QtGui

# Import from parent package - this will work when loaded by plugin manager
import sys
if __name__ == "__main__":
    # For testing standalone
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asammdf.gui.plugins.base import BasePlugin
from .config import AIASPROConfig

logger = logging.getLogger("asammdf.plugins.aiaspro")


class AIASPROPlugin(BasePlugin):
    """AI Assistant Pro plugin for intelligent automotive data analysis"""
    
    def __init__(self):
        """Initialize AIASPRO plugin"""
        super().__init__()
        self.name = "AI Assistant Pro"
        self.version = "0.1.0"
        self.description = "AI-powered automotive data analysis"
        self.author = "AIASPRO Team"
        
        # Plugin state
        self.config = AIASPROConfig()
        self.assistant_widget = None
        self.menu_actions = {}
        
    def initialize(self, main_window: QtWidgets.QMainWindow) -> bool:
        """Initialize AIASPRO plugin
        
        Args:
            main_window: The asammdf MainWindow instance
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self.main_window = main_window
            
            # Load configuration
            self.config.load()
            logger.info(f"Loaded configuration: LLM provider={self.config.llm.provider}")
            
            # Setup is successful
            logger.info("AIASPRO plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AIASPRO: {e}", exc_info=True)
            return False
    
    def create_menu_items(self) -> Dict[str, QtWidgets.QAction]:
        """Create AI Assistant menu items
        
        Returns:
            Dict of menu actions
        """
        actions = {}
        
        # Open AI Assistant action
        open_action = QtWidgets.QAction("Open AI Assistant", self.main_window)
        open_action.setShortcut("Ctrl+Shift+A")
        open_action.setStatusTip("Open the AI Assistant window")
        open_action.triggered.connect(self._open_assistant)
        actions["open_assistant"] = open_action
        
        # Quick Query action
        quick_action = QtWidgets.QAction("Quick AI Query", self.main_window)
        quick_action.setShortcut("Ctrl+Shift+Q")
        quick_action.setStatusTip("Open quick AI query dialog")
        quick_action.triggered.connect(self._quick_query)
        actions["quick_query"] = quick_action
        
        # Separator
        separator = QtWidgets.QAction(self.main_window)
        separator.setSeparator(True)
        actions["separator"] = separator
        
        # Settings action
        settings_action = QtWidgets.QAction("AI Settings...", self.main_window)
        settings_action.setStatusTip("Configure AI Assistant settings")
        settings_action.triggered.connect(self._open_settings)
        actions["settings"] = settings_action
        
        self.menu_actions = actions
        return actions
    
    def create_widgets(self) -> Dict[str, QtWidgets.QWidget]:
        """Create AI Assistant widgets
        
        Returns:
            Dict of widgets
        """
        widgets = {}
        
        # Create AI Assistant widget on demand
        if not self.assistant_widget:
            try:
                from .ui.assistant_widget import AIAssistantWidget
                self.assistant_widget = AIAssistantWidget(self.main_window)
                self.assistant_widget.setObjectName("AIAssistantWidget")
            except ImportError as e:
                logger.warning(f"Could not import AIAssistantWidget: {e}")
                # Return empty dict if UI not implemented yet
                return widgets
        
        widgets["ai_assistant"] = self.assistant_widget
        return widgets
    
    def create_toolbar_actions(self) -> list:
        """Create toolbar actions
        
        Returns:
            List of toolbar actions
        """
        actions = []
        
        # AI Assistant toolbar button
        ai_action = QtWidgets.QAction(self.main_window)
        ai_action.setText("AI")
        ai_action.setToolTip("Open AI Assistant (Ctrl+Shift+A)")
        ai_action.triggered.connect(self._open_assistant)
        
        # Try to set icon if available
        icon_path = Path(__file__).parent / "resources" / "ai_icon.png"
        if icon_path.exists():
            ai_action.setIcon(QtGui.QIcon(str(icon_path)))
        
        actions.append(ai_action)
        return actions
    
    def on_file_loaded(self, file_widget: Any) -> None:
        """Called when a new file is loaded
        
        Args:
            file_widget: The FileWidget instance with the loaded MDF
        """
        logger.info(f"File loaded: {getattr(file_widget, 'file_name', 'unknown')}")
        
        # Update assistant widget if open
        if self.assistant_widget and hasattr(self.assistant_widget, 'set_file_widget'):
            self.assistant_widget.set_file_widget(file_widget)
    
    def on_file_closed(self, file_widget: Any) -> None:
        """Called when a file is closed
        
        Args:
            file_widget: The FileWidget instance being closed
        """
        logger.info(f"File closed: {getattr(file_widget, 'file_name', 'unknown')}")
        
        # Clear assistant widget if it was using this file
        if self.assistant_widget and hasattr(self.assistant_widget, 'clear_file_widget'):
            self.assistant_widget.clear_file_widget(file_widget)
    
    def get_settings_widget(self) -> Optional[QtWidgets.QWidget]:
        """Get settings widget for plugin configuration
        
        Returns:
            Settings widget or None
        """
        try:
            from .ui.settings_widget import AIASPROSettingsWidget
            return AIASPROSettingsWidget(self.config)
        except ImportError:
            # Settings widget not implemented yet
            return None
    
    def shutdown(self) -> None:
        """Clean shutdown of plugin"""
        logger.info("Shutting down AIASPRO plugin")
        
        # Save configuration
        try:
            self.config.save()
        except Exception as e:
            logger.error(f"Failed to save config on shutdown: {e}")
        
        # Clean up widgets
        if self.assistant_widget:
            self.assistant_widget.close()
            self.assistant_widget.deleteLater()
            self.assistant_widget = None
    
    # Private methods
    
    def _open_assistant(self) -> None:
        """Open AI Assistant in MDI area"""
        logger.info("Opening AI Assistant")
        
        # Check if MDI area exists
        if not hasattr(self.main_window, 'mdi_area'):
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "AI Assistant",
                "Please open a file first to use the AI Assistant."
            )
            return
        
        # Check if assistant is already open
        for window in self.main_window.mdi_area.subWindowList():
            if window.widget() and window.widget().objectName() == "AIAssistantWidget":
                window.setFocus()
                return
        
        # Create new assistant window
        try:
            widgets = self.create_widgets()
            assistant = widgets.get("ai_assistant")
            
            if not assistant:
                QtWidgets.QMessageBox.warning(
                    self.main_window,
                    "AI Assistant",
                    "Could not create AI Assistant widget. Please check the logs."
                )
                return
            
            # Connect to current file if available
            current_file = self._get_current_file_widget()
            if current_file and hasattr(assistant, 'set_file_widget'):
                assistant.set_file_widget(current_file)
            
            # Add to MDI area
            mdi_window = self.main_window.mdi_area.addSubWindow(assistant)
            mdi_window.setWindowTitle("AI Assistant Pro")
            mdi_window.setGeometry(100, 100, 600, 800)
            mdi_window.show()
            
        except Exception as e:
            logger.error(f"Failed to open AI Assistant: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to open AI Assistant: {str(e)}"
            )
    
    def _quick_query(self) -> None:
        """Open quick query dialog"""
        logger.info("Opening quick query dialog")
        
        # Simple input dialog for now
        query, ok = QtWidgets.QInputDialog.getText(
            self.main_window,
            "Quick AI Query",
            "Enter your question:",
            QtWidgets.QLineEdit.Normal,
            ""
        )
        
        if ok and query:
            # Open assistant and submit query
            self._open_assistant()
            # TODO: Submit query to assistant
    
    def _open_settings(self) -> None:
        """Open settings dialog"""
        logger.info("Opening AI settings")
        
        settings_widget = self.get_settings_widget()
        if settings_widget:
            dialog = QtWidgets.QDialog(self.main_window)
            dialog.setWindowTitle("AI Assistant Settings")
            dialog.setModal(True)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(settings_widget)
            
            # Add OK/Cancel buttons
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                # Save settings
                self.config.save()
        else:
            QtWidgets.QMessageBox.information(
                self.main_window,
                "AI Settings",
                "Settings dialog not implemented yet."
            )
    
    def _get_current_file_widget(self) -> Optional[Any]:
        """Get the currently active file widget
        
        Returns:
            Current file widget or None
        """
        if not hasattr(self.main_window, 'mdi_area'):
            return None
            
        active_window = self.main_window.mdi_area.activeSubWindow()
        if active_window:
            widget = active_window.widget()
            # Check if it's a file widget (has mdf attribute)
            if hasattr(widget, 'mdf'):
                return widget
        
        # Fall back to checking all windows
        for window in self.main_window.mdi_area.subWindowList():
            widget = window.widget()
            if widget and hasattr(widget, 'mdf'):
                return widget
        
        return None


# For testing
if __name__ == "__main__":
    # Test plugin creation
    plugin = AIASPROPlugin()
    print(f"Created plugin: {plugin.name} v{plugin.version}")
    print(f"Description: {plugin.description}")