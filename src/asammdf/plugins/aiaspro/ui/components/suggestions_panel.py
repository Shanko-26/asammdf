"""Suggestions panel component for AI Assistant"""

import logging
from typing import List, Optional, Callable

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger("asammdf.plugins.aiaspro.ui.components")


class SuggestionsPanel(QtWidgets.QWidget):
    """Panel for displaying AI-generated suggestions and quick actions"""
    
    # Signals
    suggestion_clicked = QtCore.Signal(str)  # Emitted when a suggestion is clicked
    
    def __init__(self, parent=None):
        """Initialize suggestions panel
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self._setup_ui()
        self._apply_styling()
        
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header
        header = QtWidgets.QLabel("💡 Try asking:")
        header.setObjectName("suggestions_header")
        layout.addWidget(header)
        
        # Suggestions container (will hold suggestion buttons)
        self.suggestions_container = QtWidgets.QVBoxLayout()
        self.suggestions_container.setSpacing(4)
        layout.addLayout(self.suggestions_container)
        
        # Initially hidden
        self.setVisible(False)
    
    def _apply_styling(self):
        """Apply custom styling"""
        self.setStyleSheet("""
            QLabel#suggestions_header {
                font-weight: bold;
                color: #0066cc;
                margin: 8px 0px 4px 0px;
                font-size: 12px;
            }
        """)
    
    def update_suggestions(self, suggestions: List[str], 
                          categories: Optional[dict] = None):
        """Update suggestions display
        
        Args:
            suggestions: List of suggestion strings
            categories: Optional dict mapping categories to suggestion lists
        """
        # Clear existing suggestions
        self._clear_suggestions()
        
        if not suggestions and not categories:
            self.setVisible(False)
            return
        
        if categories:
            self._add_categorized_suggestions(categories)
        else:
            self._add_simple_suggestions(suggestions)
        
        self.setVisible(True)
        logger.debug(f"Updated suggestions: {len(suggestions)} items")
    
    def update_for_file(self, file_widget):
        """Update suggestions based on loaded file
        
        Args:
            file_widget: The loaded FileWidget instance
        """
        if not file_widget:
            self.setVisible(False)
            return
        
        # Generate file-specific suggestions
        suggestions = self._generate_file_suggestions(file_widget)
        self.update_suggestions(suggestions)
    
    def _generate_file_suggestions(self, file_widget) -> List[str]:
        """Generate suggestions based on the loaded file
        
        Args:
            file_widget: The loaded FileWidget instance
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Basic suggestions for any file
        basic_suggestions = [
            "Show me all available channels",
            "List channels by category",
            "Show file information",
            "Display signal statistics"
        ]
        suggestions.extend(basic_suggestions)
        
        # Try to get channel information for smarter suggestions
        try:
            if hasattr(file_widget, 'mdf') and file_widget.mdf:
                mdf = file_widget.mdf
                
                # Get some channel names to create specific suggestions
                if hasattr(mdf, 'channels_db'):
                    channels = list(mdf.channels_db)
                    
                    # Look for common automotive signal patterns
                    automotive_suggestions = []
                    
                    # Engine signals
                    engine_channels = [ch for ch in channels if any(
                        term in ch.lower() for term in ['engine', 'rpm', 'speed', 'throttle']
                    )]
                    if engine_channels:
                        automotive_suggestions.append("Analyze engine performance signals")
                        automotive_suggestions.append("Plot engine RPM over time")
                    
                    # Brake signals
                    brake_channels = [ch for ch in channels if any(
                        term in ch.lower() for term in ['brake', 'pressure']
                    )]
                    if brake_channels:
                        automotive_suggestions.append("Show brake pressure analysis")
                    
                    # Temperature signals
                    temp_channels = [ch for ch in channels if any(
                        term in ch.lower() for term in ['temp', 'temperature', 'cool']
                    )]
                    if temp_channels:
                        automotive_suggestions.append("Monitor temperature signals")
                    
                    # Vehicle dynamics
                    dynamics_channels = [ch for ch in channels if any(
                        term in ch.lower() for term in ['accel', 'velocity', 'gear', 'wheel']
                    )]
                    if dynamics_channels:
                        automotive_suggestions.append("Analyze vehicle dynamics")
                    
                    suggestions.extend(automotive_suggestions)
                    
                    # Add some general analysis suggestions
                    if len(channels) > 10:
                        suggestions.extend([
                            "Find correlations between signals",
                            "Detect anomalies in the data",
                            "Create a dashboard view"
                        ])
        
        except Exception as e:
            logger.warning(f"Error generating file-specific suggestions: {e}")
        
        return suggestions[:8]  # Limit to 8 suggestions
    
    def _add_simple_suggestions(self, suggestions: List[str]):
        """Add suggestions as simple buttons
        
        Args:
            suggestions: List of suggestion strings
        """
        for suggestion in suggestions:
            self._create_suggestion_button(suggestion)
    
    def _add_categorized_suggestions(self, categories: dict):
        """Add suggestions organized by categories
        
        Args:
            categories: Dict mapping category names to suggestion lists
        """
        for category, category_suggestions in categories.items():
            # Add category header
            if len(categories) > 1:  # Only show headers if multiple categories
                category_label = QtWidgets.QLabel(f"📂 {category}")
                category_label.setStyleSheet("""
                    font-weight: bold;
                    color: #495057;
                    margin: 8px 0px 4px 0px;
                    font-size: 11px;
                """)
                self.suggestions_container.addWidget(category_label)
            
            # Add suggestions for this category
            for suggestion in category_suggestions:
                self._create_suggestion_button(suggestion)
    
    def _create_suggestion_button(self, suggestion: str):
        """Create a clickable suggestion button
        
        Args:
            suggestion: The suggestion text
        """
        button = QtWidgets.QPushButton(suggestion)
        button.setObjectName("suggestion_button")
        button.setStyleSheet("""
            QPushButton#suggestion_button {
                text-align: left;
                padding: 8px 12px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                color: #495057;
                font-size: 11px;
                margin: 1px;
            }
            QPushButton#suggestion_button:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton#suggestion_button:pressed {
                background-color: #dee2e6;
            }
        """)
        
        # Connect click to signal
        button.clicked.connect(lambda: self.suggestion_clicked.emit(suggestion))
        
        # Add to layout
        self.suggestions_container.addWidget(button)
    
    def _clear_suggestions(self):
        """Clear all suggestion widgets"""
        while self.suggestions_container.count():
            child = self.suggestions_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def clear(self):
        """Clear all suggestions and hide panel"""
        self._clear_suggestions()
        self.setVisible(False)


class QuickActionsPanel(QtWidgets.QWidget):
    """Panel for quick actions and shortcuts"""
    
    # Signals
    action_triggered = QtCore.Signal(str, dict)  # action_name, parameters
    
    def __init__(self, parent=None):
        """Initialize quick actions panel
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self._setup_ui()
        self._setup_default_actions()
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header
        header = QtWidgets.QLabel("⚡ Quick Actions")
        header.setStyleSheet("""
            font-weight: bold;
            color: #0066cc;
            margin: 8px 0px 4px 0px;
            font-size: 12px;
        """)
        layout.addWidget(header)
        
        # Actions container
        self.actions_container = QtWidgets.QHBoxLayout()
        self.actions_container.setSpacing(8)
        layout.addLayout(self.actions_container)
    
    def _setup_default_actions(self):
        """Setup default quick actions"""
        actions = [
            ("📊", "Plot Signals", "plot_signals", {}),
            ("📈", "Statistics", "show_statistics", {}),
            ("🔍", "Search", "search_channels", {}),
            ("📋", "List All", "list_channels", {}),
        ]
        
        for icon, tooltip, action_name, params in actions:
            self._create_action_button(icon, tooltip, action_name, params)
    
    def _create_action_button(self, icon: str, tooltip: str, 
                             action_name: str, parameters: dict):
        """Create a quick action button
        
        Args:
            icon: Button icon (emoji or text)
            tooltip: Button tooltip
            action_name: Action identifier
            parameters: Action parameters
        """
        button = QtWidgets.QPushButton(icon)
        button.setToolTip(tooltip)
        button.setFixedSize(40, 40)
        button.setStyleSheet("""
            QPushButton {
                border: 1px solid #dee2e6;
                border-radius: 20px;
                background-color: #f8f9fa;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #007AFF;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """)
        
        # Connect to action
        button.clicked.connect(
            lambda: self.action_triggered.emit(action_name, parameters)
        )
        
        self.actions_container.addWidget(button)
    
    def add_custom_action(self, icon: str, tooltip: str, 
                         action_name: str, parameters: dict = None):
        """Add a custom action button
        
        Args:
            icon: Button icon
            tooltip: Button tooltip
            action_name: Action identifier
            parameters: Action parameters (optional)
        """
        self._create_action_button(
            icon, tooltip, action_name, parameters or {}
        )