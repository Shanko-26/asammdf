"""AI Assistant widget for asammdf MDI integration"""

import logging
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtWidgets, QtGui

logger = logging.getLogger("asammdf.plugins.aiaspro.ui")


class AIAssistantWidget(QtWidgets.QWidget):
    """Main AI Assistant widget for MDI integration
    
    This widget provides the primary interface for interacting with the AI Assistant.
    It includes a chat interface, input area, and integration with asammdf's file system.
    """
    
    # Signals for communication with other components
    query_submitted = QtCore.Signal(str)  # Emitted when user submits a query
    response_received = QtCore.Signal(str)  # Emitted when AI response is received
    file_changed = QtCore.Signal(object)  # Emitted when associated file changes
    
    def __init__(self, main_window, parent=None):
        """Initialize AI Assistant widget
        
        Args:
            main_window: Reference to asammdf main window
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.main_window = main_window
        self.file_widget = None  # Will be set when file is loaded
        self.current_query = ""
        self.is_processing = False
        
        # Setup the user interface
        self._setup_ui()
        self._connect_signals()
        self._apply_styling()
        
        logger.info("AI Assistant widget initialized")
    
    def _setup_ui(self):
        """Setup the user interface components"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title bar with plugin info and settings
        self._create_title_bar(layout)
        
        # Status indicator
        self._create_status_indicator(layout)
        
        # Chat display area
        self._create_chat_display(layout)
        
        # Input area for queries
        self._create_input_area(layout)
        
        # Progress indicator
        self._create_progress_indicator(layout)
        
        # Suggestions panel
        self._create_suggestions_panel(layout)
        
        # Initially hide progress indicator
        self.progress_bar.setVisible(False)
    
    def _create_title_bar(self, layout):
        """Create title bar with plugin info and settings button"""
        title_layout = QtWidgets.QHBoxLayout()
        
        # Plugin title and version
        title_label = QtWidgets.QLabel("🤖 AI Assistant Pro")
        title_label.setObjectName("title_label")
        title_layout.addWidget(title_label)
        
        # Status text (will show file info)
        self.status_label = QtWidgets.QLabel("No file loaded")
        self.status_label.setObjectName("status_label")
        title_layout.addWidget(self.status_label)
        
        # Stretch to push settings button to the right
        title_layout.addStretch()
        
        # Settings button
        settings_btn = QtWidgets.QPushButton("⚙️")
        settings_btn.setObjectName("settings_button")
        settings_btn.setFixedSize(30, 30)
        settings_btn.setToolTip("Open AI Assistant settings")
        settings_btn.clicked.connect(self._open_settings)
        title_layout.addWidget(settings_btn)
        
        layout.addLayout(title_layout)
    
    def _create_status_indicator(self, layout):
        """Create status indicator for AI connection"""
        status_layout = QtWidgets.QHBoxLayout()
        
        # Connection status indicator
        self.connection_indicator = QtWidgets.QLabel("●")
        self.connection_indicator.setObjectName("connection_indicator")
        self.connection_indicator.setToolTip("AI service connection status")
        status_layout.addWidget(self.connection_indicator)
        
        # Status text
        self.connection_status = QtWidgets.QLabel("Ready")
        self.connection_status.setObjectName("connection_status")
        status_layout.addWidget(self.connection_status)
        
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
    
    def _create_chat_display(self, layout):
        """Create chat display area"""
        # Chat display with scrolling
        self.chat_display = QtWidgets.QTextEdit()
        self.chat_display.setObjectName("chat_display")
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        
        # Set placeholder text
        self.chat_display.setPlaceholderText(
            "Welcome to AI Assistant Pro!\n\n"
            "Load an MDF file and start asking questions about your data:\n"
            "• \"Show me all engine-related signals\"\n"
            "• \"Plot engine RPM and speed\"\n"
            "• \"Find any anomalies in the brake pressure data\"\n"
            "• \"What's the correlation between speed and fuel consumption?\""
        )
        
        layout.addWidget(self.chat_display)
    
    def _create_input_area(self, layout):
        """Create input area for user queries"""
        input_layout = QtWidgets.QHBoxLayout()
        
        # Query input field
        self.query_input = QtWidgets.QLineEdit()
        self.query_input.setObjectName("query_input")
        self.query_input.setPlaceholderText("Ask me about your data...")
        self.query_input.returnPressed.connect(self._handle_submit)
        input_layout.addWidget(self.query_input)
        
        # Send button
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setObjectName("send_button")
        self.send_button.clicked.connect(self._handle_submit)
        self.send_button.setDefault(True)
        input_layout.addWidget(self.send_button)
        
        # Clear button
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.setObjectName("clear_button")
        clear_button.clicked.connect(self._clear_chat)
        clear_button.setToolTip("Clear chat history")
        input_layout.addWidget(clear_button)
        
        layout.addLayout(input_layout)
    
    def _create_progress_indicator(self, layout):
        """Create progress indicator for AI processing"""
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Processing query...")
        layout.addWidget(self.progress_bar)
    
    def _create_suggestions_panel(self, layout):
        """Create suggestions panel for query suggestions"""
        # Suggestions group box
        suggestions_group = QtWidgets.QGroupBox("💡 Suggestions")
        suggestions_group.setObjectName("suggestions_group")
        suggestions_layout = QtWidgets.QVBoxLayout(suggestions_group)
        
        # Suggestions will be added dynamically
        self.suggestions_layout = suggestions_layout
        
        # Initially hide suggestions
        suggestions_group.setVisible(False)
        self.suggestions_group = suggestions_group
        
        layout.addWidget(suggestions_group)
    
    def _connect_signals(self):
        """Connect internal signals and slots"""
        # Connect query submission
        self.query_submitted.connect(self._on_query_submitted)
        self.response_received.connect(self._on_response_received)
    
    def _apply_styling(self):
        """Apply custom styling to the widget"""
        self.setStyleSheet("""
            QWidget#title_label {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                margin: 4px;
            }
            
            QWidget#status_label {
                font-size: 12px;
                color: #7f8c8d;
                margin: 4px;
            }
            
            QWidget#connection_indicator {
                color: #27ae60;
                font-size: 12px;
            }
            
            QWidget#connection_status {
                font-size: 12px;
                color: #7f8c8d;
            }
            
            QTextEdit#chat_display {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: "Segoe UI", sans-serif;
                font-size: 12px;
            }
            
            QLineEdit#query_input {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 12px;
            }
            
            QLineEdit#query_input:focus {
                border: 2px solid #007AFF;
            }
            
            QPushButton#send_button {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            
            QPushButton#send_button:hover {
                background-color: #0056b3;
            }
            
            QPushButton#send_button:pressed {
                background-color: #004494;
            }
            
            QPushButton#send_button:disabled {
                background-color: #6c757d;
            }
            
            QPushButton#clear_button {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            
            QPushButton#clear_button:hover {
                background-color: #5a6268;
            }
            
            QPushButton#settings_button {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            
            QPushButton#settings_button:hover {
                background-color: #f8f9fa;
            }
            
            QProgressBar#progress_bar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
            }
            
            QProgressBar#progress_bar::chunk {
                background-color: #007AFF;
                border-radius: 3px;
            }
            
            QGroupBox#suggestions_group {
                font-weight: bold;
                border: 1px solid #ced4da;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """)
    
    # Public interface methods
    
    def set_file_widget(self, file_widget):
        """Set the associated file widget
        
        Args:
            file_widget: The asammdf FileWidget instance
        """
        self.file_widget = file_widget
        
        # Update status
        if file_widget and hasattr(file_widget, 'file_name'):
            file_name = getattr(file_widget, 'file_name', 'Unknown file')
            self.status_label.setText(f"File: {file_name}")
            
            # Show suggestions for loaded file
            self._update_suggestions_for_file()
        else:
            self.status_label.setText("No file loaded")
            self._hide_suggestions()
        
        self.file_changed.emit(file_widget)
        logger.info(f"File widget set: {getattr(file_widget, 'file_name', 'None')}")
    
    def clear_file_widget(self, file_widget=None):
        """Clear the associated file widget
        
        Args:
            file_widget: The file widget being cleared (optional)
        """
        if file_widget is None or self.file_widget == file_widget:
            self.file_widget = None
            self.status_label.setText("No file loaded")
            self._hide_suggestions()
            self.file_changed.emit(None)
            logger.info("File widget cleared")
    
    def add_message(self, role: str, content: str, timestamp: str = None):
        """Add a message to the chat display
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            timestamp: Optional timestamp string
        """
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        
        # Format message based on role
        if role == "user":
            html = f"""
            <div style="text-align: right; margin: 8px 0;">
                <span style="background-color: #007AFF; color: white; 
                           padding: 8px 12px; border-radius: 16px; 
                           display: inline-block; max-width: 70%;">
                    {content}
                </span>
                {f'<br><small style="color: #6c757d;">{timestamp}</small>' if timestamp else ''}
            </div>
            """
        elif role == "assistant":
            html = f"""
            <div style="text-align: left; margin: 8px 0;">
                <span style="background-color: #e9ecef; color: #212529; 
                           padding: 8px 12px; border-radius: 16px; 
                           display: inline-block; max-width: 70%;">
                    {content}
                </span>
                {f'<br><small style="color: #6c757d;">{timestamp}</small>' if timestamp else ''}
            </div>
            """
        elif role == "system":
            html = f"""
            <div style="text-align: center; margin: 8px 0;">
                <small style="color: #6c757d; font-style: italic;">
                    {content}
                </small>
            </div>
            """
        
        cursor.insertHtml(html)
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def set_connection_status(self, connected: bool, message: str = ""):
        """Set AI connection status
        
        Args:
            connected: Whether AI service is connected
            message: Status message
        """
        if connected:
            self.connection_indicator.setStyleSheet("color: #27ae60;")  # Green
            self.connection_status.setText(message or "Connected")
        else:
            self.connection_indicator.setStyleSheet("color: #e74c3c;")  # Red
            self.connection_status.setText(message or "Disconnected")
    
    def set_processing(self, processing: bool):
        """Set processing state
        
        Args:
            processing: Whether AI is currently processing
        """
        self.is_processing = processing
        self.progress_bar.setVisible(processing)
        self.send_button.setEnabled(not processing)
        self.query_input.setEnabled(not processing)
        
        if processing:
            self.query_input.setPlaceholderText("Processing...")
        else:
            self.query_input.setPlaceholderText("Ask me about your data...")
    
    # Private methods
    
    def _handle_submit(self):
        """Handle query submission"""
        query = self.query_input.text().strip()
        if not query or self.is_processing:
            return
        
        # Store current query
        self.current_query = query
        
        # Clear input
        self.query_input.clear()
        
        # Add user message to chat
        self.add_message("user", query)
        
        # Set processing state
        self.set_processing(True)
        
        # Emit signal for processing
        self.query_submitted.emit(query)
    
    def _clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        logger.info("Chat history cleared")
    
    def _open_settings(self):
        """Open AI Assistant settings"""
        # This would open the settings dialog
        # For now, just show a placeholder
        QtWidgets.QMessageBox.information(
            self,
            "AI Settings",
            "Settings dialog will be implemented in the next phase."
        )
    
    def _update_suggestions_for_file(self):
        """Update suggestions based on loaded file"""
        if not self.file_widget:
            return
        
        # Sample suggestions based on automotive data
        suggestions = [
            "Show me all engine-related signals",
            "Plot engine RPM and vehicle speed",
            "Find correlations between signals",
            "Analyze brake pressure data",
            "Show signal statistics",
            "List all available channels"
        ]
        
        self._show_suggestions(suggestions)
    
    def _show_suggestions(self, suggestions: list):
        """Show suggestion buttons
        
        Args:
            suggestions: List of suggestion strings
        """
        # Clear existing suggestions
        self._clear_suggestions()
        
        # Add new suggestions
        for suggestion in suggestions[:6]:  # Limit to 6 suggestions
            btn = QtWidgets.QPushButton(suggestion)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 6px 12px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: #f8f9fa;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
            btn.clicked.connect(lambda checked, s=suggestion: self._use_suggestion(s))
            self.suggestions_layout.addWidget(btn)
        
        self.suggestions_group.setVisible(True)
    
    def _hide_suggestions(self):
        """Hide suggestions panel"""
        self.suggestions_group.setVisible(False)
    
    def _clear_suggestions(self):
        """Clear all suggestion buttons"""
        while self.suggestions_layout.count():
            child = self.suggestions_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _use_suggestion(self, suggestion: str):
        """Use a suggestion as query
        
        Args:
            suggestion: The suggestion text to use
        """
        self.query_input.setText(suggestion)
        self.query_input.setFocus()
    
    @QtCore.Slot(str)
    def _on_query_submitted(self, query: str):
        """Handle query submission signal
        
        Args:
            query: The submitted query
        """
        logger.info(f"Query submitted: {query}")
        # The actual AI processing will be handled by the orchestrator
    
    @QtCore.Slot(str)
    def _on_response_received(self, response: str):
        """Handle AI response signal
        
        Args:
            response: The AI response
        """
        # Add AI response to chat
        self.add_message("assistant", response)
        
        # Clear processing state
        self.set_processing(False)
        
        logger.info("AI response received and displayed")