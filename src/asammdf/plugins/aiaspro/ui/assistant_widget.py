"""AI Assistant widget for asammdf MDI integration"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Any

from PySide6 import QtCore, QtWidgets, QtGui

logger = logging.getLogger("asammdf.plugins.aiaspro.ui")


class AIAssistantWidget(QtWidgets.QWidget):
    """Main AI Assistant widget for MDI integration
    
    This widget provides a chat-like interface for interacting with
    AI agents that can analyze automotive data.
    """
    
    # Signals
    query_submitted = QtCore.Signal(str)
    response_received = QtCore.Signal(str)
    # Required by MDI area (must match MdiSubWindow signature)
    resized = QtCore.Signal(object, object, object)  # self, new_size, old_size
    moved = QtCore.Signal(object, object, object)    # self, new_position, old_position
    
    def __init__(self, main_window, parent=None):
        """Initialize AI Assistant widget
        
        Args:
            main_window: Reference to asammdf MainWindow
            parent: Parent widget
        """
        super().__init__(parent)
        self.main_window = main_window
        self.file_widget = None  # Current file widget
        self.ai_system = None  # Will be initialized later
        
        self._setup_ui()
        self._connect_signals()
        self._init_ai_system()
    
    def _setup_ui(self):
        """Setup the UI components"""
        self.setMinimumSize(400, 600)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title bar
        title_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("🤖 AI Assistant Pro")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: #2c3e50;
                padding: 8px;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Status indicator
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #27ae60;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        title_layout.addWidget(self.status_label)
        
        layout.addLayout(title_layout)
        
        # File info
        self.file_info = QtWidgets.QLabel("No file loaded")
        self.file_info.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 6px;
                border-radius: 4px;
                font-size: 12px;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.file_info)
        
        # Chat display
        self.chat_display = QtWidgets.QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.chat_display, 1)  # Expandable
        
        # Input area
        input_frame = QtWidgets.QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        input_layout = QtWidgets.QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        
        self.query_input = QtWidgets.QLineEdit()
        self.query_input.setPlaceholderText("Ask me about your automotive data...")
        self.query_input.setStyleSheet("""
            QLineEdit {
                border: none;
                font-size: 13px;
                padding: 8px;
                background: transparent;
            }
        """)
        
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        input_layout.addWidget(self.query_input, 1)
        input_layout.addWidget(self.send_button)
        layout.addWidget(input_frame)
        
        # Progress indicator
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Quick actions
        actions_frame = QtWidgets.QFrame()
        actions_layout = QtWidgets.QHBoxLayout(actions_frame)
        actions_layout.setContentsMargins(0, 4, 0, 0)
        
        # Quick action buttons
        quick_buttons = [
            ("📊 List Channels", "List all channels in the current file"),
            ("🔍 Search Signals", "Search for specific signals"),
            ("📈 Analyze Data", "Get data analysis suggestions"),
        ]
        
        for text, tooltip in quick_buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e9ecef;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                    color: #495057;
                }
                QPushButton:hover {
                    background-color: #dee2e6;
                }
            """)
            btn.clicked.connect(lambda checked, t=tooltip.split()[-1].lower(): self._quick_action(t))
            actions_layout.addWidget(btn)
        
        actions_layout.addStretch()
        layout.addWidget(actions_frame)
        
        # Add welcome message
        self._add_welcome_message()
    
    def _connect_signals(self):
        """Connect widget signals"""
        self.send_button.clicked.connect(self._handle_submit)
        self.query_input.returnPressed.connect(self._handle_submit)
        self.query_submitted.connect(self._process_query)
    
    def _init_ai_system(self):
        """Initialize AI system components"""
        try:
            # Import AI components
            from ..core.dependencies import AIASPRODependencies
            from ..core.orchestrator import AIOrchestrator
            from ..config import AIASPROConfig
            
            # Load configuration
            config = AIASPROConfig()
            config.load()
            
            # Setup dependencies
            self.deps = AIASPRODependencies(
                main_window=self.main_window,
                llm_config=config.llm.dict() if hasattr(config.llm, 'dict') else config.llm,
            )
            
            # Create orchestrator
            self.orchestrator = AIOrchestrator(self.deps)
            
            self.status_label.setText("AI Ready")
            self.status_label.setStyleSheet(self.status_label.styleSheet().replace("#27ae60", "#27ae60"))
            
            logger.info("AI system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI system: {e}", exc_info=True)
            self.status_label.setText("AI Error")
            self.status_label.setStyleSheet(self.status_label.styleSheet().replace("#27ae60", "#e74c3c"))
            
            # Show error in chat
            self._add_system_message(f"⚠️ AI system initialization failed: {str(e)}")
    
    def _add_welcome_message(self):
        """Add welcome message to chat"""
        welcome_html = """
        <div style="background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 12px; margin: 8px 0; border-radius: 4px;">
            <h4 style="margin: 0 0 8px 0; color: #1976d2;">🤖 Welcome to AI Assistant Pro!</h4>
            <p style="margin: 0; color: #424242; font-size: 13px;">
                I can help you analyze automotive measurement data. Try asking me:
            </p>
            <ul style="margin: 8px 0 0 20px; color: #424242; font-size: 12px;">
                <li>What channels are available in this file?</li>
                <li>Show me engine-related signals</li>
                <li>Analyze the Engine_Speed signal</li>
                <li>What types of data do I have?</li>
            </ul>
        </div>
        """
        self.chat_display.append(welcome_html)
    
    def _add_system_message(self, message: str):
        """Add system message to chat"""
        html = f"""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; margin: 4px 0; border-radius: 4px;">
            <p style="margin: 0; color: #856404; font-size: 12px;">
                <strong>System:</strong> {message}
            </p>
        </div>
        """
        self.chat_display.append(html)
    
    def _add_user_message(self, message: str):
        """Add user message to chat"""
        html = f"""
        <div style="text-align: right; margin: 8px 0;">
            <div style="background-color: #007AFF; color: white; padding: 8px 12px; border-radius: 16px; display: inline-block; max-width: 70%; text-align: left;">
                {message}
            </div>
        </div>
        """
        self.chat_display.append(html)
        self.chat_display.ensureCursorVisible()
    
    def _add_assistant_message(self, message: str):
        """Add assistant message to chat"""
        # Convert newlines to HTML breaks
        message = message.replace('\n', '<br>')
        
        html = f"""
        <div style="text-align: left; margin: 8px 0;">
            <div style="background-color: #f1f3f4; color: #333; padding: 8px 12px; border-radius: 16px; display: inline-block; max-width: 85%; border: 1px solid #e0e0e0;">
                {message}
            </div>
        </div>
        """
        self.chat_display.append(html)
        self.chat_display.ensureCursorVisible()
    
    def _handle_submit(self):
        """Handle query submission"""
        query = self.query_input.text().strip()
        if not query:
            return
        
        # Add to chat display
        self._add_user_message(query)
        self.query_input.clear()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.send_button.setEnabled(False)
        
        # Emit signal to process query
        self.query_submitted.emit(query)
    
    def _process_query(self, query: str):
        """Process query asynchronously"""
        if not hasattr(self, 'orchestrator') or not self.orchestrator:
            self._add_system_message("AI system not available. Please check configuration.")
            self._query_finished()
            return
        
        # Run in thread to avoid blocking UI
        self.worker_thread = QueryWorkerThread(self.orchestrator, query, self.deps)
        self.worker_thread.response_ready.connect(self._handle_response)
        self.worker_thread.error_occurred.connect(self._handle_error)
        self.worker_thread.finished.connect(self._query_finished)
        self.worker_thread.start()
    
    def _handle_response(self, response: str):
        """Handle AI response"""
        self._add_assistant_message(response)
    
    def _handle_error(self, error: str):
        """Handle AI error"""
        self._add_system_message(f"Error: {error}")
    
    def _query_finished(self):
        """Handle query completion"""
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)
    
    def _quick_action(self, action: str):
        """Handle quick action button clicks"""
        if action == "suggestions":
            self.query_input.setText("What analysis can I perform on this data?")
        elif action == "signals":
            self.query_input.setText("Search for engine signals")
        elif action == "channels":
            self.query_input.setText("List all channels")
        
        self._handle_submit()
    
    def set_file_widget(self, file_widget: Any):
        """Set the current file widget
        
        Args:
            file_widget: FileWidget instance with MDF data
        """
        self.file_widget = file_widget
        
        if file_widget and hasattr(file_widget, 'file_name'):
            file_name = Path(file_widget.file_name).name
            channel_count = len(file_widget.mdf.channels_db) if hasattr(file_widget, 'mdf') else 0
            
            self.file_info.setText(f"📁 {file_name} ({channel_count:,} channels)")
            self.file_info.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    padding: 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    color: #155724;
                }
            """)
            
            # Update dependencies
            if hasattr(self, 'deps'):
                self.deps.update_for_file(file_widget)
            
            # Add file loaded message
            self._add_system_message(f"File loaded: {file_name} with {channel_count:,} channels")
        else:
            self.file_info.setText("No file loaded")
            self.file_info.setStyleSheet("""
                QLabel {
                    background-color: #ecf0f1;
                    padding: 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    color: #7f8c8d;
                }
            """)
    
    def clear_file_widget(self, file_widget: Any):
        """Clear file widget if it matches the current one"""
        if self.file_widget == file_widget:
            self.file_widget = None
            self.file_info.setText("No file loaded")
            
            if hasattr(self, 'deps'):
                self.deps.clear_file_context()

    def resizeEvent(self, event):
        """Handle resize events and emit signal for MDI area"""
        old_size = event.oldSize()
        new_size = event.size()
        super().resizeEvent(event)
        
        # Emit with the signature expected by MDI area: (self, new_size, old_size)
        if old_size.isValid():
            self.resized.emit(self, new_size, old_size)

    def moveEvent(self, event):
        """Handle move events and emit signal for MDI area"""
        old_position = event.oldPos()
        new_position = event.pos()
        super().moveEvent(event)
        
        # Emit with the signature expected by MDI area: (self, new_position, old_position)
        self.moved.emit(self, new_position, old_position)


class QueryWorkerThread(QtCore.QThread):
    """Worker thread for processing AI queries"""
    
    response_ready = QtCore.Signal(str)
    error_occurred = QtCore.Signal(str)
    
    def __init__(self, orchestrator, query, deps):
        super().__init__()
        self.orchestrator = orchestrator
        self.query = query
        self.deps = deps
    
    def run(self):
        """Run the query processing"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the query
            result = loop.run_until_complete(
                self.orchestrator.route_and_execute(self.query)
            )
            
            # Get response text
            if hasattr(result, 'output'):
                response = str(result.output)
            elif hasattr(result, 'data'):
                response = str(result.data)
            else:
                response = str(result)
            
            self.response_ready.emit(response)
            loop.close()
            
        except Exception as e:
            logger.error(f"Error in query worker thread: {e}", exc_info=True)
            self.error_occurred.emit(str(e))