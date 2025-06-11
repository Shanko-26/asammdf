"""General analysis agent for automotive data analysis"""

import logging
from typing import List, Dict, Any, Optional

try:
    from pydantic_ai import Agent, RunContext
    HAS_PYDANTIC_AI = True
except ImportError:
    Agent = object
    RunContext = object
    HAS_PYDANTIC_AI = False

try:
    from .base_agent import AIASPROAgent
    from ..core.dependencies import AIASPRODependencies
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    import importlib.util
    
    base_path = Path(__file__).parent / "base_agent.py"
    spec = importlib.util.spec_from_file_location("base_agent", base_path)
    base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module)
    AIASPROAgent = base_module.AIASPROAgent
    
    deps_path = Path(__file__).parent.parent / "core" / "dependencies.py"
    spec = importlib.util.spec_from_file_location("dependencies", deps_path)
    deps_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deps_module)
    AIASPRODependencies = deps_module.AIASPRODependencies

logger = logging.getLogger("asammdf.plugins.aiaspro.agents")


class GeneralAnalysisAgent(AIASPROAgent):
    """General purpose analysis agent for automotive data
    
    This agent handles common queries about MDF files, signal analysis,
    and general automotive data exploration.
    """
    
    def __init__(self, deps):
        """Initialize General Analysis Agent
        
        Args:
            deps: Dependencies container
        """
        super().__init__(
            name="General Analysis Agent",
            model=deps.get_llm_config_for_agent() if deps else "openai:gpt-4o-mini"
        )
        
        self.deps = deps
        
        # Configure confidence scoring for this agent
        self.confidence_keywords = [
            "analyze", "show", "find", "list", "what", "which", "how",
            "signal", "channel", "data", "value", "statistics", "info",
            "tell", "explain", "describe", "summary", "overview"
        ]
        
        self.confidence_patterns = [
            r"show.*signals?",
            r"list.*channels?",
            r"analyze.*data",
            r"what.*channels?",
            r"how many.*signals?",
            r"find.*signal",
            r"statistics.*for",
            r"info.*about"
        ]
        
        # Add capabilities
        self.add_capability("List and search channels")
        self.add_capability("Show signal information and statistics")
        self.add_capability("Analyze signal patterns and trends")
        self.add_capability("Provide file information and metadata")
        self.add_capability("Answer general questions about automotive data")
        
        # Tools will be registered after LLM configuration in configure_llm()
        
        logger.info("General Analysis Agent initialized")
    
    def _register_tools(self):
        """Register tools with the agent"""
        logger.info(f"_register_tools called for {self.agent_name}")
        if not self.pydantic_agent:
            logger.warning("PydanticAI agent not available, skipping tool registration")
            return
        
        logger.info("Registering tools with PydanticAI agent...")
        
        @self.pydantic_agent.tool
        async def list_channels(ctx: RunContext[AIASPRODependencies], pattern: str = None) -> str:
            """List available channels, optionally filtered by pattern
            
            Args:
                pattern: Optional pattern to filter channel names
                
            Returns:
                String listing the channels
            """
            # Debug logging to understand what's being passed
            logger.debug(f"list_channels: ctx.deps = {ctx.deps}")
            logger.debug(f"list_channels: ctx.deps type = {type(ctx.deps)}")
            if ctx.deps:
                logger.debug(f"list_channels: available_channels = {getattr(ctx.deps, 'available_channels', 'MISSING')}")
                logger.debug(f"list_channels: current_file_name = {getattr(ctx.deps, 'current_file_name', 'MISSING')}")
            
            if not ctx.deps or not ctx.deps.available_channels:
                debug_msg = f"No file loaded. ctx.deps={ctx.deps}, available_channels={getattr(ctx.deps, 'available_channels', None) if ctx.deps else 'None'}"
                logger.warning(debug_msg)
                return "No file is currently loaded or no channels available."
            
            channels = ctx.deps.available_channels
            
            if pattern:
                pattern_lower = pattern.lower()
                channels = [ch for ch in channels if pattern_lower in ch.lower()]
            
            if not channels:
                if pattern:
                    return f"No channels found matching pattern '{pattern}'."
                else:
                    return "No channels available in the current file."
            
            # Limit display for readability
            display_channels = channels[:20]
            result = f"Found {len(channels)} channels"
            if pattern:
                result += f" matching '{pattern}'"
            if len(channels) > 20:
                result += f" (showing first 20)"
            result += ":\n\n" + "\n".join(f"• {ch}" for ch in display_channels)
            
            if len(channels) > 20:
                result += f"\n\n... and {len(channels) - 20} more channels"
            
            return result
        
        @self.pydantic_agent.tool
        async def get_signal_info(ctx: RunContext[AIASPRODependencies], signal_name: str) -> str:
            """Get detailed information about a specific signal
            
            Args:
                signal_name: Name of the signal to analyze
                
            Returns:
                String with signal information
            """
            if not ctx.deps or not ctx.deps.mdf:
                return "No file is currently loaded."
            
            try:
                # Try to get signal from MDF
                signal = ctx.deps.mdf.get(signal_name)
                
                info = f"Signal Information: {signal_name}\n"
                info += "=" * (len(signal_name) + 20) + "\n\n"
                
                # Basic properties
                info += f"📊 **Data Points**: {len(signal.samples):,}\n"
                info += f"📏 **Unit**: {signal.unit or 'No unit'}\n"
                
                # Statistical analysis
                import numpy as np
                if len(signal.samples) > 0:
                    info += f"📈 **Minimum**: {float(np.min(signal.samples)):.6f}\n"
                    info += f"📈 **Maximum**: {float(np.max(signal.samples)):.6f}\n"
                    info += f"📈 **Mean**: {float(np.mean(signal.samples)):.6f}\n"
                    info += f"📈 **Std Dev**: {float(np.std(signal.samples)):.6f}\n"
                
                # Timing information
                if len(signal.timestamps) > 1:
                    duration = signal.timestamps[-1] - signal.timestamps[0]
                    sample_rate = len(signal.samples) / duration if duration > 0 else 0
                    info += f"⏱️ **Duration**: {duration:.3f} seconds\n"
                    info += f"⏱️ **Avg Sample Rate**: {sample_rate:.2f} Hz\n"
                
                # Additional metadata
                if hasattr(signal, 'comment') and signal.comment:
                    info += f"💬 **Comment**: {signal.comment}\n"
                
                return info
                
            except Exception as e:
                return f"Error accessing signal '{signal_name}': {str(e)}\n\n" \
                       f"Available channels: {', '.join(ctx.deps.available_channels[:5])}..."
        
        @self.pydantic_agent.tool  
        async def search_channels(ctx: RunContext[AIASPRODependencies], search_term: str) -> str:
            """Search for channels containing a specific term
            
            Args:
                search_term: Term to search for in channel names
                
            Returns:
                String with search results
            """
            if not ctx.deps or not ctx.deps.available_channels:
                return "No file is currently loaded or no channels available."
            
            search_lower = search_term.lower()
            matching_channels = [
                ch for ch in ctx.deps.available_channels 
                if search_lower in ch.lower()
            ]
            
            if not matching_channels:
                return f"No channels found containing '{search_term}'.\n\n" \
                       f"Try searching for common automotive terms like:\n" \
                       f"• engine, rpm, speed, throttle\n" \
                       f"• brake, pressure, abs\n" \
                       f"• temp, temperature, coolant\n" \
                       f"• gear, transmission\n" \
                       f"• can, lin (for bus signals)"
            
            result = f"Found {len(matching_channels)} channels containing '{search_term}':\n\n"
            
            # Group by common prefixes for better organization
            grouped = {}
            for ch in matching_channels[:15]:  # Limit to 15 results
                prefix = ch.split('.')[0] if '.' in ch else 'Other'
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(ch)
            
            for prefix, channels in grouped.items():
                if prefix != 'Other':
                    result += f"**{prefix}** group:\n"
                for ch in channels:
                    result += f"  • {ch}\n"
                result += "\n"
            
            if len(matching_channels) > 15:
                result += f"... and {len(matching_channels) - 15} more matches"
            
            return result
        
        @self.pydantic_agent.tool
        async def get_file_info(ctx: RunContext) -> str:
            """Get information about the currently loaded file
            
            Returns:
                String with file information
            """
            # Debug logging
            logger.debug(f"get_file_info: ctx.deps = {ctx.deps}")
            logger.debug(f"get_file_info: ctx.deps type = {type(ctx.deps)}")
            if ctx.deps:
                logger.debug(f"get_file_info: current_file_name = {getattr(ctx.deps, 'current_file_name', 'MISSING')}")
                logger.debug(f"get_file_info: available_channels count = {len(getattr(ctx.deps, 'available_channels', []))}")
            
            if not ctx.deps:
                logger.warning("get_file_info: No dependencies available")
                return "No dependencies available."
            
            if not ctx.deps.current_file_name:
                logger.warning(f"get_file_info: No current_file_name. Attrs: {dir(ctx.deps)}")
                return "No file is currently loaded. Please load an MDF file to analyze."
            
            info = f"File Information\n"
            info += "=" * 20 + "\n\n"
            info += f"📁 **File**: {ctx.deps.current_file_name}\n"
            info += f"📊 **Channels**: {len(ctx.deps.available_channels):,}\n"
            
            if ctx.deps.mdf:
                try:
                    # Get version info
                    version = getattr(ctx.deps.mdf, 'version', 'Unknown')
                    info += f"📋 **MDF Version**: {version}\n"
                    
                    # Get measurement info if available
                    if hasattr(ctx.deps.mdf, 'header'):
                        header = ctx.deps.mdf.header
                        if hasattr(header, 'comment') and header.comment:
                            info += f"💬 **Description**: {header.comment[:100]}...\n"
                    
                except Exception as e:
                    logger.warning(f"Error getting file details: {e}")
            
            # Show some example channels by category
            if ctx.deps.available_channels:
                info += "\n**Sample Channels by Category**:\n"
                
                categories = {
                    "Engine": ["engine", "rpm", "throttle", "torque"],
                    "Vehicle": ["speed", "gear", "brake", "wheel"],
                    "Temperature": ["temp", "cool", "oil"],
                    "Pressure": ["pressure", "bar", "kpa"],
                    "Electrical": ["volt", "amp", "battery"]
                }
                
                for category, terms in categories.items():
                    category_channels = []
                    for ch in ctx.deps.available_channels:
                        if any(term in ch.lower() for term in terms):
                            category_channels.append(ch)
                            if len(category_channels) >= 3:  # Limit examples
                                break
                    
                    if category_channels:
                        info += f"  **{category}**: {', '.join(category_channels)}\n"
            
            return info
        
        @self.pydantic_agent.tool
        async def suggest_analysis(ctx: RunContext[AIASPRODependencies], user_interest: str = "") -> str:
            """Suggest analysis approaches based on available data
            
            Args:
                user_interest: Optional description of what user is interested in
                
            Returns:
                String with analysis suggestions
            """
            if not ctx.deps or not ctx.deps.available_channels:
                return "No file loaded. Please load an MDF file first."
            
            suggestions = "🔍 **Analysis Suggestions**\n\n"
            
            channels = ctx.deps.available_channels
            
            # Automotive-specific suggestions based on available signals
            automotive_categories = {
                "Engine Performance": {
                    "terms": ["engine", "rpm", "throttle", "torque", "power"],
                    "analysis": [
                        "Plot engine RPM vs throttle position",
                        "Analyze engine load patterns",
                        "Check for engine knock events",
                        "Compare power output across different conditions"
                    ]
                },
                "Vehicle Dynamics": {
                    "terms": ["speed", "accel", "brake", "wheel", "gear"],
                    "analysis": [
                        "Plot vehicle speed profile", 
                        "Analyze acceleration/deceleration patterns",
                        "Check brake pressure vs vehicle dynamics",
                        "Examine gear shift patterns"
                    ]
                },
                "Thermal Management": {
                    "terms": ["temp", "cool", "oil", "thermal"],
                    "analysis": [
                        "Monitor temperature trends",
                        "Check cooling system efficiency",
                        "Analyze thermal cycles",
                        "Identify overheating events"
                    ]
                },
                "Fuel Economy": {
                    "terms": ["fuel", "consumption", "efficiency", "mpg"],
                    "analysis": [
                        "Calculate fuel consumption rates",
                        "Correlate fuel usage with driving patterns",
                        "Analyze efficiency across different speeds",
                        "Identify fuel-saving opportunities"
                    ]
                }
            }
            
            found_categories = []
            for category, info in automotive_categories.items():
                matching_channels = [
                    ch for ch in channels 
                    if any(term in ch.lower() for term in info["terms"])
                ]
                
                if matching_channels:
                    found_categories.append(category)
                    suggestions += f"**{category}** ({len(matching_channels)} signals):\n"
                    for suggestion in info["analysis"][:2]:  # Limit to 2 suggestions per category
                        suggestions += f"  • {suggestion}\n"
                    suggestions += f"  • Example signals: {', '.join(matching_channels[:3])}\n\n"
            
            if not found_categories:
                suggestions += "Based on your data, I recommend:\n"
                suggestions += "• Start with 'list channels' to see all available signals\n"
                suggestions += "• Use 'search channels engine' to find engine-related data\n"
                suggestions += "• Try 'get signal info [signal_name]' for detailed analysis\n"
            
            # Add user-specific suggestions if provided
            if user_interest:
                suggestions += f"\n**Based on your interest in '{user_interest}'**:\n"
                interest_channels = [
                    ch for ch in channels 
                    if user_interest.lower() in ch.lower()
                ]
                if interest_channels:
                    suggestions += f"• Found {len(interest_channels)} related signals\n"
                    suggestions += f"• Start with: {', '.join(interest_channels[:3])}\n"
                else:
                    suggestions += f"• Try searching for '{user_interest}' in channel names\n"
            
            return suggestions
        
        logger.info("Tools registered with General Analysis Agent")
    
    async def run(self, query: str, deps=None, context: Dict[str, Any] = None) -> Any:
        """Run the agent with enhanced context awareness"""
        # Update dependencies if provided
        if deps:
            self.deps = deps
        
        # Configure LLM if not already done
        if self.deps and not self.pydantic_agent and HAS_PYDANTIC_AI:
            self.configure_llm(self.deps.llm_config)
            if self.pydantic_agent:
                self._register_tools()
        
        # Call parent run method
        return await super().run(query, self.deps, context)
    
    def get_specialized_info(self) -> Dict[str, Any]:
        """Get specialized information about this agent"""
        info = self.get_info()
        info.update({
            "specialization": "General automotive data analysis",
            "best_for": [
                "Channel listing and search",
                "Signal information and statistics", 
                "File overview and metadata",
                "General automotive queries",
                "Analysis suggestions"
            ],
            "example_queries": [
                "List all engine-related channels",
                "Show me information about Engine.RPM",
                "Search for brake signals",
                "What channels are available?",
                "Suggest analysis for this data"
            ]
        })
        return info