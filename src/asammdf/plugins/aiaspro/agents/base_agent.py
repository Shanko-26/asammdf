"""Base agent implementation for AIASPRO"""

import logging
import re
from typing import List, Dict, Any, Optional

try:
    from pydantic_ai import Agent, RunContext
    HAS_PYDANTIC_AI = True
except ImportError:
    # Fallback for when pydantic-ai is not available
    Agent = object
    RunContext = object
    HAS_PYDANTIC_AI = False

logger = logging.getLogger("asammdf.plugins.aiaspro.agents")


class AIASPROAgent:
    """Base class for all AIASPRO agents
    
    This class provides the foundation for specialized AI agents that can work
    with automotive measurement data through asammdf.
    """
    
    def __init__(self, 
                 name: str,
                 model: str = 'openai:gpt-4o-mini',
                 system_prompt: Optional[str] = None):
        """Initialize base agent
        
        Args:
            name: Agent name/identifier
            model: LLM model specification (e.g., "openai:gpt-4o-mini")
            system_prompt: Custom system prompt (optional)
        """
        self.agent_name = name
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Confidence scoring configuration
        self.confidence_keywords: List[str] = []
        self.confidence_patterns: List[str] = []
        self.confidence_threshold = 0.5
        
        # Agent capabilities
        self.capabilities = []
        self.supported_file_types = ["mdf", "mf4", "dat"]
        
        # Tool registry
        self.tools = {}
        
        # Initialize the underlying agent if pydantic-ai is available
        if HAS_PYDANTIC_AI:
            self._init_pydantic_agent()
        else:
            self.pydantic_agent = None
        
        logger.info(f"Initialized agent: {self.agent_name}")
    
    def configure_llm(self, llm_config: dict):
        """Configure the LLM with API key and other settings
        
        Args:
            llm_config: LLM configuration dict with api_key, etc.
        """
        if not HAS_PYDANTIC_AI:
            logger.warning("PydanticAI not available, using fallback mode")
            return
        
        try:
            # Set up environment variable for OpenAI
            import os
            api_key = llm_config.get("api_key")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Now initialize the PydanticAI agent
                self.pydantic_agent = Agent(
                    model=self.model,
                    system_prompt=self.system_prompt
                )
                
                # Register tools after agent creation (hook for subclasses)
                if hasattr(self, '_register_tools'):
                    self._register_tools()
                
                logger.info(f"PydanticAI agent configured for {self.agent_name}")
            else:
                logger.warning("No API key provided for PydanticAI agent")
                
        except Exception as e:
            logger.error(f"Failed to configure PydanticAI agent: {e}")
            self.pydantic_agent = None
    
    def _init_pydantic_agent(self):
        """Initialize the underlying PydanticAI agent"""
        try:
            # For now, we'll initialize without API key and configure it later
            # when we have access to the dependencies
            self.pydantic_agent = None  # Will be initialized in configure_llm method
        except Exception as e:
            logger.error(f"Failed to initialize PydanticAI agent: {e}")
            self.pydantic_agent = None
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt for this agent
        
        Returns:
            Default system prompt string
        """
        return f"""You are {self.agent_name}, an AI assistant specialized in 
automotive data analysis using asammdf. You have direct access to MDF 
files and can perform various analyses on automotive signals.

Your capabilities include:
- Analyzing measurement data from automotive ECUs
- Working with CAN bus, LIN bus, and other automotive communication data
- Processing signals like engine RPM, vehicle speed, brake pressure, etc.
- Generating plots and visualizations
- Performing statistical analysis and pattern detection
- Providing insights about vehicle performance and behavior

You should be helpful, accurate, and provide actionable insights about
automotive measurement data. Always consider the automotive context when
analyzing signals and data patterns."""
    
    def calculate_confidence(self, query: str, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for handling this query
        
        This method determines how well-suited this agent is for handling
        a particular query based on keywords, patterns, and context.
        
        Args:
            query: The user query string
            context: Optional context information
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        query_lower = query.lower()
        
        # Keyword matching (high confidence indicators)
        keyword_score = 0.0
        for keyword in self.confidence_keywords:
            if keyword.lower() in query_lower:
                keyword_score += 0.3
        
        # Pattern matching (specific query patterns)
        pattern_score = 0.0
        for pattern in self.confidence_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                pattern_score += 0.4
        
        # Context boost (file loaded, relevant data available)
        context_score = 0.0
        if context:
            if context.get('has_file'):
                context_score += 0.2
            
            if context.get('channel_count', 0) > 0:
                context_score += 0.1
            
            # Boost for automotive-related channels
            channels = context.get('available_channels', [])
            automotive_terms = ['engine', 'brake', 'speed', 'rpm', 'throttle', 
                              'gear', 'wheel', 'temp', 'pressure', 'can', 'lin']
            
            automotive_channels = sum(1 for ch in channels 
                                    if any(term in ch.lower() for term in automotive_terms))
            
            if automotive_channels > 0:
                context_score += min(0.2, automotive_channels * 0.02)
        
        # Combine scores
        score = min(keyword_score + pattern_score + context_score, 1.0)
        
        logger.debug(f"Confidence for '{query}': {score:.2f} "
                    f"(keywords: {keyword_score:.2f}, "
                    f"patterns: {pattern_score:.2f}, "
                    f"context: {context_score:.2f})")
        
        return score
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Check if this agent can handle the given query
        
        Args:
            query: The user query
            context: Optional context information
            
        Returns:
            True if agent can handle the query
        """
        confidence = self.calculate_confidence(query, context)
        return confidence >= self.confidence_threshold
    
    def add_capability(self, capability: str):
        """Add a capability to this agent
        
        Args:
            capability: Capability description
        """
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            logger.debug(f"Added capability to {self.agent_name}: {capability}")
    
    def register_tool(self, tool_name: str, tool_function):
        """Register a tool with this agent
        
        Args:
            tool_name: Name of the tool
            tool_function: Tool function to register
        """
        self.tools[tool_name] = tool_function
        
        # Register with PydanticAI agent if available
        if self.pydantic_agent and HAS_PYDANTIC_AI:
            try:
                # Decorate the function as a tool
                decorated_tool = self.pydantic_agent.tool(tool_function)
                logger.debug(f"Registered tool '{tool_name}' with PydanticAI agent")
            except Exception as e:
                logger.warning(f"Could not register tool '{tool_name}' with PydanticAI: {e}")
        
        logger.info(f"Registered tool: {tool_name}")
    
    async def run(self, query: str, deps=None, context: Dict[str, Any] = None) -> Any:
        """Run the agent with a query
        
        Args:
            query: User query
            deps: Dependencies object
            context: Additional context
            
        Returns:
            Agent response
        """
        if not HAS_PYDANTIC_AI or not self.pydantic_agent:
            # Fallback implementation
            return await self._fallback_run(query, deps, context)
        
        try:
            # Use PydanticAI to run the query with proper dependency injection
            # PydanticAI passes dependencies as the deps parameter to RunContext
            result = await self.pydantic_agent.run(query, deps=deps)
            
            logger.info(f"Agent {self.agent_name} completed query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error running agent {self.agent_name}: {e}")
            logger.debug(f"Dependencies passed: {deps}")
            # Return fallback response
            return await self._fallback_run(query, deps, context)
    
    async def run_stream(self, query: str, deps=None, context: Dict[str, Any] = None):
        """Run the agent with streaming response
        
        Args:
            query: User query
            deps: Dependencies object
            context: Additional context
            
        Returns:
            Async generator for streaming response
        """
        if not HAS_PYDANTIC_AI or not self.pydantic_agent:
            # Fallback: return single response
            result = await self._fallback_run(query, deps, context)
            yield str(result)
            return
        
        try:
            # Use PydanticAI streaming
            async with self.pydantic_agent.run_stream(query, deps=deps) as result:
                async for message in result.stream():
                    yield message
                    
        except Exception as e:
            logger.error(f"Error in streaming run for {self.agent_name}: {e}")
            yield f"Error: {str(e)}"
    
    async def _fallback_run(self, query: str, deps=None, context: Dict[str, Any] = None) -> str:
        """Fallback implementation when PydanticAI is not available
        
        Args:
            query: User query
            deps: Dependencies object
            context: Additional context
            
        Returns:
            Fallback response string
        """
        logger.warning(f"Using fallback implementation for agent {self.agent_name}")
        
        # Simple pattern-based responses for testing
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['hello', 'hi', 'hey']):
            return f"Hello! I'm {self.agent_name}, ready to help with automotive data analysis."
        
        elif 'channels' in query_lower or 'signals' in query_lower:
            if deps and hasattr(deps, 'available_channels'):
                channel_count = len(deps.available_channels)
                return f"I found {channel_count} channels in the current file. " \
                       f"Would you like me to list them or analyze specific ones?"
            else:
                return "No file is currently loaded. Please load an MDF file to see available channels."
        
        elif 'help' in query_lower:
            return f"I'm {self.agent_name} and I can help you with:\n" + \
                   "\n".join(f"• {cap}" for cap in self.capabilities[:5])
        
        else:
            return f"I understand you're asking about: '{query}'. " \
                   f"This is a fallback response from {self.agent_name}. " \
                   f"For full functionality, please ensure PydanticAI is properly configured."
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": self.agent_name,
            "model": self.model,
            "capabilities": self.capabilities,
            "tools": list(self.tools.keys()),
            "confidence_threshold": self.confidence_threshold,
            "supported_file_types": self.supported_file_types,
            "has_pydantic_ai": HAS_PYDANTIC_AI and self.pydantic_agent is not None
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.agent_name}', model='{self.model}')"


class AgentRegistry:
    """Registry for managing multiple agents"""
    
    def __init__(self):
        """Initialize agent registry"""
        self.agents: Dict[str, AIASPROAgent] = {}
        self.default_agent: Optional[str] = None
    
    def register(self, agent: AIASPROAgent, is_default: bool = False):
        """Register an agent
        
        Args:
            agent: Agent to register
            is_default: Whether this should be the default agent
        """
        self.agents[agent.agent_name] = agent
        
        if is_default or not self.default_agent:
            self.default_agent = agent.agent_name
        
        logger.info(f"Registered agent: {agent.agent_name}")
    
    def get_agent(self, name: str) -> Optional[AIASPROAgent]:
        """Get agent by name
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(name)
    
    def get_default_agent(self) -> Optional[AIASPROAgent]:
        """Get the default agent
        
        Returns:
            Default agent instance or None
        """
        if self.default_agent:
            return self.agents.get(self.default_agent)
        return None
    
    def find_best_agent(self, query: str, context: Dict[str, Any] = None) -> Optional[AIASPROAgent]:
        """Find the best agent for a query
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Best agent or default agent
        """
        best_agent = None
        highest_confidence = 0.0
        
        for agent in self.agents.values():
            confidence = agent.calculate_confidence(query, context)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_agent = agent
        
        # Return best agent if confidence is high enough, otherwise default
        if best_agent and highest_confidence >= best_agent.confidence_threshold:
            return best_agent
        
        return self.get_default_agent()
    
    def list_agents(self) -> List[str]:
        """List all registered agent names
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents
        
        Returns:
            Dict mapping agent names to their capabilities
        """
        return {name: agent.capabilities for name, agent in self.agents.items()}