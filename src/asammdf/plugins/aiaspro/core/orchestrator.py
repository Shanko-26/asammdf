"""AI orchestrator for managing multiple agents and routing queries"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator

try:
    from .dependencies import AIASPRODependencies
    from ..agents.base_agent import AIASPROAgent, AgentRegistry
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    import importlib.util
    
    # Load dependencies module
    deps_path = Path(__file__).parent / "dependencies.py"
    spec = importlib.util.spec_from_file_location("dependencies", deps_path)
    deps_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deps_module)
    AIASPRODependencies = deps_module.AIASPRODependencies
    
    # Load base agent module  
    agent_path = Path(__file__).parent.parent / "agents" / "base_agent.py"
    spec = importlib.util.spec_from_file_location("base_agent", agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    AIASPROAgent = agent_module.AIASPROAgent
    AgentRegistry = agent_module.AgentRegistry

logger = logging.getLogger("asammdf.plugins.aiaspro.core")


class AIOrchestrator:
    """Orchestrates multiple agents and routes queries to the best agent
    
    The orchestrator is responsible for:
    - Managing multiple specialized agents
    - Routing queries to the most appropriate agent
    - Handling streaming responses
    - Managing conversation context
    - Fallback handling when agents fail
    """
    
    def __init__(self, dependencies: AIASPRODependencies):
        """Initialize orchestrator
        
        Args:
            dependencies: Dependencies container
        """
        self.deps = dependencies
        self.registry = AgentRegistry()
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
        
        # Set session ID in dependencies
        self.deps.set_session_id(self.session_id)
        
        # Configuration
        self.max_history_length = 50
        self.enable_streaming = True
        self.fallback_enabled = True
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"AI Orchestrator initialized with session {self.session_id}")
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        try:
            # Import and initialize agents
            from ..agents.general_agent import GeneralAnalysisAgent
            
            # Create general analysis agent
            general_agent = GeneralAnalysisAgent(self.deps)
            
            # Configure with API key
            self._configure_agent_llm(general_agent)
            
            self.registry.register(general_agent, is_default=True)
            
            logger.info(f"Initialized {len(self.registry.list_agents())} agents")
            
        except ImportError as e:
            logger.warning(f"Some agents could not be imported: {e}")
            # Fallback import for standalone testing
            try:
                import importlib.util
                from pathlib import Path
                
                # Load general agent directly
                agent_path = Path(__file__).parent.parent / "agents" / "general_agent.py"
                spec = importlib.util.spec_from_file_location("general_agent", agent_path)
                agent_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agent_module)
                
                # Create general analysis agent
                general_agent = agent_module.GeneralAnalysisAgent(self.deps)
                
                # Configure with API key
                self._configure_agent_llm(general_agent)
                
                self.registry.register(general_agent, is_default=True)
                
                logger.info(f"Initialized {len(self.registry.list_agents())} agents (standalone mode)")
                
            except Exception as fallback_e:
                logger.warning(f"Fallback agent loading also failed: {fallback_e}")
                # Create a minimal fallback agent
                self._create_fallback_agent()
    
    def _configure_agent_llm(self, agent):
        """Configure an agent with LLM settings
        
        Args:
            agent: Agent to configure
        """
        if hasattr(agent, 'configure_llm'):
            try:
                agent.configure_llm(self.deps.llm_config)
                logger.debug(f"Configured LLM for agent: {agent.agent_name}")
            except Exception as e:
                logger.warning(f"Failed to configure LLM for {agent.agent_name}: {e}")
    
    def _create_fallback_agent(self):
        """Create a minimal fallback agent when imports fail"""
        try:
            from ..agents.base_agent import AIASPROAgent
        except ImportError:
            # Fallback import for standalone testing
            import importlib.util
            from pathlib import Path
            
            base_path = Path(__file__).parent.parent / "agents" / "base_agent.py"
            spec = importlib.util.spec_from_file_location("base_agent", base_path)
            base_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base_module)
            AIASPROAgent = base_module.AIASPROAgent
        
        # Create a basic agent with minimal functionality
        fallback_agent = AIASPROAgent(
            name="Fallback Agent",
            model=self.deps.get_llm_config_for_agent()
        )
        
        # Add basic capabilities
        fallback_agent.add_capability("Basic query handling")
        fallback_agent.add_capability("File information")
        fallback_agent.confidence_keywords = ["help", "hello", "info", "status"]
        
        # Configure LLM
        self._configure_agent_llm(fallback_agent)
        
        self.registry.register(fallback_agent, is_default=True)
        logger.info("Created fallback agent")
    
    async def route_and_execute(self, query: str, context: Dict[str, Any] = None) -> Any:
        """Route query to best agent and execute
        
        Args:
            query: User query string
            context: Optional context information
            
        Returns:
            Agent response
        """
        # Build context if not provided
        if context is None:
            context = self._build_context()
        
        # Add query to conversation history
        self._add_to_history("user", query)
        
        # Route to best agent
        agent = await self._route_query(query, context)
        
        if not agent:
            response = "I'm sorry, but I couldn't find an appropriate agent to handle your query."
            self._add_to_history("system", response)
            return self._create_response(response)
        
        try:
            # Execute query with selected agent
            logger.info(f"Routing query to agent: {agent.agent_name}")
            result = await agent.run(query, deps=self.deps, context=context)
            
            # Extract response text
            response_text = self._extract_response_text(result)
            self._add_to_history("assistant", response_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing query with agent {agent.agent_name}: {e}")
            
            if self.fallback_enabled:
                fallback_response = await self._fallback_execution(query, context)
                self._add_to_history("assistant", fallback_response)
                return self._create_response(fallback_response)
            else:
                error_response = f"I encountered an error processing your query: {str(e)}"
                self._add_to_history("system", error_response)
                return self._create_response(error_response)
    
    async def route_and_execute_stream(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Route query and execute with streaming response
        
        Args:
            query: User query string
            context: Optional context information
            
        Yields:
            Response chunks as they become available
        """
        # Build context if not provided
        if context is None:
            context = self._build_context()
        
        # Add query to conversation history
        self._add_to_history("user", query)
        
        # Route to best agent
        agent = await self._route_query(query, context)
        
        if not agent:
            response = "I'm sorry, but I couldn't find an appropriate agent to handle your query."
            self._add_to_history("system", response)
            yield response
            return
        
        try:
            logger.info(f"Streaming query to agent: {agent.agent_name}")
            
            response_parts = []
            async for chunk in agent.run_stream(query, deps=self.deps, context=context):
                response_parts.append(str(chunk))
                yield str(chunk)
            
            # Add complete response to history
            complete_response = "".join(response_parts)
            self._add_to_history("assistant", complete_response)
            
        except Exception as e:
            logger.error(f"Error in streaming execution with agent {agent.agent_name}: {e}")
            
            if self.fallback_enabled:
                fallback_response = await self._fallback_execution(query, context)
                self._add_to_history("assistant", fallback_response)
                yield fallback_response
            else:
                error_response = f"Error processing query: {str(e)}"
                self._add_to_history("system", error_response)
                yield error_response
    
    async def _route_query(self, query: str, context: Dict[str, Any]) -> Optional[AIASPROAgent]:
        """Route query to the best available agent
        
        Args:
            query: User query
            context: Context information
            
        Returns:
            Best agent for the query or None
        """
        # Use registry to find best agent
        agent = self.registry.find_best_agent(query, context)
        
        if agent:
            logger.debug(f"Routed query to agent: {agent.agent_name}")
        else:
            logger.warning("No suitable agent found for query")
        
        return agent
    
    def _build_context(self) -> Dict[str, Any]:
        """Build context from current state
        
        Returns:
            Context dictionary
        """
        base_context = self.deps.get_context()
        
        # Add conversation history
        base_context.update({
            "conversation_history": self.conversation_history[-10:],  # Last 10 messages
            "session_id": self.session_id,
            "orchestrator_info": {
                "available_agents": self.registry.list_agents(),
                "capabilities": self.registry.get_capabilities()
            }
        })
        
        return base_context
    
    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _extract_response_text(self, result: Any) -> str:
        """Extract text from agent response
        
        Args:
            result: Agent response result
            
        Returns:
            Response text string
        """
        if hasattr(result, 'data'):
            return str(result.data)
        elif hasattr(result, 'content'):
            return str(result.content)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    def _create_response(self, text: str) -> Any:
        """Create a standardized response object
        
        Args:
            text: Response text
            
        Returns:
            Response object
        """
        # Simple response object for fallback
        class SimpleResponse:
            def __init__(self, data):
                self.data = data
        
        return SimpleResponse(text)
    
    async def _fallback_execution(self, query: str, context: Dict[str, Any]) -> str:
        """Execute fallback response when agents fail
        
        Args:
            query: Original query
            context: Context information
            
        Returns:
            Fallback response string
        """
        logger.info("Executing fallback response")
        
        # Simple pattern-based fallback responses
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['hello', 'hi', 'hey']):
            return "Hello! I'm your AI assistant for automotive data analysis. How can I help you today?"
        
        elif any(term in query_lower for term in ['help', 'what can you do']):
            capabilities = []
            for agent_name in self.registry.list_agents():
                agent = self.registry.get_agent(agent_name)
                if agent:
                    capabilities.extend(agent.capabilities)
            
            if capabilities:
                return "I can help you with:\n" + "\n".join(f"• {cap}" for cap in capabilities[:5])
            else:
                return "I can help you analyze automotive measurement data from MDF files."
        
        elif any(term in query_lower for term in ['status', 'info']):
            file_info = "No file loaded"
            if context.get('has_file'):
                file_name = context.get('file_name', 'Unknown')
                channel_count = context.get('channel_count', 0)
                file_info = f"File: {file_name} ({channel_count} channels)"
            
            return f"System Status:\n• {file_info}\n• Available agents: {len(self.registry.list_agents())}\n• Session: {self.session_id[:8]}..."
        
        else:
            return f"I understand you're asking about '{query}'. I'm currently operating in fallback mode. " \
                   f"Please check that all AI services are properly configured and try again."
    
    # Public interface methods
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status
        
        Returns:
            Status dictionary
        """
        return {
            "session_id": self.session_id,
            "agents": {
                name: agent.get_info() 
                for name, agent in self.registry.agents.items()
            },
            "default_agent": self.registry.default_agent,
            "conversation_length": len(self.conversation_history),
            "dependencies_ready": self.deps.is_ready(),
            "configuration": {
                "max_history_length": self.max_history_length,
                "enable_streaming": self.enable_streaming,
                "fallback_enabled": self.fallback_enabled
            }
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def set_dependencies(self, dependencies: AIASPRODependencies):
        """Update dependencies
        
        Args:
            dependencies: New dependencies container
        """
        self.deps = dependencies
        
        # Update all agents with new dependencies
        for agent in self.registry.agents.values():
            if hasattr(agent, 'update_dependencies'):
                agent.update_dependencies(dependencies)
        
        logger.info("Dependencies updated for all agents")
    
    def add_agent(self, agent: AIASPROAgent, is_default: bool = False):
        """Add a new agent to the orchestrator
        
        Args:
            agent: Agent to add
            is_default: Whether this should be the default agent
        """
        self.registry.register(agent, is_default)
        logger.info(f"Added agent to orchestrator: {agent.agent_name}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def simulate_query(self, query: str) -> Dict[str, Any]:
        """Simulate query routing without execution (for testing)
        
        Args:
            query: Query to simulate
            
        Returns:
            Simulation results
        """
        context = self._build_context()
        
        # Calculate confidence for each agent
        agent_scores = {}
        for name, agent in self.registry.agents.items():
            confidence = agent.calculate_confidence(query, context)
            agent_scores[name] = confidence
        
        # Find best agent
        best_agent = self.registry.find_best_agent(query, context)
        
        return {
            "query": query,
            "agent_scores": agent_scores,
            "selected_agent": best_agent.agent_name if best_agent else None,
            "context": context
        }