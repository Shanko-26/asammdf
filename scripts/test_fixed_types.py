#!/usr/bin/env python3
"""Test the fixed type hints and dataclass dependencies"""

import asyncio
import os
import json
import sys
import importlib.util
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_module_from_file(name, file_path):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def setup_api_key():
    """Set up API key from config"""
    config_path = Path.home() / ".asammdf" / "plugins" / "aiaspro" / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        api_key = config.get("llm", {}).get("api_key", "")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
    
    return False

async def test_fixed_implementation():
    """Test our fixed implementation with proper type hints"""
    print("Testing Fixed Type Hints and Dataclass Dependencies")
    print("=" * 60)
    
    # Setup API key
    if not setup_api_key():
        print("❌ Could not set up API key")
        return False
    
    try:
        # Load dependencies
        print("1. Creating dataclass dependencies...")
        deps_path = Path(__file__).parent.parent / "src/asammdf/plugins/aiaspro/core/dependencies.py"
        deps_module = load_module_from_file("dependencies", deps_path)
        
        # Create dependencies with dataclass syntax
        deps = deps_module.AIASPRODependencies(
            current_file_name="test_fixed.mf4",
            available_channels=["Engine.RPM", "Engine.Throttle", "Vehicle.Speed", "Brake.Pressure"],
            llm_config={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY")
            }
        )
        
        print(f"   ✓ Dependencies created: {type(deps)}")
        print(f"   → File: {deps.current_file_name}")
        print(f"   → Channels: {deps.available_channels}")
        print(f"   → Is dataclass: {hasattr(deps, '__dataclass_fields__')}")
        
        # Load general agent
        print("\n2. Creating agent with type hints...")
        agent_path = Path(__file__).parent.parent / "src/asammdf/plugins/aiaspro/agents/general_agent.py"
        agent_module = load_module_from_file("general_agent", agent_path)
        
        # Create agent
        agent = agent_module.GeneralAnalysisAgent(deps)
        agent.configure_llm(deps.llm_config)
        
        print(f"   ✓ Agent created: {agent.agent_name}")
        print(f"   → PydanticAI agent: {agent.pydantic_agent}")
        
        # Check if tools are registered by examining the agent
        if hasattr(agent.pydantic_agent, '_tools'):
            print(f"   → Tools registered: {len(agent.pydantic_agent._tools)}")
        else:
            print("   → Tools registration status: unknown")
        
        # Test queries
        print("\n3. Testing queries with fixed dependencies...")
        
        test_queries = [
            "What file is currently loaded?",
            "List all channels", 
            "Search for engine channels"
        ]
        
        success_count = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            try:
                result = await agent.run(query, deps=deps)
                response = str(result.output if hasattr(result, 'output') else result)
                print(f"   → Response: {response[:100]}...")
                
                # Check if our dependencies were used
                if "test_fixed.mf4" in response or "Engine.RPM" in response:
                    print("   ✅ SUCCESS: Dependencies reached the tools!")
                    success_count += 1
                else:
                    print("   ❌ FAIL: Dependencies did not reach the tools")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n   Results: {success_count}/{len(test_queries)} queries used dependencies")
        
        return success_count == len(test_queries)
        
    except Exception as e:
        print(f"\n❌ Fixed implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_fixed_implementation())