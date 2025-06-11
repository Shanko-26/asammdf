#!/usr/bin/env python
"""Setup AIASPRO development environment"""

import subprocess
import sys
from pathlib import Path

def setup_dev_env():
    project_root = Path(__file__).parent.parent
    
    print("Setting up AIASPRO development environment...")
    
    # Install AIASPRO requirements
    req_file = project_root / "src/asammdf/plugins/aiaspro/requirements.txt"
    print(f"\nInstalling AIASPRO requirements from {req_file}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    
    # Install development requirements
    print("\nInstalling development requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-qt", "pytest-cov"])
    
    # Create config directory
    config_dir = Path.home() / ".asammdf" / "plugins" / "aiaspro"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample config if not exists
    config_file = config_dir / "config.json"
    if not config_file.exists():
        import json
        sample_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-your-api-key-here",
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "enable_auto_analysis": True,
            "cache_results": True,
            "user_requirements": ""
        }
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"\n✓ Created sample config at {config_file}")
        print("  → Please update with your OpenAI API key")
    
    print("\n✅ Development environment setup complete!")

if __name__ == "__main__":
    setup_dev_env()
