#!/usr/bin/env python
"""
AIASPRO Setup Script
Creates the directory structure and initial files for the AI Assistant Pro plugin
"""

import os
import sys
from pathlib import Path
import json

def create_directory_structure():
    """Create the full directory structure for AIASPRO plugin"""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Define directory structure
    directories = [
        # Plugin system directories
        "src/asammdf/gui/plugins",
        
        # AIASPRO plugin directories
        "src/asammdf/plugins/aiaspro",
        "src/asammdf/plugins/aiaspro/core",
        "src/asammdf/plugins/aiaspro/agents",
        "src/asammdf/plugins/aiaspro/tools",
        "src/asammdf/plugins/aiaspro/services",
        "src/asammdf/plugins/aiaspro/ui",
        "src/asammdf/plugins/aiaspro/ui/components",
        "src/asammdf/plugins/aiaspro/tests",
    ]
    
    # Create directories
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
        
        # Create __init__.py files
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            print(f"  → Added __init__.py")
    
    return project_root

def create_plugin_manifest(project_root):
    """Create plugin manifest file"""
    manifest = {
        "name": "AI Assistant Pro",
        "version": "0.1.0",
        "author": "AIASPRO Team",
        "description": "AI-powered automotive data analysis for asammdf",
        "entry_point": "asammdf.plugins.aiaspro.plugin:AIASPROPlugin",
        "dependencies": [
            "pydantic-ai>=0.0.7",
            "openai>=1.0.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0"
        ],
        "min_asammdf_version": "7.0.0",
        "settings": {
            "llm_provider": "openai",
            "model": "gpt-4o-mini",
            "enable_auto_analysis": True
        }
    }
    
    manifest_path = project_root / "src/asammdf/plugins/aiaspro/manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created plugin manifest: {manifest_path}")

def create_requirements_file(project_root):
    """Create requirements file for AIASPRO"""
    requirements = """# AIASPRO Plugin Requirements
pydantic-ai>=0.0.7
openai>=1.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
cryptography>=3.4.0
asyncio>=3.4.3
"""
    
    req_path = project_root / "src/asammdf/plugins/aiaspro/requirements.txt"
    req_path.write_text(requirements)
    print(f"✓ Created requirements file: {req_path}")

def create_readme(project_root):
    """Create README for AIASPRO plugin"""
    readme = """# AI Assistant Pro (AIASPRO) Plugin for asammdf

## Overview
AIASPRO is a native AI integration plugin for asammdf that provides intelligent automotive data analysis capabilities through natural language queries.

## Features
- Natural language data exploration
- Automated pattern detection and insights
- Intelligent plotting and visualization
- Requirements-based analysis
- Direct MDF object access for high performance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Enable the plugin in asammdf:
The plugin will be automatically discovered and loaded by asammdf's plugin system.

## Configuration

Create a config file at `~/.asammdf/plugins/aiaspro/config.json`:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "your-api-key-here"
  },
  "enable_auto_analysis": true,
  "cache_results": true
}
```

## Usage

1. Open asammdf GUI
2. Load an MDF file
3. Go to `AI Assistant` menu → `Open AI Assistant`
4. Start asking questions about your data!

## Development

Run tests:
```bash
pytest src/asammdf/plugins/aiaspro/tests/ -v
```

## License
Same as asammdf (LGPL v3)
"""
    
    readme_path = project_root / "src/asammdf/plugins/aiaspro/README.md"
    readme_path.write_text(readme)
    print(f"✓ Created README: {readme_path}")

def create_development_scripts(project_root):
    """Create development helper scripts"""
    
    # Test runner script
    test_script = """#!/usr/bin/env python
\"\"\"Run AIASPRO tests\"\"\"

import subprocess
import sys
from pathlib import Path

def run_tests():
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "src/asammdf/plugins/aiaspro/tests"
    
    cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]
    
    print("Running AIASPRO tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()
"""
    
    script_path = project_root / "scripts/test_aiaspro.py"
    script_path.write_text(test_script)
    script_path.chmod(0o755)
    print(f"✓ Created test runner: {script_path}")
    
    # Development setup script
    dev_setup = """#!/usr/bin/env python
\"\"\"Setup AIASPRO development environment\"\"\"

import subprocess
import sys
from pathlib import Path

def setup_dev_env():
    project_root = Path(__file__).parent.parent
    
    print("Setting up AIASPRO development environment...")
    
    # Install AIASPRO requirements
    req_file = project_root / "src/asammdf/plugins/aiaspro/requirements.txt"
    print(f"\\nInstalling AIASPRO requirements from {req_file}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    
    # Install development requirements
    print("\\nInstalling development requirements...")
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
        
        print(f"\\n✓ Created sample config at {config_file}")
        print("  → Please update with your OpenAI API key")
    
    print("\\n✅ Development environment setup complete!")

if __name__ == "__main__":
    setup_dev_env()
"""
    
    dev_script_path = project_root / "scripts/setup_aiaspro_dev.py"
    dev_script_path.write_text(dev_setup)
    dev_script_path.chmod(0o755)
    print(f"✓ Created dev setup script: {dev_script_path}")

def main():
    """Main setup function"""
    print("AIASPRO Plugin Setup")
    print("=" * 50)
    
    # Create directory structure
    project_root = create_directory_structure()
    
    print("\nCreating configuration files...")
    
    # Create manifest
    create_plugin_manifest(project_root)
    
    # Create requirements
    create_requirements_file(project_root)
    
    # Create README
    create_readme(project_root)
    
    # Create development scripts
    create_development_scripts(project_root)
    
    print("\n✅ AIASPRO plugin structure created successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/setup_aiaspro_dev.py")
    print("2. Update ~/.asammdf/plugins/aiaspro/config.json with your API key")
    print("3. Start implementing the plugin components")

if __name__ == "__main__":
    main()