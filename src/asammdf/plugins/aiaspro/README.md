# AI Assistant Pro (AIASPRO) Plugin for asammdf

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
