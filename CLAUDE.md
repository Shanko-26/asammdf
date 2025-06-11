# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

asammdf is a fast parser and editor for ASAM MDF (Measurement Data Format) files supporting versions 2 (.dat), 3 (.mdf), and 4 (.mf4). It provides both a Python API and GUI for working with automotive measurement data files.

## Standard rules
1. Always give me commands to run scripts instead of running them on your own. The reason is,
that I have a proper venv set and you dont. 
2. Refer Pydantic AI docs online if you are ever in doubt. https://ai.pydantic.dev/llms.txt

## Development Commands

### Testing
```bash
# Run all tests
pytest --cov --cov-report=lcov

# Run specific test file
pytest test/test_mdf.py

# Run with specific Python version using tox
tox -e py310,py311,py312,py313
```

### Code Quality
```bash
# Run type checking
mypy

# Run linting and formatting
ruff check
ruff format --check

# Fix formatting issues
ruff format

# Run all quality checks
tox -e mypy,ruff
```

### Building
```bash
# Build wheel (requires scikit-build-core)
pip install build
python -m build

# Install in development mode with all extras
pip install --editable .[decode,encryption,export,export-matlab-v5,filesystem,gui,plot,symbolic-math]
```

### Documentation
```bash
# Build documentation
tox -e doc
# Or directly: sphinx-build --builder html --nitpicky doc doc/_build/html
```

## Architecture

### Core Components

- **MDF Class** (`src/asammdf/mdf.py`): Main interface for reading/writing MDF files
- **Signal Class** (`src/asammdf/signal.py`): Unified signal processing for all MDF versions
- **Block Parsers** (`src/asammdf/blocks/`): Version-specific MDF block implementations
  - `mdf_v2.py`, `mdf_v3.py`, `mdf_v4.py`: Version-specific parsers
  - `v2_v3_blocks.py`, `v4_blocks.py`: Block structure definitions
  - `v2_v3_constants.py`, `v4_constants.py`: Format constants

### GUI Components

- **Main Application** (`src/asammdf/app/asammdfgui.py`): Entry point for GUI
- **Widgets** (`src/asammdf/gui/widgets/`): Reusable UI components
- **Dialogs** (`src/asammdf/gui/dialogs/`): Modal dialogs for specific operations
- **UI Files** (`src/asammdf/gui/ui/`): Qt Designer .ui files and generated Python code

### Performance Optimizations

- **C Extensions** (`src/asammdf/blocks/cutils.c`): Critical path optimizations
- The library gracefully falls back to Python implementations if C extensions fail to load
- Check `__cextension__` attribute to verify C extension availability

### File Format Support

- **MDF v2/v3**: Legacy format support with some limitations
- **MDF v4**: Full support including compression, arrays, and bus logging
- **Bus Logging**: CAN/LIN bus data extraction with database support (.dbc/.arxml)

## Key Design Patterns

- **Version Abstraction**: MDF class provides unified interface across all MDF versions
- **Signal Processing**: Signal class handles time-domain operations independently of MDF version  
- **Lazy Loading**: Data is loaded on-demand to handle large files efficiently
- **Memory Mapping**: Uses mmap for efficient large file access

## Testing Strategy

- Tests are organized by functionality in `test/` directory
- GUI tests use PyAutoGUI for automation
- Separate test requirements in `test/requirements.txt`
- Coverage reporting enabled by default
- Timeout protection (600s) for long-running tests

## AIASPRO Plugin Status

### ✅ Completed Features
- **Plugin System Foundation**: Complete plugin architecture with discovery and loading
- **AIASPRO Core Infrastructure**: Dependencies injection, agent orchestration, and PydanticAI integration
- **Real MDF Integration**: Full support for loading and analyzing real automotive data files
- **Multi-Agent System**: General analysis agent with automotive data tools
- **Type-Safe AI Tools**: Proper dependency injection using RunContext[AIASPRODependencies]

### 🔧 Current Architecture

```
src/asammdf/plugins/aiaspro/
├── core/
│   ├── dependencies.py     # Dependency injection container (dataclass)
│   └── orchestrator.py     # Multi-agent orchestration system
├── agents/
│   ├── base_agent.py      # Base agent with PydanticAI integration
│   └── general_agent.py   # General analysis agent with automotive tools
└── services/
    └── mdf_data_service.py # MDF data access service
```

### 🧪 Testing Commands

```bash
# Test real MDF integration with your file
python3 scripts/test_real_mdf_fixed.py

# Test fixed dependency injection
python3 scripts/test_fixed_types.py
```

### 🎯 Key Technical Achievements
1. **Dependency Injection Working**: Fixed tool registration timing and type hints
2. **Real MDF Data**: Successfully loads `my_sample.mf4` with 41 channels of ECM data
3. **PydanticAI Integration**: Proper agent-tool communication with RunContext
4. **C Extensions Built**: asammdf cutils module compiled for performance
5. **Type Safety**: Full type annotations with dataclass dependencies

### 📊 Real Data Analysis Capabilities
- Engine speed analysis (1093-5838 RPM range)
- Temperature monitoring (ECM_IntakeAirTemp, ECM_CoolantTemp)
- Pressure analysis (ECM_OilPressure, ECM_FuelRailPressure)
- Automotive signal categorization and search
- Statistical analysis with 12,000+ data points per channel

### ✅ GUI Integration Complete (December 2024)
- **Qt GUI Integration**: AI Assistant Pro fully integrated into asammdf's MDI system
- **Menu Integration**: "AI Assistant Pro" menu with keyboard shortcuts (Ctrl+Shift+A)
- **File Context Awareness**: Automatically detects loaded MDF files and provides contextual analysis
- **Real-time Chat Interface**: Interactive chat with automotive data analysis capabilities
- **MDI Window Management**: Proper focus and window handling in asammdf's interface

### 🎯 Live Demo Features Working
- **Engine Performance Analysis**: Real-time analysis of ECM data (RPM, torque, temperature)
- **Channel Search and Categorization**: Intelligent automotive signal grouping
- **Statistical Analysis**: Min/max/mean calculations with 12,000+ data points
- **Interactive Chat**: Natural language queries about automotive data
- **Auto-suggestions**: Context-aware analysis recommendations

### 🚀 Next Development Phase
- Specialized agents (plotting, requirements analysis, export)
- Advanced automotive pattern detection and diagnostics
- Multi-file comparison and benchmarking features