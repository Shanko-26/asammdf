# AIASPRO Development Scripts

This folder contains essential scripts for AIASPRO (AI Assistant Pro) development and testing.

## 🚀 Essential Scripts

### `test_real_mdf_fixed.py` ⭐ **MAIN TEST**
**Status**: ✅ Working with real MDF data
**Purpose**: Tests the complete AIASPRO system with your actual `my_sample.mf4` file

```bash
python3 scripts/test_real_mdf_fixed.py
```

**What it does:**
- Loads real MDF file (41 ECM channels)
- Tests AI agents with automotive data
- Validates dependency injection
- Tests orchestrator with real queries
- Proves end-to-end functionality

**Output**: Confirms real data analysis with engine speed, temperatures, pressures

---

### `test_fixed_types.py` 🔧 **DEPENDENCY TEST**
**Status**: ✅ Working
**Purpose**: Tests the fixed dependency injection system

```bash
python3 scripts/test_fixed_types.py  
```

**What it tests:**
- Dataclass dependencies creation
- PydanticAI tool registration
- Type hint corrections (`RunContext[AIASPRODependencies]`)
- Tool-to-dependency communication

---

### `setup_aiaspro_dev.py` 📦 **DEVELOPMENT SETUP**
**Status**: ✅ Working
**Purpose**: Sets up AIASPRO development environment

```bash
python3 scripts/setup_aiaspro_dev.py
```

**What it does:**
- Creates plugin directory structure
- Sets up configuration files
- Installs Python dependencies
- Configures API keys

---

### `install_aiaspro_deps.py` 📥 **DEPENDENCY INSTALLER**
**Status**: ✅ Working  
**Purpose**: Installs required dependencies for AIASPRO

```bash
python3 scripts/install_aiaspro_deps.py
```

**What it installs:**
- PydanticAI for agent framework
- OpenAI API client
- Additional AI/ML dependencies

---

### `setup_aiaspro.py` ⚙️ **BASIC SETUP**
**Status**: ✅ Working
**Purpose**: Basic AIASPRO setup and configuration

```bash
python3 scripts/setup_aiaspro.py
```

## 🧪 Development Workflow

1. **First time setup:**
   ```bash
   python3 scripts/setup_aiaspro_dev.py
   python3 scripts/install_aiaspro_deps.py
   ```

2. **Test dependency injection:**
   ```bash
   python3 scripts/test_fixed_types.py
   ```

3. **Test with real data:**
   ```bash
   python3 scripts/test_real_mdf_fixed.py
   ```

## 🎯 What's Working

- ✅ Real MDF file loading (`my_sample.mf4` with 41 channels)
- ✅ PydanticAI dependency injection
- ✅ Multi-agent orchestration
- ✅ Automotive data analysis tools
- ✅ Type-safe AI tool communication

## 📊 Real Data Capabilities

The working system can analyze:
- **Engine Data**: ECM_EngineSpeed (1093-5838 RPM)
- **Temperature**: ECM_IntakeAirTemp, ECM_CoolantTemp  
- **Pressure**: ECM_OilPressure, ECM_FuelRailPressure
- **Flow**: ECM_MAFRate (Mass Air Flow)
- **12,000+ data points per channel**

## 🚀 Next Steps

- Integrate with Qt GUI
- Add specialized agents (plotting, requirements)
- Enhance automotive analysis features