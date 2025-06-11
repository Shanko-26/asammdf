#!/usr/bin/env python3
"""Safe installation of AIASPRO dependencies"""

import subprocess
import sys
from pathlib import Path

def check_current_deps():
    """Check current installed packages"""
    print("Current environment packages:")
    print("-" * 50)
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if any(pkg in line.lower() for pkg in ["numpy", "scipy", "scikit", "pandas", "asammdf"]):
            print(f"  {line}")
    print()

def check_compatibility(package):
    """Check if a package is compatible without installing"""
    print(f"Checking compatibility for {package}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", package],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  ❌ Potential conflict: {result.stderr}")
        return False
    else:
        # Check for numpy version changes
        if "numpy" in result.stdout:
            print(f"  ⚠️  May affect numpy version")
        print(f"  ✓ Appears compatible")
        return True

def install_package(package):
    """Install a single package"""
    print(f"\nInstalling {package}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"  ✓ Successfully installed {package}")
        return True
    else:
        print(f"  ❌ Failed to install {package}")
        print(f"     Error: {result.stderr}")
        return False

def main():
    """Main installation process"""
    print("AIASPRO Dependencies Installation")
    print("=" * 50)
    
    # Check current environment
    check_current_deps()
    
    # Read requirements
    req_file = Path(__file__).parent.parent / "src/asammdf/plugins/aiaspro/requirements.txt"
    with open(req_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    print(f"Required packages: {', '.join(requirements)}\n")
    
    # First, let's check what would happen
    print("Checking compatibility...")
    print("-" * 50)
    
    issues = []
    for req in requirements:
        if not check_compatibility(req):
            issues.append(req)
    
    if issues:
        print(f"\n⚠️  Potential issues with: {', '.join(issues)}")
        response = input("\nDo you want to proceed anyway? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Install packages
    print("\nProceeding with installation...")
    print("-" * 50)
    
    # Install in order of least likely to cause conflicts
    install_order = [
        "cryptography>=3.4.0",     # Usually no conflicts
        "openai>=1.0.0",           # Minimal dependencies
        "pydantic-ai>=0.0.7",      # Should be safe
        "scipy>=1.10.0",           # May require specific numpy
        "scikit-learn>=1.3.0",     # May require specific numpy/scipy
    ]
    
    failed = []
    for package in install_order:
        if not install_package(package):
            failed.append(package)
    
    # Summary
    print("\n" + "=" * 50)
    if failed:
        print(f"❌ Failed to install: {', '.join(failed)}")
        print("\nTroubleshooting:")
        print("1. Try updating pip: pip install --upgrade pip")
        print("2. Install failed packages individually")
        print("3. Check for specific version conflicts")
    else:
        print("✅ All dependencies installed successfully!")
        
        # Also install dev dependencies
        print("\nInstalling development dependencies...")
        dev_deps = ["pytest", "pytest-asyncio", "pytest-qt", "pytest-cov"]
        for dep in dev_deps:
            install_package(dep)
        
        print("\n✅ Installation complete!")
        print("\nNext steps:")
        print("1. Configure your OpenAI API key in ~/.asammdf/plugins/aiaspro/config.json")
        print("2. Run: python scripts/test_aiaspro_plugin.py")

if __name__ == "__main__":
    main()