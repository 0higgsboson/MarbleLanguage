#!/usr/bin/env python3
"""
Quick dependency installer for Marble Language v2
Checks for and installs required packages for full functionality
"""

import subprocess
import sys
import importlib


def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("Marble Language v2 - Dependency Installer")
    print("=" * 50)
    
    # Required packages for full functionality
    packages = {
        'torch': 'torch>=2.0.0',
        'numpy': 'numpy>=1.21.0', 
        'matplotlib': 'matplotlib>=3.3.0',
        'tqdm': 'tqdm>=4.60.0'
    }
    
    missing_packages = []
    
    # Check what's installed
    print("Checking installed packages:")
    for package, pip_name in packages.items():
        if check_package(package):
            print(f"  ✓ {package} - already installed")
        else:
            print(f"  ❌ {package} - not installed")
            missing_packages.append(pip_name)
    
    if not missing_packages:
        print("\n🎉 All packages are already installed!")
        print("You can now use all Marble Language v2 features including:")
        print("  - Real-time loss plotting")
        print("  - Training database")
        print("  - Progress bars")
        return
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    
    # Install missing packages
    failed_packages = []
    for package in missing_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"  ✓ {package} installed successfully")
        else:
            print(f"  ❌ Failed to install {package}")
            failed_packages.append(package)
    
    # Final status
    print("\n" + "=" * 50)
    if failed_packages:
        print("⚠️  Installation completed with some failures:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nTry installing manually with:")
        print(f"  pip install {' '.join(failed_packages)}")
    else:
        print("🎉 All packages installed successfully!")
        print("\nYou can now run training with full visualization:")
        print("  python3 marble_transformer_pretraining.py")
    
    # Show feature status
    print("\nFeature availability:")
    if check_package('matplotlib'):
        print("  ✓ Real-time plotting: Available")
    else:
        print("  ❌ Real-time plotting: Not available (matplotlib missing)")
    
    if check_package('torch'):
        print("  ✓ Neural network training: Available")
    else:
        print("  ❌ Neural network training: Not available (torch missing)")
    
    if check_package('tqdm'):
        print("  ✓ Progress bars: Available")
    else:
        print("  ❌ Progress bars: Not available (tqdm missing)")


if __name__ == "__main__":
    main()