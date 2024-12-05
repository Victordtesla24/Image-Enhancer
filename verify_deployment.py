#!/usr/bin/env python3
"""
Deployment verification script for Image-Enhancer
Checks all necessary components and requirements before deployment
"""

import os
import sys
import pkg_resources
import yaml
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python version meets requirements"""
    required_version = (3, 11)
    current_version = sys.version_info[:2]

    print(f"\n✓ Checking Python version:")
    if current_version >= required_version:
        print(
            f"  ✓ Python version {'.'.join(map(str, current_version))} meets requirement"
        )
        return True
    else:
        print(
            f"  ✗ Python version {'.'.join(map(str, current_version))} does not meet minimum requirement of {'.'.join(map(str, required_version))}"
        )
        return False


def check_dependencies():
    """Verify all required packages are installed"""
    print("\n✓ Checking dependencies:")
    required_packages = [
        "opencv-python",
        "numpy",
        "Pillow",
        "PyYAML",
        "pytest",
        "pytest-cov",
    ]

    all_installed = True
    for package in required_packages:
        try:
            pkg_resources.require(package)
            print(f"  ✓ {package} is installed")
        except pkg_resources.DistributionNotFound:
            print(f"  ✗ {package} is not installed")
            all_installed = False
    return all_installed


def check_critical_files():
    """Verify all critical files exist"""
    print("\n✓ Checking critical files:")
    critical_files = [
        "config/5k_quality_settings.yaml",
        "src/utils/image_processor.py",
        "enhance_and_verify.py",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} is missing")
            all_exist = False
    return all_exist


def check_config():
    """Verify configuration file is valid"""
    print("\n✓ Checking configuration:")
    try:
        with open("config/5k_quality_settings.yaml", "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["resolution", "color", "quality", "enhancement"]
        all_keys_present = True
        for key in required_keys:
            if key in config:
                print(f"  ✓ Configuration contains {key} settings")
            else:
                print(f"  ✗ Configuration missing {key} settings")
                all_keys_present = False
        return all_keys_present
    except Exception as e:
        print(f"  ✗ Error reading configuration: {str(e)}")
        return False


def run_tests():
    """Run test suite"""
    print("\n✓ Running tests:")
    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--cov=src/"], capture_output=True, text=True
        )
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ Error running tests: {str(e)}")
        return False


def verify_image_processor():
    """Verify ImageProcessor can be imported and initialized"""
    print("\n✓ Verifying ImageProcessor:")
    try:
        from src.utils.image_processor import ImageProcessor

        processor = ImageProcessor()
        print("  ✓ ImageProcessor successfully initialized")
        return True
    except Exception as e:
        print(f"  ✗ Error initializing ImageProcessor: {str(e)}")
        return False


def main():
    """Run all deployment verification checks"""
    print("=== Image-Enhancer Deployment Verification ===")

    checks = [
        check_python_version(),
        check_dependencies(),
        check_critical_files(),
        check_config(),
        verify_image_processor(),
        run_tests(),
    ]

    print("\n=== Verification Summary ===")
    if all(checks):
        print("\n✅ All checks passed - Ready for deployment!")
        return 0
    else:
        print("\n❌ Some checks failed - Please fix issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
