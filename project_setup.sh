#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🚀 Setting up Image Enhancer project..."

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install base dependencies
echo "📦 Installing base dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Initialize git repository if not exists
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git lfs install
fi

# Create project structure
echo "📁 Creating project structure..."
mkdir -p src/{components,utils/{core,quality_management}}
mkdir -p tests
mkdir -p docs
mkdir -p models/{super_resolution,style_transfer,detail_enhancement,artifact_removal}
mkdir -p data/{raw,processed,temp}

# Create documentation files
echo "📚 Creating documentation..."
cat > README.md << EOL
# Image Enhancer

Advanced image enhancement system with AI-powered features and 5K resolution support.

## Features

- AI-powered image enhancement
- 5K resolution support
- Real-time quality feedback
- Adaptive enhancement pipeline
- Multi-model processing

## Requirements

- Python 3.8+
- CUDA capable GPU (8GB+ VRAM recommended)
- Git LFS

## Installation

\`\`\`bash
./project_setup.sh
\`\`\`

## Usage

\`\`\`bash
python main.py --input path/to/image --output path/to/output
\`\`\`

## Documentation

See \`docs/\` directory for detailed documentation.
EOL

# Create requirements file if not exists
if [ ! -f "requirements.txt" ]; then
    echo "📝 Creating requirements.txt..."
    cat > requirements.txt << EOL
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.7.0
pillow>=9.0.0
scikit-image>=0.19.0
tensorflow>=2.12.0
onnxruntime-gpu>=1.14.0
openvino>=2023.0.0
pytest>=7.0.0
black>=22.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
EOL
fi

# Create setup.py
echo "📝 Creating setup.py..."
cat > setup.py << EOL
from setuptools import setup, find_packages

setup(
    name="image-enhancer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced image enhancement system with AI-powered features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-enhancer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
EOL

# Create main application file
echo "📝 Creating main application file..."
cat > main.py << EOL
"""Main application entry point."""

import argparse
import logging
from pathlib import Path

from src.utils.core.processor import Processor
from src.components.user_interface import ProgressUI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Image Enhancer")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--quality", type=float, default=0.95, help="Target quality (0-1)")
    parser.add_argument("--resolution", type=str, default="5k", help="Target resolution")
    args = parser.parse_args()

    # Initialize components
    processor = Processor()
    progress_ui = ProgressUI()

    try:
        # Process image
        processor.process_image(
            args.input,
            args.output,
            target_quality=args.quality,
            target_resolution=args.resolution,
            progress_callback=progress_ui.update_progress
        )
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
EOL

# Create pre-commit config
echo "📝 Creating pre-commit config..."
cat > .pre-commit-config.yaml << EOL
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]
EOL

# Initialize pre-commit
echo "🔧 Initializing pre-commit..."
pre-commit install

# Create initial git commit
echo "💾 Creating initial commit..."
git add .
git commit -m "Initial project setup"

echo "✨ Project setup complete!"
echo "🚀 Run 'source venv/bin/activate' to activate the virtual environment"
</code_block_to_apply_changes_from> 