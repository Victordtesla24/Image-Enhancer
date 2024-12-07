#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Verifying project structure and dependencies..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python version $python_version is compatible${NC}"
else
    echo -e "${RED}✗ Python version $python_version is not compatible. Required: >= $required_version${NC}"
    exit 1
fi

# Check CUDA availability
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}✓ CUDA is available${NC}"
else
    echo -e "${YELLOW}! CUDA is not available. GPU acceleration will be disabled${NC}"
fi

# Check required directories
required_dirs=("src" "tests" "docs" "models" "data")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ Directory $dir exists${NC}"
    else
        echo -e "${RED}✗ Creating missing directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Check required files
required_files=("requirements.txt" "setup.py" "README.md" "docs/implementation_plans.md")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ File $file exists${NC}"
    else
        echo -e "${RED}✗ Missing required file: $file${NC}"
        exit 1
    fi
done

# Install/upgrade dependencies
echo "📦 Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run linting
echo "🔍 Running code quality checks..."
flake8 src tests
black src tests --check
mypy src

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v

# Check models directory
echo "🤖 Checking AI models..."
models_dir="models"
required_models=("super_resolution" "style_transfer" "detail_enhancement" "artifact_removal")
for model in "${required_models[@]}"; do
    if [ -d "$models_dir/$model" ]; then
        echo -e "${GREEN}✓ Model $model exists${NC}"
    else
        echo -e "${YELLOW}! Model $model not found. Will be downloaded during runtime${NC}"
    fi
done

# Check for large file support
if git lfs status >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Git LFS is configured${NC}"
else
    echo -e "${RED}✗ Git LFS is not configured. Required for model storage${NC}"
    exit 1
fi

# Verify GPU memory
if command -v nvidia-smi &> /dev/null; then
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if [ "$gpu_memory" -ge 8000 ]; then
        echo -e "${GREEN}✓ GPU memory is sufficient ($gpu_memory MB)${NC}"
    else
        echo -e "${YELLOW}! GPU memory might be insufficient for 5K processing ($gpu_memory MB)${NC}"
    fi
fi

echo "✨ Verification complete!" 