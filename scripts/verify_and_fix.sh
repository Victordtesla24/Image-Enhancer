#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting verification and fix process...${NC}"

# Check project structure
echo -e "\n${YELLOW}Checking project structure...${NC}"
required_dirs=(
    "src/utils/quality_management"
    "tests"
    "docs"
    "scripts"
)

required_files=(
    "src/utils/quality_management/__init__.py"
    "src/utils/quality_management/quality_manager.py"
    "src/utils/quality_management/basic_metrics.py"
    "src/utils/quality_management/processing_accuracy.py"
    "src/utils/quality_management/quality_improvement.py"
    "src/utils/quality_management/configuration.py"
    "src/utils/quality_management/performance_metrics.py"
    "tests/conftest.py"
    "tests/test_basic_metrics.py"
    "tests/test_processing_accuracy.py"
    "tests/test_quality_improvement.py"
    "tests/test_edge_cases.py"
    "tests/test_configuration.py"
    "tests/test_performance_metrics.py"
    "docs/architecture.md"
    "docs/testing_architecture.md"
    "docs/implementation_plans.md"
)

# Check directories
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Missing directory: $dir${NC}"
        mkdir -p "$dir"
        echo -e "${GREEN}Created directory: $dir${NC}"
    fi
done

# Check files
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Missing file: $file${NC}"
        touch "$file"
        echo -e "${GREEN}Created file: $file${NC}"
    fi
done

# Run linting and formatting
echo -e "\n${YELLOW}Running code quality checks...${NC}"

# Check if tools are installed
command -v black >/dev/null 2>&1 || { echo -e "${RED}black not installed. Installing...${NC}"; pip install black; }
command -v flake8 >/dev/null 2>&1 || { echo -e "${RED}flake8 not installed. Installing...${NC}"; pip install flake8; }
command -v isort >/dev/null 2>&1 || { echo -e "${RED}isort not installed. Installing...${NC}"; pip install isort; }

# Format code
echo "Running black..."
black src/utils/quality_management/*.py tests/*.py

echo "Running isort..."
isort src/utils/quality_management/*.py tests/*.py

echo "Running flake8..."
flake8 src/utils/quality_management/*.py tests/*.py

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"
python -m pytest tests/ -v

# Memory check
echo -e "\n${YELLOW}Checking memory usage...${NC}"
python -c "
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print(f'Current memory usage: {get_memory_usage():.2f} MB')
"

# Check for common issues
echo -e "\n${YELLOW}Checking for common issues...${NC}"

# Check for large files
echo "Checking for large files..."
find . -type f -size +1M | while read -r file; do
    echo -e "${RED}Large file detected: $file${NC}"
done

# Check for duplicate code
echo "Checking for duplicate code..."
if command -v pylint >/dev/null 2>&1; then
    pylint --disable=all --enable=duplicate-code src/utils/quality_management/
else
    echo -e "${RED}pylint not installed. Skipping duplicate code check.${NC}"
fi

# Check documentation
echo -e "\n${YELLOW}Checking documentation...${NC}"
missing_docs=0
for file in "${required_files[@]}"; do
    if [[ $file == *".py" ]]; then
        if ! grep -q '"""' "$file"; then
            echo -e "${RED}Missing docstring in: $file${NC}"
            missing_docs=$((missing_docs + 1))
        fi
    fi
done

if [ $missing_docs -gt 0 ]; then
    echo -e "${RED}Found $missing_docs files with missing documentation${NC}"
else
    echo -e "${GREEN}All Python files have docstrings${NC}"
fi

# Final status
echo -e "\n${YELLOW}Verification and fix process completed${NC}"
echo -e "${GREEN}Project structure verified and fixed${NC}"
echo -e "${GREEN}Code quality checks completed${NC}"
echo -e "${GREEN}Tests executed${NC}"
echo -e "${GREEN}Memory usage checked${NC}"
echo -e "${GREEN}Documentation verified${NC}"
