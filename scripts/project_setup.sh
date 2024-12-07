#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up project structure...${NC}"

# Create directory structure
directories=(
    "src/utils/quality_management"
    "tests"
    "docs"
    "scripts"
    "models"
    "cache/models"
    "cache/sessions"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo -e "${GREEN}Created directory: $dir${NC}"
done

# Create quality management module files
quality_management_files=(
    "__init__.py"
    "quality_manager.py"
    "basic_metrics.py"
    "processing_accuracy.py"
    "quality_improvement.py"
    "configuration.py"
    "performance_metrics.py"
)

for file in "${quality_management_files[@]}"; do
    touch "src/utils/quality_management/$file"
    echo -e "${GREEN}Created file: src/utils/quality_management/$file${NC}"
done

# Create test files
test_files=(
    "conftest.py"
    "test_basic_metrics.py"
    "test_processing_accuracy.py"
    "test_quality_improvement.py"
    "test_edge_cases.py"
    "test_configuration.py"
    "test_performance_metrics.py"
)

for file in "${test_files[@]}"; do
    touch "tests/$file"
    echo -e "${GREEN}Created file: tests/$file${NC}"
done

# Create documentation files
doc_files=(
    "architecture.md"
    "testing_architecture.md"
    "implementation_plans.md"
    "quickstart.md"
    "development_context.md"
)

for file in "${doc_files[@]}"; do
    touch "docs/$file"
    echo -e "${GREEN}Created file: docs/$file${NC}"
done

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << EOL
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
PyYAML>=5.4.0
pytest>=6.2.0
black>=21.5b2
flake8>=3.9.0
isort>=5.9.0
pylint>=2.8.0
streamlit>=1.0.0
psutil>=5.8.0
EOL
    echo -e "${GREEN}Created requirements.txt${NC}"
fi

# Create pyproject.toml if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
    cat > pyproject.toml << EOL
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q"
testpaths = [
    "tests",
]
EOL
    echo -e "${GREEN}Created pyproject.toml${NC}"
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}Initialized git repository${NC}"
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
env/
venv/
ENV/
.idea/
.vscode/
*.swp
.DS_Store
cache/
models/
*.log
EOL
        echo -e "${GREEN}Created .gitignore${NC}"
    fi
fi

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Setup test infrastructure
echo -e "\n${YELLOW}Setting up test infrastructure...${NC}"
python -m pytest --collect-only

# Initialize error handling
echo -e "\n${YELLOW}Setting up error handling...${NC}"
if [ ! -d "logs" ]; then
    mkdir logs
    echo -e "${GREEN}Created logs directory${NC}"
fi

# Final setup steps
echo -e "\n${YELLOW}Running final setup steps...${NC}"
# Run verify_and_fix.sh if it exists
if [ -f "scripts/verify_and_fix.sh" ]; then
    bash scripts/verify_and_fix.sh
fi

echo -e "\n${GREEN}Project setup completed successfully!${NC}"
echo -e "You can now start developing by:"
echo -e "1. Activating your virtual environment"
echo -e "2. Running 'streamlit run streamlit_app.py' to start the application"
echo -e "3. Running 'pytest' to execute tests"
