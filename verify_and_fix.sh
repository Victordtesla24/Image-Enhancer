#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check and fix Python environment
check_python_env() {
    echo -e "${YELLOW}Verifying Python environment...${NC}"
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found. Creating...${NC}"
        python3 -m venv venv
        source venv/bin/activate
    fi

    # Update pip and dependencies
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo -e "${RED}requirements.txt not found. Creating...${NC}"
        pip freeze > requirements.txt
    fi
}

# Function to verify and fix project structure
verify_project_structure() {
    echo -e "${YELLOW}Verifying project structure...${NC}"
    
    # Create missing directories
    directories=("src/components" "src/utils" "tests/unit" "tests/integration" "data" "models" "docs")
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            echo -e "${RED}Creating missing directory: $dir${NC}"
            mkdir -p "$dir"
        fi
    done

    # Verify __init__.py files
    init_files=("src/__init__.py" "src/components/__init__.py" "src/utils/__init__.py" "tests/__init__.py")
    for file in "${init_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Creating missing file: $file${NC}"
            touch "$file"
        fi
    done
}

# Function to verify and fix code style
fix_code_style() {
    echo -e "${YELLOW}Fixing code style...${NC}"
    
    # Run black formatter
    black src/ tests/ || true
    
    # Run isort
    isort src/ tests/ || true
    
    # Run pylint
    pylint src/ tests/ || true
}

# Function to verify and fix Git configuration
verify_git_config() {
    echo -e "${YELLOW}Verifying Git configuration...${NC}"
    
    if [ ! -d ".git" ]; then
        echo -e "${RED}Git repository not initialized. Initializing...${NC}"
        git init
    fi

    if [ ! -f ".gitignore" ]; then
        echo -e "${RED}Creating .gitignore file...${NC}"
        cat > .gitignore << EOL
venv/
__pycache__/
*.pyc
.env
.coverage
.pytest_cache/
*.log
.DS_Store
EOL
    fi
}

# Function to verify Streamlit configuration
verify_streamlit_config() {
    echo -e "${YELLOW}Verifying Streamlit configuration...${NC}"
    
    if [ ! -d ".streamlit" ]; then
        mkdir .streamlit
        cat > .streamlit/config.toml << EOL
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
EOL
    fi
}

# Function to verify and fix tests
verify_tests() {
    echo -e "${YELLOW}Verifying test configuration...${NC}"
    
    if [ ! -f "pytest.ini" ]; then
        echo -e "${RED}Creating pytest.ini...${NC}"
        cat > pytest.ini << EOL
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --verbose --cov=src
EOL
    fi
}

# Main execution
main() {
    echo -e "${GREEN}Starting verification and fix process...${NC}"
    
    check_python_env
    verify_project_structure
    fix_code_style
    verify_git_config
    verify_streamlit_config
    verify_tests
    
    echo -e "${GREEN}Verification and fix process completed!${NC}"
}

main 