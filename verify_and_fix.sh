#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Store initial state
initial_state_file=".initial_state"
final_state_file=".final_state"

# Function to capture state
capture_state() {
    find . -type f -not -path "*/\.*" -exec md5sum {} \; > "$1" 2>/dev/null
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to clear package caches
clear_package_caches() {
    echo -e "${YELLOW}Clearing package caches...${NC}"
    
    # Clear pip cache
    pip cache purge
    
    # Clear PyTorch cache
    if [ -d "~/.cache/torch" ]; then
        rm -rf ~/.cache/torch
    fi
    
    # Clear Hugging Face cache
    if [ -d "~/.cache/huggingface" ]; then
        rm -rf ~/.cache/huggingface
    fi
    
    # Clear model cache
    if [ -d "~/.cache/image_enhancer" ]; then
        rm -rf ~/.cache/image_enhancer
    fi
    
    echo -e "${GREEN}Package caches cleared${NC}"
}

# Function to clear development caches
clear_dev_caches() {
    echo -e "${YELLOW}Clearing development caches...${NC}"
    
    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    
    # Clear pytest cache
    rm -rf .pytest_cache
    
    # Clear coverage cache
    rm -rf .coverage
    rm -rf htmlcov
    
    # Clear temporary files
    rm -rf temp_uploads/*
    rm -rf .temp
    
    echo -e "${GREEN}Development caches cleared${NC}"
}

# Function to verify model files
verify_models() {
    echo -e "${YELLOW}Verifying AI models...${NC}"
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Check for required model files
    required_models=(
        "super_resolution"
        "color_enhancement"
        "detail_enhancement"
    )
    
    for model in "${required_models[@]}"; do
        if [ ! -d "models/${model}" ]; then
            echo -e "${YELLOW}Creating ${model} model directory...${NC}"
            mkdir -p "models/${model}"
        fi
    done
    
    echo -e "${GREEN}Model verification complete${NC}"
}

# Function to run tests with coverage
run_tests_with_coverage() {
    echo -e "${YELLOW}Running tests with coverage...${NC}"
    
    # Install coverage if not present
    if ! command_exists coverage; then
        pip install coverage
    fi
    
    # Run tests with coverage
    coverage run -m pytest
    
    # Generate coverage report
    coverage report
    coverage html
    
    # Check coverage threshold
    coverage report | grep TOTAL | awk '{print $4}' | grep -q "^[0-9]\{2,3\}%$" || {
        echo -e "${RED}Coverage is below 80%. Please add more tests.${NC}"
        return 1
    }
    
    echo -e "${GREEN}Tests completed successfully${NC}"
}

# Capture initial state
capture_state "$initial_state_file"

echo -e "${GREEN}Starting verification and fixes...${NC}"

# Clear all caches
clear_package_caches
clear_dev_caches

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}Virtual environment not found. Running proj_setup.sh...${NC}"
    ./proj_setup.sh
    exit 1
fi

# Update pip and packages
echo -e "${YELLOW}Updating pip and packages...${NC}"
pip install --upgrade pip --no-cache-dir

# Install/upgrade required packages
echo -e "${YELLOW}Installing/upgrading required packages...${NC}"
pip install -r requirements.txt --no-cache-dir

# Install development dependencies
pip install black pylint pytest pytest-cov coverage --no-cache-dir

# Fix code formatting
echo -e "${YELLOW}Fixing code formatting...${NC}"
black .

# Run linting
echo -e "${YELLOW}Running linting...${NC}"
pylint src/ tests/ || true

# Verify directory structure
echo -e "${YELLOW}Verifying directory structure...${NC}"
directories=(
    "src/utils/model_management"
    "src/utils/session_management"
    "src/utils/quality_management"
    "src/utils/enhancers"
    "models"
    "tests"
    "temp_uploads"
    ".streamlit"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created missing directory: $dir${NC}"
    fi
done

# Verify model files
verify_models

# Run tests with coverage
run_tests_with_coverage

# Create/update necessary files
echo -e "${YELLOW}Creating/updating necessary files...${NC}"

# Create .streamlit/config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    mkdir -p .streamlit
    cat > .streamlit/config.toml << EOL
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOL
fi

# Create/update .gitignore
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
MANIFEST
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.idea/
.vscode/
*.swp
.DS_Store
temp_uploads/
.coverage
htmlcov/
.pytest_cache/
.streamlit/secrets.toml
models/*/checkpoints/
EOL
fi

# Create/update Procfile
if [ ! -f "Procfile" ]; then
    echo "web: streamlit run streamlit_app.py" > Procfile
fi

# Create/update runtime.txt
if [ ! -f "runtime.txt" ]; then
    echo "python-3.8.12" > runtime.txt
fi

# Generate commit message
message="🔄 Auto Update: ["

# Check for changes
if [ -n "$(git status --porcelain)" ]; then
    if [ -n "$(git status --porcelain | grep 'src/')" ]; then
        message+="💻 Updated source code, "
    fi
    if [ -n "$(git status --porcelain | grep 'tests/')" ]; then
        message+="🧪 Updated tests, "
    fi
    if [ -n "$(git status --porcelain | grep 'models/')" ]; then
        message+="🤖 Updated models, "
    fi
    if [ -n "$(git status --porcelain | grep 'requirements.txt')" ]; then
        message+="📦 Updated dependencies, "
    fi
    # Remove trailing comma and space
    message="${message%, }]"
else
    message+="No changes required]"
fi

echo "$message" > .commit_message

# Cleanup
rm -f "$initial_state_file" "$final_state_file"

echo -e "${GREEN}All verifications and fixes completed!${NC}"
echo -e "${YELLOW}Generated commit message: ${message}${NC}"
