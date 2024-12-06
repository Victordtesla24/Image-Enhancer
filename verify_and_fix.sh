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

# Function to generate commit message
generate_commit_message() {
    local message="🔄 Auto Update: "
    local changes=()

    # Capture changes in dependencies
    if [ -f "requirements.txt" ]; then
        if ! cmp -s "requirements.txt" "requirements.txt.bak" 2>/dev/null; then
            changes+=("📦 Updated dependencies")
        fi
    fi

    # Check for Streamlit file changes
    if [ -f "streamlit_app.py" ] && [ -f "streamlit_app.py.bak" ]; then
        if ! cmp -s "streamlit_app.py" "streamlit_app.py.bak"; then
            changes+=("🎨 Modified Streamlit app")
        fi
    fi

    # Check directory structure changes
    if [ -n "$(git status --porcelain | grep '?? /')" ]; then
        changes+=("📁 Updated directory structure")
    fi

    # Check for config changes
    if [ -f ".streamlit/config.toml" ] && [ -f ".streamlit/config.toml.bak" 2>/dev/null ]; then
        if ! cmp -s ".streamlit/config.toml" ".streamlit/config.toml.bak"; then
            changes+=("⚙️ Modified configuration")
        fi
    fi

    # Check for documentation changes
    if git status --porcelain | grep -q "docs/"; then
        changes+=("📚 Updated documentation")
    fi

    # Check for test changes
    if git status --porcelain | grep -q "tests/"; then
        changes+=("🧪 Modified tests")
    fi

    # Check for source code changes
    if git status --porcelain | grep -q "src/"; then
        changes+=("💻 Updated source code")
    fi

    # If no specific changes detected, add generic update message
    if [ ${#changes[@]} -eq 0 ]; then
        changes+=("🔧 General maintenance and improvements")
    fi

    # Combine all changes into commit message
    message+="["
    for i in "${!changes[@]}"; do
        if [ $i -gt 0 ]; then
            message+=", "
        fi
        message+="${changes[$i]}"
    done
    message+="]"

    echo "$message"
}

# Backup current state
backup_files() {
    if [ -f "requirements.txt" ]; then
        cp requirements.txt requirements.txt.bak 2>/dev/null
    fi
    if [ -f "streamlit_app.py" ]; then
        cp streamlit_app.py streamlit_app.py.bak 2>/dev/null
    fi
    if [ -f ".streamlit/config.toml" ]; then
        cp .streamlit/config.toml .streamlit/config.toml.bak 2>/dev/null
    fi
}

# Cleanup backup files
cleanup_backups() {
    rm -f requirements.txt.bak streamlit_app.py.bak .streamlit/config.toml.bak
    rm -f "$initial_state_file" "$final_state_file"
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

# Capture initial state
backup_files
capture_state "$initial_state_file"

echo -e "${GREEN}Starting verification and fixes...${NC}"

# Clear all caches before starting
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

# Clean uninstall of key packages
echo -e "${YELLOW}Cleaning package installations...${NC}"
pip uninstall -y torch torchvision super-image huggingface-hub

# Reinstall packages with compatible versions
echo -e "${YELLOW}Installing packages with compatible versions...${NC}"
pip install torch==2.0.1 torchvision==0.15.2 --no-cache-dir
pip install huggingface-hub==0.8.1 --no-cache-dir
pip install super-image==0.1.7 --no-cache-dir

# Install Streamlit if not present
if ! pip show streamlit > /dev/null; then
    echo -e "${YELLOW}Installing Streamlit...${NC}"
    pip install streamlit --no-cache-dir
fi

# Install all requirements fresh
pip install -r requirements.txt --no-deps --no-cache-dir

# Fix code formatting
echo -e "${YELLOW}Fixing code formatting...${NC}"
if command_exists black; then
    black .
else
    pip install black --no-cache-dir
    black .
fi

# Run linting
echo -e "${YELLOW}Running linting...${NC}"
if command_exists pylint; then
    pylint src/ tests/ || true
else
    pip install pylint --no-cache-dir
    pylint src/ tests/ || true
fi

# Verify directory structure
echo -e "${YELLOW}Verifying directory structure...${NC}"
directories=("src" "tests" "assets" "models" "temp_uploads" ".streamlit")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created missing directory: $dir${NC}"
    fi
done

# Fix Streamlit app file structure
echo -e "${YELLOW}Fixing Streamlit app file structure...${NC}"

# Remove duplicate app files
if [ -f "src/app.py" ]; then
    rm src/app.py
fi
if [ -f "src/streamlit_app.py" ]; then
    rm src/streamlit_app.py
fi

# Update imports in streamlit_app.py
if [ -f "streamlit_app.py" ]; then
    echo -e "${YELLOW}Updating imports in streamlit_app.py...${NC}"
    sed -i '' 's/from src\./from /g' streamlit_app.py
fi

# Create .streamlit/config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    echo -e "${YELLOW}Creating Streamlit config...${NC}"
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

# Update requirements.txt with Streamlit-specific requirements
echo -e "${YELLOW}Updating requirements.txt with Streamlit requirements...${NC}"
pip freeze | grep -v "pkg-resources" > requirements.txt

# Ensure core dependencies are in requirements.txt
core_deps=(
    "streamlit"
    "pillow"
    "torch"
    "torchvision"
    "super-image"
    "huggingface-hub"
    "opencv-python-headless"
    "numpy"
)

for dep in "${core_deps[@]}"; do
    if ! grep -q "^$dep" requirements.txt; then
        pip install "$dep"
    fi
done

# Update requirements.txt again after ensuring core deps
pip freeze | grep -v "pkg-resources" > requirements.txt

# Remove redundant files
echo -e "${YELLOW}Cleaning up redundant files...${NC}"
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -r {} +

# Create Streamlit-specific files
echo -e "${YELLOW}Creating Streamlit-specific files...${NC}"

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
EOL
fi

# Create Procfile for Streamlit deployment if it doesn't exist
if [ ! -f "Procfile" ]; then
    echo "web: streamlit run streamlit_app.py" > Procfile
fi

# Create runtime.txt for Python version if it doesn't exist
if [ ! -f "runtime.txt" ]; then
    echo "python-3.8.12" > runtime.txt
fi

# Capture final state and generate commit message
capture_state "$final_state_file"
commit_message=$(generate_commit_message)
echo "$commit_message" > .commit_message

# Cleanup
cleanup_backups

echo -e "${GREEN}All Streamlit deployment files created and verified!${NC}"
echo -e "${YELLOW}Generated commit message: ${commit_message}${NC}"
