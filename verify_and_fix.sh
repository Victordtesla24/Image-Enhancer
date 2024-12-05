#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting verification and fixes...${NC}"

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

# Create README.md if it doesn't exist
if [ ! -f "README.md" ]; then
    echo -e "${YELLOW}Creating README.md...${NC}"
    cat > README.md << EOL
# Image Enhancer

A Streamlit application for enhancing image quality using deep learning.

## Features
- Upload images
- Enhance image quality
- Download enhanced images

## Setup
1. Clone the repository
2. Run \`./proj_setup.sh\` to set up the virtual environment
3. Run \`./run.sh\` to start the application

## Requirements
- Python 3.7+
- See requirements.txt for full dependencies
EOL
fi

# Move app.py to streamlit_app.py if needed
if [ -f "src/app.py" ] && [ ! -f "streamlit_app.py" ]; then
    echo -e "${YELLOW}Moving app.py to streamlit_app.py...${NC}"
    cp src/app.py streamlit_app.py
    # Update imports in streamlit_app.py
    sed -i '' 's/from src\./from /g' streamlit_app.py
fi

# Remove FastAPI specific directories
if [ -d "src/static" ]; then
    echo -e "${YELLOW}Removing FastAPI specific directories...${NC}"
    rm -rf src/static
fi

# Verify init files
init_files=("src/__init__.py" "src/components/__init__.py" "src/utils/__init__.py" "tests/__init__.py")
for file in "${init_files[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo -e "${GREEN}Created missing file: $file${NC}"
    fi
done

# Remove redundant files
echo -e "${YELLOW}Cleaning up redundant files...${NC}"
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -r {} +

# Update requirements.txt with fixed versions
echo -e "${YELLOW}Updating requirements.txt...${NC}"
pip freeze > requirements.txt

# Create .streamlit directory and config
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

# Final cache clear
clear_dev_caches

echo -e "${GREEN}Verification and fixes completed successfully!${NC}"
