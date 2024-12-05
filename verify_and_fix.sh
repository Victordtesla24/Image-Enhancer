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
pip install --upgrade pip

# Ensure PyTorch and super-image are installed correctly
if ! pip show torch > /dev/null || ! pip show super-image > /dev/null; then
    echo -e "${YELLOW}Installing PyTorch and super-image...${NC}"
    pip install torch torchvision --find-links https://download.pytorch.org/whl/torch_stable.html
    pip uninstall -y huggingface-hub super-image
    pip install huggingface-hub==0.8.1
    pip install super-image==0.1.7
fi

pip install -r requirements.txt

# Fix code formatting
echo -e "${YELLOW}Fixing code formatting...${NC}"
if command_exists black; then
    black src/ tests/
else
    pip install black
    black src/ tests/
fi

# Run linting
echo -e "${YELLOW}Running linting...${NC}"
if command_exists pylint; then
    pylint src/ tests/ || true
else
    pip install pylint
    pylint src/ tests/ || true
fi

# Verify directory structure
echo -e "${YELLOW}Verifying directory structure...${NC}"
directories=("src" "src/components" "src/utils" "tests" "config" "assets")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created missing directory: $dir${NC}"
    fi
done

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

# Update requirements.txt
echo -e "${YELLOW}Updating requirements.txt...${NC}"
pip freeze > requirements.txt

# Verify Streamlit configuration
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

echo -e "${GREEN}Verification and fixes completed successfully!${NC}" 