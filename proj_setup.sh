#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting project setup...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create directory if it doesn't exist
create_dir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "${GREEN}Created directory: $1${NC}"
    fi
}

# Check Python version
if ! command_exists python3; then
    echo -e "${RED}Python3 is not installed. Please install Python3 first.${NC}"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
else
    source venv/bin/activate
fi

# Create project structure
create_dir_if_not_exists "src"
create_dir_if_not_exists "src/components"
create_dir_if_not_exists "src/utils"
create_dir_if_not_exists "tests"
create_dir_if_not_exists "config"
create_dir_if_not_exists "assets"

# Create necessary files if they don't exist
touch src/__init__.py
touch src/components/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install --upgrade pip
pip install streamlit pillow opencv-python-headless pytest pytest-cov black pylint

# Generate requirements.txt
pip freeze > requirements.txt

# Setup Git if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
    
    # Create .gitignore
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

    # Prompt for Git credentials
    echo -e "${YELLOW}Please enter your Git credentials:${NC}"
    read -p "Username: " git_username
    read -p "Email: " git_email
    
    git config user.name "$git_username"
    git config user.email "$git_email"
fi

# Setup Google Cloud credentials
echo -e "${YELLOW}Setting up Google Cloud credentials...${NC}"
if ! command_exists gcloud; then
    echo -e "${RED}Google Cloud SDK not found. Please install it first.${NC}"
else
    read -p "Enter your Google Cloud project ID: " project_id
    gcloud config set project "$project_id"
    
    # Create .env file for API keys
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        touch .env
        echo "GOOGLE_CLOUD_PROJECT=$project_id" >> .env
    fi
fi

# Create pytest.ini
cat > pytest.ini << EOL
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --verbose --cov=src
EOL

# Create initial Streamlit app
if [ ! -f "src/app.py" ]; then
    cat > src/app.py << EOL
import streamlit as st

def main():
    st.title("Image Enhancement App")
    st.write("Upload an image to enhance it!")

if __name__ == "__main__":
    main()
EOL
fi

echo -e "${GREEN}Project setup completed successfully!${NC}" 