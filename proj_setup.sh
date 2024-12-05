#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting project setup...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to securely store credentials
store_credentials() {
    local cred_dir="$HOME/.streamlit_credentials"
    mkdir -p "$cred_dir"
    chmod 700 "$cred_dir"
    
    # Get and encrypt credentials
    read -p "Enter GitHub username: " github_user
    read -sp "Enter GitHub token: " github_token
    echo
    read -p "Enter GCloud project ID: " gcloud_project
    read -sp "Enter GCloud service account key path: " gcloud_key
    echo

    # Store encrypted credentials
    echo "$github_user" | openssl enc -aes-256-cbc -salt -out "$cred_dir/github_user.enc"
    echo "$github_token" | openssl enc -aes-256-cbc -salt -out "$cred_dir/github_token.enc"
    echo "$gcloud_project" | openssl enc -aes-256-cbc -salt -out "$cred_dir/gcloud_project.enc"
    echo "$gcloud_key" | openssl enc -aes-256-cbc -salt -out "$cred_dir/gcloud_key.enc"
}

# Check and install Python dependencies
setup_python() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    
    if ! command_exists python3; then
        echo -e "${RED}Python3 not found. Installing...${NC}"
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    fi

    # Create and activate virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install required packages
    pip install --upgrade pip
    pip install streamlit
    pip install pytest pytest-cov black isort pylint
    pip install pillow opencv-python-headless numpy
    
    # Create requirements.txt
    pip freeze > requirements.txt
}

# Setup project structure
setup_project_structure() {
    echo -e "${YELLOW}Creating project structure...${NC}"
    
    # Create directory structure
    mkdir -p src/components
    mkdir -p src/utils
    mkdir -p tests/unit
    mkdir -p tests/integration
    mkdir -p data
    mkdir -p models
    mkdir -p docs

    # Create basic files
    touch src/__init__.py
    touch src/components/__init__.py
    touch src/utils/__init__.py
    touch tests/__init__.py
    
    # Create main app file
    cat > src/app.py << EOL
import streamlit as st

def main():
    st.title("Image Enhancement App")
    st.write("Upload an image to enhance it!")

if __name__ == "__main__":
    main()
EOL

    # Create test configuration
    cat > pytest.ini << EOL
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --verbose --cov=src
EOL
}

# Setup Git repository
setup_git() {
    echo -e "${YELLOW}Setting up Git repository...${NC}"
    
    if ! command_exists git; then
        echo -e "${RED}Git not found. Installing...${NC}"
        sudo apt-get install -y git
    fi

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

    git add .
    git commit -m "Initial commit"
}

# Setup Google Cloud
setup_gcloud() {
    echo -e "${YELLOW}Setting up Google Cloud...${NC}"
    
    if ! command_exists gcloud; then
        echo -e "${RED}Google Cloud SDK not found. Installing...${NC}"
        curl https://sdk.cloud.google.com | bash
        exec -l $SHELL
    fi
}

# Main execution
main() {
    setup_python
    setup_project_structure
    setup_git
    setup_gcloud
    store_credentials
    
    echo -e "${GREEN}Project setup completed successfully!${NC}"
    echo -e "${YELLOW}Please run verify_and_fix.sh to ensure everything is properly configured.${NC}"
}

main 