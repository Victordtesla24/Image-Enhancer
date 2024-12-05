#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    deactivate 2>/dev/null
    exit 1
}
trap cleanup INT TERM

# Configure git settings
REPO_URL="https://github.com/Victordtesla24/Image-Enhancer.git"
BRANCH="main"

echo -e "${GREEN}Starting application...${NC}"

# Check required tools
for tool in python git streamlit; do
    if ! command_exists "$tool"; then
        echo -e "${RED}Error: $tool is not installed${NC}"
        exit 1
    fi
done

# Check if verify_and_fix.sh is executable
if [ ! -x "./verify_and_fix.sh" ]; then
    echo -e "${YELLOW}Making verify_and_fix.sh executable...${NC}"
    chmod +x ./verify_and_fix.sh
fi

# First, run verify_and_fix.sh
echo -e "${YELLOW}Running verification and fixes...${NC}"
./verify_and_fix.sh

# Check and activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found${NC}"
    exit 1
fi

if ! source venv/bin/activate 2>/dev/null; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
pytest

# Git repository verification and setup
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    git remote add origin $REPO_URL
else
    # Verify remote URL
    CURRENT_URL=$(git config --get remote.origin.url)
    if [ "$CURRENT_URL" != "$REPO_URL" ]; then
        echo -e "${YELLOW}Updating remote URL...${NC}"
        git remote set-url origin $REPO_URL || git remote add origin $REPO_URL
    fi
fi

# Ensure on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${YELLOW}Switching to $BRANCH branch...${NC}"
    git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH
fi

# If tests pass, commit changes
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}Tests passed. Checking for changes...${NC}"
    
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}Changes detected. Committing...${NC}"
        git add .
        read -p "Enter commit message: " commit_message
        git commit -m "$commit_message"
        git push origin $BRANCH || echo -e "${YELLOW}Failed to push to remote repository${NC}"
    else
        echo -e "${GREEN}No changes to commit${NC}"
    fi
    
    # Run the Streamlit app
    echo -e "${GREEN}Starting Streamlit app...${NC}"
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    if ! streamlit run streamlit_app.py; then
        echo -e "${RED}Failed to start Streamlit app${NC}"
        exit 1
    fi
else
    echo -e "${RED}Tests failed. Please fix the issues before running the app.${NC}"
    exit 1
fi
