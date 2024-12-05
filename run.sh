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

# Configure git branch
BRANCH=${BRANCH:-main}

echo -e "${GREEN}Starting application...${NC}"

# Check required tools
for tool in python git pytest uvicorn; do
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

# If tests pass, commit changes
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}Tests passed. Committing changes...${NC}"
    
    # Check if it's a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${YELLOW}Not a git repository - skipping git operations${NC}"
    else
        git add .
        read -p "Enter commit message: " commit_message
        git commit -m "$commit_message"
        git push origin $BRANCH || echo -e "${YELLOW}No remote repository configured${NC}"
    fi
    
    # Run the FastAPI app with uvicorn
    echo -e "${GREEN}Starting FastAPI app...${NC}"
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    if ! uvicorn src.app:app --reload --host 0.0.0.0 --port 8000; then
        echo -e "${RED}Failed to start FastAPI app${NC}"
        exit 1
    fi
else
    echo -e "${RED}Tests failed. Please fix the issues before running the app.${NC}"
    exit 1
fi
