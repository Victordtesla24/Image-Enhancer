#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check environment
check_environment() {
    echo -e "${YELLOW}Checking environment...${NC}"
    
    # Run verify_and_fix.sh if it exists
    if [ -f "verify_and_fix.sh" ]; then
        bash verify_and_fix.sh
    else
        echo -e "${RED}verify_and_fix.sh not found!${NC}"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run pytest with coverage
    pytest --cov=src tests/
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Tests failed! Please fix the issues before running the app.${NC}"
        exit 1
    fi
}

# Function to commit changes to Git
commit_changes() {
    echo -e "${YELLOW}Committing changes to Git...${NC}"
    
    # Check if there are any changes
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        read -p "Enter commit message: " commit_message
        git commit -m "$commit_message"
        git push origin main || echo -e "${RED}Failed to push to remote repository${NC}"
    else
        echo -e "${GREEN}No changes to commit${NC}"
    fi
}

# Function to run the Streamlit app
run_app() {
    echo -e "${GREEN}Starting Streamlit app...${NC}"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run the app
    streamlit run src/app.py
}

# Main execution
main() {
    echo -e "${GREEN}Starting application...${NC}"
    
    check_environment
    run_tests
    commit_changes
    run_app
}

main 