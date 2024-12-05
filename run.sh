#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
REPO_URL="https://github.com/Victordtesla24/Image-Enhancer.git"
BRANCH="main"
REQUIRED_PYTHON_VERSION="3.8"
STREAMLIT_PORT=8501

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    echo -e "${!level}${message}${NC}"
}

# Enhanced cleanup function
cleanup() {
    log_message "YELLOW" "\nCleaning up..."
    deactivate 2>/dev/null
    exit 1
}
trap cleanup INT TERM

# Function to check Python version
check_python_version() {
    local current_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! command -v python >/dev/null 2>&1; then
        log_message "RED" "Python is not installed"
        exit 1
    fi
    
    if [ "$(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$current_version" | sort -V | head -n1)" != "$REQUIRED_PYTHON_VERSION" ]; then
        log_message "RED" "Python version $REQUIRED_PYTHON_VERSION or higher is required. Current version: $current_version"
        exit 1
    fi
}

# Function to check if port is available
check_port_available() {
    if lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null ; then
        log_message "RED" "Port $STREAMLIT_PORT is already in use"
        exit 1
    fi
}

# Function to safely remove directory/file if it exists
safe_remove() {
    if [ -e "$1" ]; then
        rm -rf "$1"
    fi
}

# Function to clear all caches
clear_caches() {
    log_message "YELLOW" "Clearing all caches..."
    
    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    
    # Clear various cache directories
    safe_remove ".pytest_cache"
    safe_remove ".coverage"
    safe_remove "htmlcov"
    safe_remove "~/.streamlit/cache"
    safe_remove ".streamlit/cache"
    safe_remove "~/.cache/image_enhancer"
    safe_remove "~/.cache/torch"
    safe_remove "~/.cache/huggingface"
    safe_remove "~/.cache/chromium"
    safe_remove "~/.cache/google-chrome"
    safe_remove "temp_uploads/*"
    safe_remove ".temp"
    
    # Clear pip cache
    pip cache purge 2>/dev/null
    
    log_message "GREEN" "All caches cleared successfully"
}

# Function to verify dependencies
verify_dependencies() {
    if [ ! -f "requirements.txt" ]; then
        log_message "RED" "requirements.txt not found"
        exit 1
    fi
    
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        log_message "RED" "Failed to install dependencies"
        exit 1
    fi
}

# Function to handle git operations
setup_git() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_message "YELLOW" "Initializing git repository..."
        git init
        if ! git remote add origin $REPO_URL; then
            log_message "RED" "Failed to add git remote"
            exit 1
        fi
    else
        # Verify remote URL
        CURRENT_URL=$(git config --get remote.origin.url)
        if [ "$CURRENT_URL" != "$REPO_URL" ]; then
            log_message "YELLOW" "Updating remote URL..."
            git remote set-url origin $REPO_URL || git remote add origin $REPO_URL
        fi
    fi

    # Ensure on main branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        log_message "YELLOW" "Switching to $BRANCH branch..."
        git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH
    fi
}

# Main execution starts here
log_message "GREEN" "Starting application..."

# Initial checks
check_python_version
check_port_available

# Check required tools
for tool in python git streamlit; do
    if ! command_exists "$tool"; then
        log_message "RED" "Error: $tool is not installed"
        exit 1
    fi
done

# Clear all caches before starting
clear_caches

# Check if verify_and_fix.sh is executable
if [ ! -x "./verify_and_fix.sh" ]; then
    log_message "YELLOW" "Making verify_and_fix.sh executable..."
    chmod +x ./verify_and_fix.sh
fi

# Run verify_and_fix.sh
log_message "YELLOW" "Running verification and fixes..."
./verify_and_fix.sh

# Check and activate virtual environment
if [ ! -d "venv" ]; then
    log_message "RED" "Virtual environment not found"
    exit 1
fi

if ! source venv/bin/activate 2>/dev/null; then
    log_message "RED" "Failed to activate virtual environment"
    exit 1
fi

# Verify dependencies
verify_dependencies

# Run tests with fresh cache
log_message "YELLOW" "Running tests..."
if ! pytest --cache-clear; then
    log_message "RED" "Tests failed. Please fix the issues before running the app."
    exit 1
fi

# Setup git repository
setup_git

# Handle git changes if tests pass
if [ -n "$(git status --porcelain)" ]; then
    log_message "YELLOW" "Changes detected. Committing..."
    git add .
    read -p "Enter commit message: " commit_message
    if ! git commit -m "$commit_message"; then
        log_message "RED" "Failed to commit changes"
        exit 1
    fi
    if ! git push origin $BRANCH; then
        log_message "YELLOW" "Failed to push to remote repository"
    fi
else
    log_message "GREEN" "No changes to commit"
fi

# Clear Streamlit cache before running
log_message "YELLOW" "Clearing Streamlit cache before startup..."
safe_remove "~/.streamlit/cache"

# Run the Streamlit app
log_message "GREEN" "Starting Streamlit app..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
if ! streamlit run streamlit_app.py; then
    log_message "RED" "Failed to start Streamlit app"
    exit 1
fi
