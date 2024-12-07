#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    source venv/bin/activate
fi

# Run verification script
echo "🔍 Running verification checks..."
./verify_and_fix.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}Verification failed. Please fix the issues and try again.${NC}"
    exit 1
fi

# Check GPU availability and memory
echo "🔧 Checking GPU resources..."
if command -v nvidia-smi &> /dev/null; then
    gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    if [ "$gpu_memory" -lt 8000 ]; then
        echo -e "${YELLOW}Warning: Less than 8GB GPU memory available. Performance might be affected.${NC}"
    fi
    nvidia-smi
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU found. Running in CPU mode.${NC}"
fi

# Parse command line arguments
POSITIONAL_ARGS=()
QUALITY=0.95
RESOLUTION="5k"
BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift
      shift
      ;;
    --quality)
      QUALITY="$2"
      shift
      shift
      ;;
    --resolution)
      RESOLUTION="$2"
      shift
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate input arguments
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo -e "${RED}Error: Input and output paths are required.${NC}"
    echo "Usage: ./run.sh --input <path> --output <path> [--quality <0-1>] [--resolution <4k|5k|8k>] [--batch-size <n>]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Start monitoring
echo "📊 Starting system monitoring..."
top -b -n 1 > system_stats_start.txt

# Run the enhancement process
echo "🚀 Starting image enhancement..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Target Quality: $QUALITY"
echo "Target Resolution: $RESOLUTION"
echo "Batch Size: $BATCH_SIZE"

PYTHONPATH=. python main.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --quality "$QUALITY" \
    --resolution "$RESOLUTION" \
    --batch-size "$BATCH_SIZE"

exit_code=$?

# Capture final system stats
top -b -n 1 > system_stats_end.txt

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✨ Enhancement completed successfully!${NC}"
    
    # Compare input and output
    if command -v identify &> /dev/null; then
        echo "📊 Image comparison:"
        echo "Input:"
        identify "$INPUT"
        echo "Output:"
        identify "$OUTPUT"
    fi
else
    echo -e "${RED}❌ Enhancement failed with exit code $exit_code${NC}"
fi

# Cleanup
rm -f system_stats_start.txt system_stats_end.txt

exit $exit_code 