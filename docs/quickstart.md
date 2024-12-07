# Quick Start Guide

## Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (optional but recommended)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-enhancer.git
cd image-enhancer
```

2. Run the setup script:
```bash
bash scripts/project_setup.sh
```

3. Create and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Starting the Application
```bash
bash scripts/run.sh
```

### Basic Operations
1. Upload an image through the web interface
2. Select enhancement options:
   - Super Resolution
   - Detail Enhancement
   - Color Enhancement
3. Adjust quality parameters
4. Process the image
5. Download the enhanced result

### Batch Processing
```python
from src.utils.image_processor import ImageProcessor

processor = ImageProcessor()
results = processor.process_batch(['image1.jpg', 'image2.jpg'])
```

### API Usage
```python
from src.utils.core.processor import Processor

# Initialize processor
processor = Processor()

# Process single image
result = processor.process_image('input.jpg')

# Save result
result.save('output.jpg')
```

## Configuration

### Quality Settings
```python
config = {
    "enhancement": {
        "super_resolution": {"enabled": True, "strength": 0.7},
        "detail": {"enabled": True, "strength": 0.8},
        "color": {"enabled": True, "strength": 0.6}
    },
    "quality": {
        "target_metrics": {
            "sharpness": 0.8,
            "color_accuracy": 0.8,
            "detail_preservation": 0.8,
            "noise_level": 0.2
        }
    }
}
```

### Performance Tuning
```python
# GPU Settings
processor.use_gpu = True
processor.batch_size = 4

# Cache Settings
processor.enable_cache = True
processor.cache_dir = 'cache/models'
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_processor.py

# Run with coverage
pytest --cov=src
```

### Code Style
```bash
# Format code
black .
isort .

# Check style
flake8 src tests
```

## Troubleshooting

### Common Issues
1. GPU not detected
   - Check CUDA installation
   - Update GPU drivers

2. Memory errors
   - Reduce batch size
   - Enable memory optimization

3. Quality issues
   - Adjust enhancement parameters
   - Check input image quality

### Getting Help
- Check documentation in `docs/`
- Submit issues on GitHub
- Contact support team

## Next Steps
1. Read the full documentation
2. Try example notebooks
3. Explore advanced features
4. Join the community
