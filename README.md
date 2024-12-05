# Image-Enhancer

A professional image enhancement system that converts images to 5K quality while maintaining optimal visual characteristics.

## Features

- 5K resolution enhancement (5120x2880)
- Advanced sharpness enhancement
- Noise reduction
- Color optimization
- Quality verification system
- Comprehensive metrics analysis

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- PIL (Pillow)
- PyYAML
- Pytest (for testing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Enhancer.git
cd Image-Enhancer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Enhance a single image:
```bash
python enhance_and_verify.py input_image.png
```

The enhanced image will be saved with '_5k' suffix (e.g., input_image_5k.png)

### Quality Metrics

The system verifies the following metrics:
- Resolution: 5120x2880 (5K)
- Color Depth: RGB
- DPI: 300+
- Dynamic Range: 220-255
- Sharpness: 70+
- Noise Level: Below 120
- File Size: Minimum 1.5MB

## Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src/
```

## Configuration

Quality settings can be adjusted in `config/5k_quality_settings.yaml`

## Project Structure

```
Image-Enhancer/
├── config/
│   └── 5k_quality_settings.yaml
├── src/
│   ├── utils/
│   │   ├── image_processor.py
│   │   └── models/
│   └── components/
├── tests/
│   └── test_image_processor.py
├── enhance_and_verify.py
└── requirements.txt
```

## Deployment

1. Ensure all tests pass
2. Verify configuration settings
3. Check system requirements
4. Deploy using your preferred method (Docker, direct installation, etc.)

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
