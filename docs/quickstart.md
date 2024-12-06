# 5K AI Image Enhancer - Quick Start Guide

## Overview
The 5K AI Image Enhancer is a powerful tool that enhances images to 5K quality using advanced image processing techniques. This guide will help you get started quickly.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Victordtesla24/Image-Enhancer.git
cd Image-Enhancer
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Application

1. **Start the application**
```bash
streamlit run streamlit_app.py
```

2. **Access the web interface**
- Open your browser and go to `http://localhost:8501`
- The application will automatically open in your default browser

## Using the Enhancer

### 1. Enhancement Models
The application offers three powerful enhancement models:

- **Super Resolution**
  - Increases image resolution while preserving details
  - Uses advanced multi-step processing with Lanczos resampling
  - Best for upscaling images to higher resolutions

- **Color Enhancement**
  - Optimizes color balance and vibrancy
  - Uses LAB color space processing
  - Perfect for improving color accuracy and contrast

- **Detail Enhancement**
  - Sharpens and enhances image details
  - Uses multi-scale enhancement with noise reduction
  - Ideal for bringing out fine details

### 2. Resolution Settings

Available presets:
- 5K (5120x2880) - Default
- 4K (3840x2160)
- 2K (2048x1080)
- Full HD (1920x1080)
- Custom (user-defined)

### 3. Step-by-Step Enhancement

1. **Select Enhancement Models**
   - Choose one or more models from the sidebar
   - All models are selected by default for optimal results

2. **Choose Resolution**
   - Select from presets or specify custom dimensions
   - Higher resolutions require more processing time

3. **Upload Image**
   - Click "Browse files" or drag and drop
   - Supports PNG, JPG, JPEG formats
   - Maximum file size: 200MB

4. **Process Image**
   - Click "Enhance Image"
   - Monitor progress in real-time
   - View side-by-side comparison

5. **Download Result**
   - Click "Download Enhanced Image"
   - Enhanced image saves as PNG format

## Tips for Best Results

1. **Image Quality**
   - Start with the highest quality source image available
   - Larger source images generally produce better results

2. **Model Selection**
   - Use Super Resolution for upscaling
   - Add Color Enhancement for vibrant results
   - Use Detail Enhancement for sharper images

3. **Resolution Choice**
   - Choose resolution based on intended use
   - Higher resolutions require more processing time
   - Consider source image size when selecting target resolution

## Troubleshooting

### Common Issues

1. **Upload Errors**
   - Ensure file format is supported (PNG, JPG, JPEG)
   - Check file size is under 200MB
   - Verify image isn't corrupted

2. **Processing Issues**
   - Ensure enough system memory is available
   - Try processing with fewer models if system is slow
   - Check internet connection is stable

3. **Quality Issues**
   - Start with higher quality source images
   - Try different combinations of enhancement models
   - Adjust resolution settings as needed

### Getting Help

- Check the full documentation in `docs/architecture.md`
- Review error messages for specific issues
- Ensure all dependencies are properly installed

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Modern web browser
- Internet connection for initial setup

## Additional Resources

- Full architecture documentation: `docs/architecture.md`
- Source code: `src/` directory
- Configuration files: `config/` directory
- Test images: `tests/data/` directory

This quick start guide provides essential information to get you up and running with the 5K AI Image Enhancer. For more detailed information, please refer to the architecture documentation.
