from setuptools import setup, find_packages

setup(
    name="image-enhancer",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.7.0",
        "pillow>=9.0.0",
        "scikit-image>=0.19.0",
        "tensorflow>=2.12.0",
        "onnxruntime-gpu>=1.14.0",
        "openvino>=2023.0.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "image-enhancer=src.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced image enhancement system with AI-powered features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-enhancer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Framework :: Streamlit",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
) 