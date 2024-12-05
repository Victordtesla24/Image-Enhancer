"""Test the core image enhancement functionality"""

import logging
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import os
from src.utils.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_image_path(tmp_path_factory):
    """Create a test image and return its path"""
    test_dir = tmp_path_factory.mktemp("test_images")
    image_path = test_dir / "test.png"

    # Create a test image with good dynamic range and sharpness
    size = (500, 500)
    # Create base pattern
    x = np.linspace(0, 50, size[0])
    y = np.linspace(0, 50, size[1])
    xx, yy = np.meshgrid(x, y)
    # Create pattern with high contrast and sharp edges
    pattern = np.sin(xx) * np.cos(yy) * 127 + 128

    # Create RGB image with full dynamic range
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img_array[:, :, 0] = pattern  # Red channel
    img_array[:, :, 1] = np.roll(pattern, 100, axis=0)  # Green channel
    img_array[:, :, 2] = np.roll(pattern, -100, axis=0)  # Blue channel

    # Add some sharp edges
    img_array[200:300, 200:300] = 255  # White square
    img_array[100:200, 100:200] = 0  # Black square

    img = Image.fromarray(img_array, "RGB")
    img.save(image_path, quality=100)

    return str(image_path)


@pytest.fixture(scope="module")
def processor():
    """Create an ImageProcessor instance that's reused across tests"""
    return ImageProcessor()


def test_initialization(processor):
    """Test that ImageProcessor initializes correctly"""
    assert processor.config is not None, "Config should be loaded"
    assert "resolution" in processor.config, "Config should contain resolution settings"
    assert "quality" in processor.config, "Config should contain quality settings"


def test_5k_enhancement(processor, test_image_path, tmp_path):
    """Test that image enhancement to 5K produces expected results"""
    output_path = str(tmp_path / "enhanced_5k.png")

    # Enhance image
    result = processor.enhance_to_5k(test_image_path, output_path)
    assert result is True, "Enhancement should complete successfully"

    # Verify enhanced image exists and has correct properties
    assert os.path.exists(output_path), "Enhanced image should exist"
    enhanced_img = Image.open(output_path)

    # Check resolution
    assert (
        enhanced_img.size[0] == processor.config["resolution"]["width"]
    ), "Width should match 5K"
    assert (
        enhanced_img.size[1] == processor.config["resolution"]["height"]
    ), "Height should match 5K"
    assert enhanced_img.mode == "RGB", "Enhanced image should be RGB"


def test_quality_verification(processor, test_image_path):
    """Test image quality verification functionality"""
    # Verify original test image
    results = processor.verify_5k_quality(test_image_path)

    assert isinstance(results, dict), "Verification should return a dictionary"
    assert "passed" in results, "Results should include pass/fail status"
    assert "metrics" in results, "Results should include metrics"
    assert "failures" in results, "Results should include any failures"

    # Check specific metrics
    metrics = results["metrics"]
    assert "resolution" in metrics, "Should include resolution"
    assert "color_depth" in metrics, "Should include color depth"
    assert "dpi" in metrics, "Should include DPI"
    assert "sharpness" in metrics, "Should include sharpness"
    assert "noise_level" in metrics, "Should include noise level"
    assert "file_size_mb" in metrics, "Should include file size"


def test_enhanced_image_quality(processor, test_image_path, tmp_path):
    """Test that enhanced image meets quality requirements"""
    output_path = str(tmp_path / "quality_test.png")

    # Enhance image
    processor.enhance_to_5k(test_image_path, output_path)

    # Verify enhanced image quality
    results = processor.verify_5k_quality(output_path)

    # Check key quality metrics
    metrics = results["metrics"]
    resolution = metrics["resolution"].split("x")
    assert int(resolution[0]) == processor.config["resolution"]["width"]
    assert int(resolution[1]) == processor.config["resolution"]["height"]
    assert metrics["color_depth"] == "RGB"

    # Verify DPI meets minimum requirement
    dpi = float(metrics["dpi"].split(",")[0])
    assert dpi >= processor.config["quality"]["dpi"]

    # Check if enhanced image passes verification
    assert results[
        "passed"
    ], f"Enhanced image should pass verification. Failures: {results.get('failures', [])}"
