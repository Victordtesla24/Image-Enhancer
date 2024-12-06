"""Test the core image enhancement functionality"""

import logging
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import os
from src.utils.image_processor import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_image_path(tmp_path_factory):
    """Create a test image and return its path"""
    test_dir = tmp_path_factory.mktemp("test_images")
    image_path = test_dir / "test.png"

    # Create a test image with good dynamic range and sharpness
    size = (1024, 1024)  # Larger size for better quality

    # Create base pattern with higher frequency for better sharpness
    x = np.linspace(0, 100, size[0])
    y = np.linspace(0, 100, size[1])
    xx, yy = np.meshgrid(x, y)

    # Create multiple patterns for better dynamic range
    pattern1 = np.sin(xx * 0.5) * np.cos(yy * 0.5) * 127 + 128
    pattern2 = np.sin(xx * 0.2) * np.cos(yy * 0.2) * 127 + 128
    pattern3 = np.sin(xx * 0.1) * np.cos(yy * 0.1) * 127 + 128

    # Combine patterns for more detail
    pattern = pattern1 * 0.5 + pattern2 * 0.3 + pattern3 * 0.2

    # Ensure full dynamic range
    pattern = (
        (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255
    ).astype(np.uint8)

    # Create RGB image with full dynamic range
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Add varied patterns to each channel for better color range
    img_array[:, :, 0] = pattern  # Red channel
    img_array[:, :, 1] = np.roll(pattern, size[0] // 4, axis=0)  # Green channel
    img_array[:, :, 2] = np.roll(pattern, -size[0] // 4, axis=0)  # Blue channel

    # Add high contrast elements
    img_array[size[0] // 4 : size[0] // 2, size[1] // 4 : size[1] // 2] = (
        255  # White square
    )
    img_array[size[0] // 2 : 3 * size[0] // 4, size[1] // 2 : 3 * size[1] // 4] = (
        0  # Black square
    )

    # Add diagonal lines for sharpness
    for i in range(size[0]):
        img_array[i, i] = 255
        img_array[i, size[1] - i - 1] = 255

    # Create image and save with high quality
    img = Image.fromarray(img_array, "RGB")

    # Save with high quality settings
    img.save(image_path, format="PNG", optimize=False, quality=100, dpi=(300, 300))

    return str(image_path)


@pytest.fixture(scope="module")
def processor():
    """Create an ImageEnhancer instance that's reused across tests"""
    return ImageEnhancer()


def test_initialization(processor):
    """Test that ImageEnhancer initializes correctly"""
    assert processor.config is not None, "Config should be loaded"
    assert "resolution" in processor.config, "Config should contain resolution settings"
    assert "quality" in processor.config, "Config should contain quality settings"


def test_5k_enhancement(processor, test_image_path, tmp_path):
    """Test that image enhancement to 5K produces expected results"""
    output_path = str(tmp_path / "enhanced_5k.png")

    # Enhance image
    enhanced_img, _ = processor.enhance_image(
        Image.open(test_image_path),
        target_width=processor.config["resolution"]["width"],
        models=["Super Resolution"],
    )

    # Save with high quality settings
    enhanced_img.save(
        output_path, format="PNG", optimize=False, quality=100, dpi=(300, 300)
    )

    # Verify enhanced image exists and has correct properties
    assert os.path.exists(output_path), "Enhanced image should exist"
    enhanced_img = Image.open(output_path)

    # Check resolution
    assert (
        enhanced_img.size[0] == processor.config["resolution"]["width"]
    ), "Width should match 5K"
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
    enhanced_img, _ = processor.enhance_image(
        Image.open(test_image_path),
        target_width=processor.config["resolution"]["width"],
        models=["Super Resolution", "Color Enhancement", "Detail Enhancement"],
    )

    # Save with high quality settings
    enhanced_img.save(
        output_path, format="PNG", optimize=False, quality=100, dpi=(300, 300)
    )

    # Verify enhanced image quality
    results = processor.verify_5k_quality(output_path)

    # Check key quality metrics
    metrics = results["metrics"]
    resolution = metrics["resolution"].split("x")
    assert int(resolution[0]) == processor.config["resolution"]["width"]
    assert metrics["color_depth"] == "RGB"

    # Verify DPI meets minimum requirement
    dpi = float(metrics["dpi"].split(",")[0])
    assert dpi >= processor.config["quality"]["dpi"]

    # Check if enhanced image passes verification
    assert results[
        "passed"
    ], f"Enhanced image should pass verification. Failures: {results.get('failures', [])}"
