"""Test the core image enhancement functionality"""

import logging
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
from src.utils.image_processor import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a small test image in memory instead of loading from disk
def create_test_image():
    """Create a small RGB test image"""
    # Create a 100x100 RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array, "RGB")


@pytest.fixture(scope="module")
def enhancer():
    """Create an ImageEnhancer instance that's reused across tests"""
    return ImageEnhancer()


def test_enhance_image(enhancer):
    """Test that image enhancement produces expected results"""
    # Create a small test image
    test_image = create_test_image()
    logger.info("Original image size: %s", test_image.size)

    # Test enhancement with a very small target width for speed
    target_width = 128  # Use tiny width for faster testing
    logger.info("Testing enhancement to width: %s", target_width)

    # Enhance image
    enhanced_image, enhancement_details = enhancer.enhance_image(
        test_image, target_width=target_width
    )

    # Verify basic properties
    assert enhanced_image.mode == "RGB", "Enhanced image should be RGB"
    assert enhanced_image.size[0] == target_width, f"Width should be {target_width}"
    assert enhanced_image.size[1] > 0, "Height should be positive"
    logger.info("Successfully enhanced to %s", enhanced_image.size)

    # Verify enhancement details
    assert isinstance(
        enhancement_details, dict
    ), "Enhancement details should be a dictionary"
    assert (
        "source_size" in enhancement_details
    ), "Enhancement details should include source size"
    assert (
        "target_size" in enhancement_details
    ), "Enhancement details should include target size"
    assert (
        "processing_time" in enhancement_details
    ), "Enhancement details should include processing time"
