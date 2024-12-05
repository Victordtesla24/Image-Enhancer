"""Test the core image enhancement functionality"""

import logging
from pathlib import Path

import pytest
from PIL import Image

from src.utils.image_processor import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a smaller test size to speed up testing
TEST_IMAGE_PATH = Path(__file__).parent / "data" / "test_image.jpg"


@pytest.fixture(scope="session")
def enhancer():
    """Create an ImageEnhancer instance that's reused across tests"""
    return ImageEnhancer()


def test_enhance_image(enhancer):
    """Test that image enhancement produces expected results"""
    # Load test image
    with Image.open(TEST_IMAGE_PATH) as test_image:
        logger.info("Original image size: %s", test_image.size)
        # Test enhancement with a single target width for speed
        target_width = 512  # Use smaller width for faster testing
        logger.info("Testing enhancement to width: %s", target_width)
        # Enhance image
        enhanced = enhancer.enhance_image(test_image, target_width=target_width)
        # Verify basic properties
        assert enhanced.mode == "RGB", "Enhanced image should be RGB"
        assert enhanced.size[0] == target_width, f"Width should be {target_width}"
        assert enhanced.size[1] > 0, "Height should be positive"
        logger.info("Successfully enhanced to %s", enhanced.size)
