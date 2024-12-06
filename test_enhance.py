"""Test script for image enhancement"""

import logging
from PIL import Image
from src.utils.image_processor import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhancement():
    """Test image enhancement functionality"""
    try:
        # Initialize enhancer
        enhancer = ImageEnhancer()

        # Load test image
        input_image = Image.open("small_test.png")
        logger.info(f"Loaded test image: {input_image.size}")

        # Define enhancement models to use
        models = ["Super Resolution", "Color Enhancement", "Detail Enhancement"]

        # Process image
        logger.info("Starting enhancement process...")
        enhanced_image, details = enhancer.enhance_image(
            input_image, target_width=5120, models=models  # 5K width
        )

        # Save result
        enhanced_image.save("enhanced_test.png")
        logger.info(f"Enhancement completed. Details: {details}")

        return True, details

    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}")
        return False, str(e)


if __name__ == "__main__":
    success, result = test_enhancement()
    if success:
        print("Enhancement successful!")
        print("Enhancement details:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(f"Enhancement failed: {result}")
