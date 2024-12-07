"""Create a test image for UI testing"""

import os

import numpy as np
from PIL import Image, ImageDraw


def create_test_image():
    """Create a test image with standard dimensions and format"""
    # Create a 640x480 RGB image
    width, height = 640, 480
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Draw some shapes to make it interesting
    # Rectangle
    draw.rectangle([100, 100, 300, 200], fill="blue")
    # Circle
    draw.ellipse([350, 250, 450, 350], fill="red")
    # Line
    draw.line([50, 400, 590, 400], fill="green", width=5)

    # Add some text
    draw.text((width // 2 - 50, 50), "Test Image", fill="black")

    # Save with high quality JPEG settings
    output_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    image.save(output_path, "JPEG", quality=95, optimize=True)
    print(f"Created test image at: {output_path}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    print(f"Image format: JPEG")


if __name__ == "__main__":
    create_test_image()
