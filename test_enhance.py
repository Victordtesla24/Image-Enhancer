from PIL import Image
from src.utils.image_processor import ImageEnhancer
import torch


def main():
    # Initialize the enhancer
    enhancer = ImageEnhancer()

    # Load and prepare test image
    input_image = Image.open("test_image.png")

    # First enhance with detail preservation
    print("Enhancing image...")
    enhanced_image, details = enhancer.enhance_image(
        input_image,
        target_width=1920,  # Full HD resolution for optimal quality
        models=["detail", "superres"],  # Focus on detail first, then resolution
    )

    # Then enhance color while preserving contrast
    enhanced_image, color_details = enhancer.enhance_image(
        enhanced_image,
        target_width=1920,
        models=["color"],  # Separate color enhancement pass
    )

    # Save with maximum quality
    enhanced_image.save("enhanced_output.png", quality=100, optimize=False)

    print("\nEnhancement details:")
    print(f"Source size: {details['source_size']}")
    print(f"Final size: {enhanced_image.size[0]}x{enhanced_image.size[1]}")
    print(
        f"Total processing time: {float(details['processing_time'].rstrip('s')) + float(color_details['processing_time'].rstrip('s')):.2f}s"
    )
    print("\nModels used:")
    for model in details["models_used"] + color_details["models_used"]:
        print(f"- {model['name']}: {model['description']}")


if __name__ == "__main__":
    main()
