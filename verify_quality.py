from PIL import Image
import numpy as np


def analyze_image_quality(image_path):
    """Analyze image quality metrics"""
    img = Image.open(image_path)
    print(f"\nAnalyzing {image_path}:")
    print(f"Resolution: {img.size}")
    print(f"Mode: {img.mode}")
    print(f"Format: {img.format}")

    # Convert to numpy array for analysis
    img_array = np.array(img)

    # Calculate basic statistics
    print(f"Mean pixel value: {np.mean(img_array):.2f}")
    print(f"Std deviation: {np.std(img_array):.2f}")
    print(f"Dynamic range: {np.min(img_array)} - {np.max(img_array)}")

    return img_array


# Analyze both original and enhanced images
print("Original Image Analysis:")
original = analyze_image_quality("test_image.png")

print("\nEnhanced Image Analysis:")
enhanced = analyze_image_quality("enhanced_output.png")

# Calculate improvement metrics
if original.shape == enhanced.shape:
    mse = np.mean((original - enhanced) ** 2)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        print(f"\nImprovement Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Peak Signal-to-Noise Ratio: {psnr:.2f} dB")
