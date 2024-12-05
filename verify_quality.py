from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
import cv2
import os


def verify_5k_requirements(image_path):
    """Verify if image meets 5K quality requirements"""
    try:
        img = Image.open(image_path)
        img_cv = cv2.imread(image_path)

        # Convert to numpy array for analysis
        img_array = np.array(img)

        # Initialize results dictionary
        results = {"passed": True, "metrics": {}, "failures": []}

        # 1. Resolution Check (5K = 5120 x 2880)
        width, height = img.size
        results["metrics"]["resolution"] = f"{width}x{height}"
        if width < 5120 or height < 2880:
            results["passed"] = False
            results["failures"].append("Resolution below 5K standard (5120x2880)")

        # 2. Color Depth Check
        bit_depth = img.mode
        results["metrics"]["color_depth"] = bit_depth
        if bit_depth not in ["RGB", "RGBA"]:
            results["passed"] = False
            results["failures"].append(f"Inadequate color depth: {bit_depth}")

        # 3. DPI Check
        try:
            dpi = img.info.get("dpi", (72, 72))
            results["metrics"]["dpi"] = dpi
            if dpi[0] < 300 or dpi[1] < 300:
                results["passed"] = False
                results["failures"].append(f"DPI below 300: {dpi}")
        except:
            results["metrics"]["dpi"] = "Not available"

        # 4. Dynamic Range Check
        dynamic_range = np.max(img_array) - np.min(img_array)
        results["metrics"]["dynamic_range"] = dynamic_range
        if dynamic_range < 200:  # Assuming 8-bit color depth
            results["passed"] = False
            results["failures"].append(f"Limited dynamic range: {dynamic_range}")

        # 5. Sharpness Check
        if img_cv is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            results["metrics"]["sharpness"] = laplacian_var
            if laplacian_var < 100:  # Threshold for acceptable sharpness
                results["passed"] = False
                results["failures"].append(f"Image not sharp enough: {laplacian_var}")

        # 6. Noise Analysis
        std_dev = np.std(img_array)
        results["metrics"]["noise_level"] = std_dev
        if std_dev > 50:  # Threshold for acceptable noise
            results["passed"] = False
            results["failures"].append(f"High noise level: {std_dev}")

        # 7. File Size Check (for compression quality)
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
        results["metrics"]["file_size_mb"] = file_size
        if file_size < 2:  # Minimum 2MB for high-quality 5K
            results["passed"] = False
            results["failures"].append(f"File size too small: {file_size:.2f}MB")

        return results

    except Exception as e:
        return {
            "passed": False,
            "metrics": {},
            "failures": [f"Error during verification: {str(e)}"],
        }


def print_verification_results(results):
    """Print verification results in a formatted way"""
    print("\n=== 5K Image Quality Verification Results ===")
    print("\nMetrics:")
    for metric, value in results["metrics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value}")

    print("\nVerification Status:", "PASSED" if results["passed"] else "FAILED")

    if not results["passed"]:
        print("\nFailures:")
        for failure in results["failures"]:
            print(f"- {failure}")


if __name__ == "__main__":
    # Verify both original and enhanced images
    print("\nVerifying Original Image:")
    original_results = verify_5k_requirements("test_image.png")
    print_verification_results(original_results)

    print("\nVerifying Enhanced Image:")
    enhanced_results = verify_5k_requirements("enhanced_output.png")
    print_verification_results(enhanced_results)
