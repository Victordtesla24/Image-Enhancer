from src.utils.image_processor import ImageProcessor
import sys
import json


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python enhance_and_verify.py <input_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.rsplit(".", 1)[0] + "_5k." + input_path.rsplit(".", 1)[1]

    processor = ImageProcessor()

    # First verify original image
    print("\nVerifying Original Image:")
    original_results = processor.verify_5k_quality(input_path)
    print_verification_results(original_results)

    # Enhance image
    print("\nEnhancing image to 5K quality standards...")
    processor.enhance_to_5k(input_path, output_path)

    # Verify enhanced image
    print("\nVerifying Enhanced Image:")
    enhanced_results = processor.verify_5k_quality(output_path)
    print_verification_results(enhanced_results)

    # Save verification results
    results = {"original": original_results, "enhanced": enhanced_results}

    with open("verification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nVerification results have been saved to verification_results.json")


if __name__ == "__main__":
    main()
