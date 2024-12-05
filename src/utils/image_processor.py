import cv2
import numpy as np
from PIL import Image
import yaml
import os


class ImageProcessor:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        config_path = os.path.join("config", "5k_quality_settings.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def enhance_to_5k(self, image_path, output_path):
        """Enhance image to meet 5K quality standards"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # 1. Initial Noise Reduction
        img = cv2.fastNlMeansDenoisingColored(
            img, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21
        )

        # 2. Resolution Enhancement with quality preservation
        target_width = self.config["resolution"]["width"]
        target_height = self.config["resolution"]["height"]

        # Use multiple steps for better quality upscaling
        current_width, current_height = img.shape[1], img.shape[0]
        scale_factor = min(target_width / current_width, target_height / current_height)

        # Perform incremental upscaling for better quality
        while scale_factor > 2:
            img = cv2.resize(
                img,
                (int(current_width * 2), int(current_height * 2)),
                interpolation=cv2.INTER_CUBIC,
            )
            # Apply sharpening after each upscale
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
            current_width, current_height = img.shape[1], img.shape[0]
            scale_factor /= 2

        # Final resize to target resolution
        img = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )

        # 3. Enhanced Sharpening
        # First pass: Strong sharpening
        kernel = np.array([[-2, -2, -2], [-2, 19, -2], [-2, -2, -2]]) / 3.0
        img = cv2.filter2D(img, -1, kernel)

        # Second pass: Unsharp masking
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        img = cv2.addWeighted(img, 2.5, gaussian, -1.5, 0)

        # 4. Color Enhancement
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        # Enhance contrast of L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels
        img_lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        # 5. Final Detail Enhancement
        # Edge enhancement
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edge_map = cv2.filter2D(img, -1, edge_kernel)
        img = cv2.addWeighted(img, 1.2, edge_map, 0.3, 0)

        # Local contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 6. Final adjustments
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=5)  # Contrast  # Brightness

        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Set DPI to slightly above requirement
        dpi = self.config["quality"]["dpi"] + 1

        # Save with maximum quality
        pil_img.save(output_path, "PNG", dpi=(dpi, dpi), optimize=False)

        return True

    def verify_5k_quality(self, image_path):
        """Verify if image meets 5K quality standards"""
        try:
            img = Image.open(image_path)
            img_cv = cv2.imread(image_path)

            # Convert to numpy array for analysis
            img_array = np.array(img)

            # Initialize results dictionary
            results = {"passed": True, "metrics": {}, "failures": []}

            # 1. Resolution Check
            width, height = img.size
            results["metrics"]["resolution"] = f"{width}x{height}"
            if (
                width < self.config["resolution"]["width"]
                or height < self.config["resolution"]["height"]
            ):
                results["passed"] = False
                results["failures"].append(
                    f"Resolution below 5K standard ({self.config['resolution']['width']}x{self.config['resolution']['height']})"
                )

            # 2. Color Depth Check
            bit_depth = img.mode
            results["metrics"]["color_depth"] = bit_depth
            if bit_depth not in ["RGB", "RGBA"]:
                results["passed"] = False
                results["failures"].append(f"Inadequate color depth: {bit_depth}")

            # 3. DPI Check
            dpi = img.info.get("dpi", (72, 72))
            results["metrics"]["dpi"] = f"{dpi[0]:.2f}, {dpi[1]:.2f}"
            if (
                dpi[0] < self.config["quality"]["dpi"]
                or dpi[1] < self.config["quality"]["dpi"]
            ):
                results["passed"] = False
                results["failures"].append(
                    f"DPI below {self.config['quality']['dpi']}: {dpi}"
                )

            # 4. Dynamic Range Check
            dynamic_range = int(np.max(img_array) - np.min(img_array))
            results["metrics"]["dynamic_range"] = dynamic_range
            if dynamic_range < self.config["color"]["dynamic_range"]["min"]:
                results["passed"] = False
                results["failures"].append(f"Limited dynamic range: {dynamic_range}")

            # 5. Sharpness Check
            if img_cv is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                results["metrics"]["sharpness"] = f"{laplacian_var:.2f}"
                if laplacian_var < self.config["quality"]["min_sharpness"]:
                    results["passed"] = False
                    results["failures"].append(
                        f"Image not sharp enough: {laplacian_var:.2f}"
                    )

            # 6. Noise Analysis
            std_dev = float(np.std(img_array))
            results["metrics"]["noise_level"] = f"{std_dev:.2f}"
            if std_dev > self.config["quality"]["max_noise_level"]:
                results["passed"] = False
                results["failures"].append(f"High noise level: {std_dev:.2f}")

            # 7. File Size Check
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
            results["metrics"]["file_size_mb"] = f"{file_size:.2f}"
            if file_size < self.config["quality"]["min_file_size_mb"]:
                results["passed"] = False
                results["failures"].append(f"File size too small: {file_size:.2f}MB")

            return results

        except Exception as e:
            return {
                "passed": False,
                "metrics": {},
                "failures": [f"Error during verification: {str(e)}"],
            }
