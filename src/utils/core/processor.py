"""Core image processor module"""

import logging
import time
import numpy as np
from PIL import Image
import yaml
import os
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoreProcessor:
    """Core image processing functionality"""

    def __init__(self):
        """Initialize core processor"""
        self.config = self._load_config()
        logger.info("Core processor initialized")

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = os.path.join("config", "5k_quality_settings.yaml")
            logger.info(f"Loading config from: {config_path}")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def pil_to_numpy(self, image):
        """Convert PIL Image to numpy array"""
        return np.array(image)

    def numpy_to_pil(self, array, dpi=(300, 300)):
        """Convert numpy array to PIL Image with specified DPI"""
        img = Image.fromarray(array)
        img.info["dpi"] = dpi
        return img

    def verify_image_quality(self, image):
        """Verify if image meets quality standards"""
        try:
            quality_checks = {}
            passed = True

            # 1. Resolution Check
            width, height = image.size
            quality_checks["resolution"] = {
                "value": f"{width}x{height}",
                "required": f"{self.config['resolution']['width']}x{self.config['resolution']['height']}",
                "passed": width >= self.config["resolution"]["width"]
                and height >= self.config["resolution"]["height"],
            }
            passed = passed and quality_checks["resolution"]["passed"]

            # 2. Color Depth Check
            bit_depth = "32" if image.mode == "RGBA" else "24"
            quality_checks["color_depth"] = {
                "value": f"{bit_depth}-bit {image.mode}",
                "required": f"{self.config['color']['bit_depth']}-bit {self.config['color']['color_space']}",
                "passed": image.mode in ["RGB", "RGBA"],
            }
            passed = passed and quality_checks["color_depth"]["passed"]

            # 3. DPI Check
            dpi = image.info.get("dpi", (72, 72))
            quality_checks["dpi"] = {
                "value": f"{dpi[0]:.2f}, {dpi[1]:.2f}",
                "required": str(self.config["quality"]["dpi"]),
                "passed": dpi[0] >= self.config["quality"]["dpi"]
                and dpi[1] >= self.config["quality"]["dpi"],
            }
            passed = passed and quality_checks["dpi"]["passed"]

            # 4. Dynamic Range Check
            img_array = np.array(image)
            dynamic_range = int(np.max(img_array) - np.min(img_array))
            quality_checks["dynamic_range"] = {
                "value": str(dynamic_range),
                "required": f"{self.config['color']['dynamic_range']['min']}-{self.config['color']['dynamic_range']['max']}",
                "passed": dynamic_range >= self.config["color"]["dynamic_range"]["min"],
            }
            passed = passed and quality_checks["dynamic_range"]["passed"]

            # 5. Sharpness Check
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_checks["sharpness"] = {
                "value": f"{laplacian_var:.2f}",
                "required": str(self.config["quality"]["min_sharpness"]),
                "passed": laplacian_var >= self.config["quality"]["min_sharpness"],
            }
            passed = passed and quality_checks["sharpness"]["passed"]

            # 6. Noise Level Check
            std_dev = float(np.std(img_array))
            quality_checks["noise_level"] = {
                "value": f"{std_dev:.2f}",
                "required": f"<= {self.config['quality']['max_noise_level']}",
                "passed": std_dev <= self.config["quality"]["max_noise_level"],
            }
            passed = passed and quality_checks["noise_level"]["passed"]

            # 7. File Size Check (estimated)
            estimated_size = (width * height * (4 if image.mode == "RGBA" else 3)) / (
                1024 * 1024
            )
            quality_checks["file_size"] = {
                "value": f"{estimated_size:.2f}MB",
                "required": f">= {self.config['quality']['min_file_size_mb']}MB",
                "passed": estimated_size >= self.config["quality"]["min_file_size_mb"],
            }
            passed = passed and quality_checks["file_size"]["passed"]

            return passed, quality_checks

        except Exception as e:
            logger.error(f"Error in quality verification: {str(e)}")
            return False, {"error": str(e)}

    def calculate_target_size(self, current_size, target_width):
        """Calculate target size maintaining aspect ratio"""
        current_width, current_height = current_size
        aspect_ratio = current_height / current_width
        target_height = int(target_width * aspect_ratio)
        return (target_width, target_height)

    def track_processing_time(self):
        """Context manager for tracking processing time"""

        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.end = time.time()
                self.duration = self.end - self.start

        return Timer()
