"""Image enhancement module using OpenCV and PIL"""

import cv2
import numpy as np
from PIL import Image
import yaml
import os
import time
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.classes")


class ImageEnhancer:
    def __init__(self):
        logger.info("Initializing ImageEnhancer")
        self.config = self._load_config()
        self.device = "cpu"  # Simplified since we're using OpenCV
        logger.info(f"Using device: {self.device}")

        # Initialize models list with updated descriptions
        self.models = [
            {
                "name": "Super Resolution",
                "description": "Intelligently upscales image resolution using advanced multi-step processing with Lanczos resampling and adaptive sharpening",
                "internal_name": "super_resolution",
            },
            {
                "name": "Color Enhancement",
                "description": "Optimizes color balance and vibrancy using LAB color space processing and adaptive contrast enhancement",
                "internal_name": "color_enhancement",
            },
            {
                "name": "Detail Enhancement",
                "description": "Enhances image details using multi-scale contrast enhancement and advanced noise reduction techniques",
                "internal_name": "detail_enhancement",
            },
        ]

        logger.info("ImageEnhancer initialized successfully")

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

    def get_available_models(self):
        """Return list of available enhancement models"""
        return [
            {"name": model["name"], "description": model["description"]}
            for model in self.models
        ]

    def enhance_image(self, input_image, target_width, models, progress_callback=None):
        """Enhance image using selected models"""
        try:
            logger.info(f"Starting image enhancement with models: {models}")
            start_time = time.time()

            # Convert PIL Image to numpy array
            img_array = np.array(input_image)

            # Store original size for enhancement details
            source_size = input_image.size
            logger.info(f"Source image size: {source_size}")

            # Track which models were used
            models_used = []

            # Calculate total steps for progress tracking
            total_steps = len(models)
            current_step = 0

            # Process image with selected models
            for model_name in models:
                logger.info(f"Processing with model: {model_name}")
                if progress_callback:
                    progress_callback(
                        current_step / total_steps, f"Applying {model_name}..."
                    )

                # Get internal model name
                internal_name = model_name.lower().replace(" ", "_")

                # Apply appropriate enhancement based on model
                if internal_name == "super_resolution":
                    img_array = self._apply_super_resolution(img_array, target_width)
                    models_used.append(
                        {
                            "name": "Super Resolution",
                            "description": "Enhanced resolution using multi-step upscaling and detail preservation",
                        }
                    )

                elif internal_name == "color_enhancement":
                    img_array = self._apply_color_enhancement(img_array)
                    models_used.append(
                        {
                            "name": "Color Enhancement",
                            "description": "Enhanced color balance and vibrancy using LAB color processing",
                        }
                    )

                elif internal_name == "detail_enhancement":
                    img_array = self._apply_detail_enhancement(img_array)
                    models_used.append(
                        {
                            "name": "Detail Enhancement",
                            "description": "Enhanced image details and sharpness with noise reduction",
                        }
                    )

                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps, f"Processing...")

            # Convert final result to PIL Image
            logger.info("Converting final result to PIL Image")
            enhanced_img = Image.fromarray(img_array)

            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Enhancement completed in {processing_time:.2f} seconds")

            # Prepare enhancement details
            enhancement_details = {
                "source_size": f"{source_size[0]}x{source_size[1]}",
                "target_size": f"{enhanced_img.size[0]}x{enhanced_img.size[1]}",
                "models_used": models_used,
                "processing_time": f"{processing_time:.2f} seconds",
            }

            return enhanced_img, enhancement_details

        except Exception as e:
            logger.error(f"Error during image enhancement: {str(e)}")
            raise

    def _apply_super_resolution(self, img, target_width):
        """Apply super resolution enhancement"""
        try:
            logger.info("Applying super resolution")

            # Convert to BGR for OpenCV processing
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            current_height, current_width = img.shape[:2]
            scale_factor = target_width / current_width
            target_height = int(current_height * scale_factor)

            logger.info(
                f"Scaling from {current_width}x{current_height} to {target_width}x{target_height}"
            )

            # Use multiple steps for better quality upscaling
            while scale_factor > 2:
                # Apply Gaussian blur before upscaling to reduce noise
                img = cv2.GaussianBlur(img, (0, 0), 1.0)

                # Upscale by 2x using Lanczos
                img = cv2.resize(
                    img,
                    (int(current_width * 2), int(current_height * 2)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

                # Apply adaptive sharpening
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9.0
                img = cv2.filter2D(img, -1, kernel)

                current_width *= 2
                current_height *= 2
                scale_factor /= 2

            # Final resize to target resolution using Lanczos
            img = cv2.resize(
                img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
            )

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Super resolution completed")
            return img

        except Exception as e:
            logger.error(f"Error in super resolution: {str(e)}")
            raise

    def _apply_color_enhancement(self, img):
        """Apply color enhancement"""
        try:
            logger.info("Applying color enhancement")

            # Convert to BGR for OpenCV processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Convert to LAB color space for better color processing
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)

            # Enhance lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Enhance color channels with careful adjustment
            a = cv2.convertScaleAbs(a, alpha=1.1, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)

            # Merge channels
            img_lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

            # Fine-tune overall contrast and brightness
            img = cv2.convertScaleAbs(img, alpha=1.1, beta=3)

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Color enhancement completed")
            return img

        except Exception as e:
            logger.error(f"Error in color enhancement: {str(e)}")
            raise

    def _apply_detail_enhancement(self, img):
        """Apply detail enhancement"""
        try:
            logger.info("Applying detail enhancement")

            # Convert to BGR for OpenCV processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Initial noise reduction with detail preservation
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
            )

            # Multi-space enhancement
            # 1. YUV space for luminance enhancement
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # 2. LAB space for local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)
            l = clahe.apply(l)
            img_lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

            # Advanced detail enhancement
            # 1. Unsharp masking for overall detail boost
            gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

            # 2. Adaptive sharpening for fine details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
            img = cv2.filter2D(img, -1, kernel)

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Detail enhancement completed")
            return img

        except Exception as e:
            logger.error(f"Error in detail enhancement: {str(e)}")
            raise

    def verify_5k_quality(self, image_path):
        """Verify if image meets 5K quality standards"""
        try:
            logger.info(f"Starting quality verification for {image_path}")
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

            # 3. DPI Check with tolerance
            dpi = img.info.get("dpi", (72, 72))
            results["metrics"]["dpi"] = f"{dpi[0]:.2f}, {dpi[1]:.2f}"
            tolerance = 0.01  # 1% tolerance
            min_dpi = self.config["quality"]["dpi"] * (1 - tolerance)
            if dpi[0] < min_dpi or dpi[1] < min_dpi:
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

            logger.info("Quality verification completed")
            return results

        except Exception as e:
            logger.error(f"Error in quality verification: {str(e)}")
            return {
                "passed": False,
                "metrics": {},
                "failures": [f"Error during verification: {str(e)}"],
            }
