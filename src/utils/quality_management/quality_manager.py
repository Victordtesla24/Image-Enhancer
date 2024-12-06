"""Quality Management System"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for image quality metrics"""

    resolution: Tuple[int, int]
    dpi: Tuple[float, float]
    sharpness: float
    noise_level: float
    dynamic_range: int
    color_depth: int
    file_size_mb: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    color_accuracy: Optional[float] = None
    contrast_score: Optional[float] = None
    detail_score: Optional[float] = None


class QualityManager:
    """Manages image quality assessment and validation"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_quality_metrics(
        self, image: Image.Image, original: Optional[Image.Image] = None
    ) -> QualityMetrics:
        """Compute comprehensive quality metrics"""
        # Basic metrics
        resolution = image.size
        dpi = image.info.get("dpi", (72, 72))
        file_size_mb = (
            image.size[0]
            * image.size[1]
            * (4 if image.mode == "RGBA" else 3)
            / (1024 * 1024)
        )

        # Convert to numpy array for advanced metrics
        img_array = np.array(image)

        # Compute sharpness
        sharpness = self._compute_sharpness(img_array)

        # Compute noise level
        noise_level = self._compute_noise_level(img_array)

        # Compute dynamic range
        dynamic_range = self._compute_dynamic_range(img_array)

        # Get color depth
        color_depth = 32 if image.mode == "RGBA" else 24

        # Initialize comparison metrics
        psnr_value = None
        ssim_value = None
        color_accuracy = None

        # Compute comparison metrics if original image is provided
        if original is not None:
            original_array = np.array(original)
            if original_array.shape == img_array.shape:
                psnr_value = self._compute_psnr(original_array, img_array)
                ssim_value = self._compute_ssim(original_array, img_array)
                color_accuracy = self._compute_color_accuracy(original_array, img_array)

        # Compute additional metrics
        contrast_score = self._compute_contrast_score(img_array)
        detail_score = self._compute_detail_score(img_array)

        return QualityMetrics(
            resolution=resolution,
            dpi=dpi,
            sharpness=sharpness,
            noise_level=noise_level,
            dynamic_range=dynamic_range,
            color_depth=color_depth,
            file_size_mb=file_size_mb,
            psnr=psnr_value,
            ssim=ssim_value,
            color_accuracy=color_accuracy,
            contrast_score=contrast_score,
            detail_score=detail_score,
        )

    def validate_quality(self, metrics: QualityMetrics) -> Tuple[bool, Dict]:
        """Validate quality metrics against requirements"""
        validation_results = {}
        passed = True

        # Resolution check
        min_width = self.config["resolution"]["width"]
        min_height = self.config["resolution"]["height"]
        resolution_passed = (
            metrics.resolution[0] >= min_width and metrics.resolution[1] >= min_height
        )
        validation_results["resolution"] = {
            "passed": resolution_passed,
            "value": f"{metrics.resolution[0]}x{metrics.resolution[1]}",
            "required": f"{min_width}x{min_height}",
        }
        passed = passed and resolution_passed

        # DPI check
        dpi_passed = (
            metrics.dpi[0] >= self.config["quality"]["dpi"]
            and metrics.dpi[1] >= self.config["quality"]["dpi"]
        )
        validation_results["dpi"] = {
            "passed": dpi_passed,
            "value": f"{metrics.dpi[0]}, {metrics.dpi[1]}",
            "required": str(self.config["quality"]["dpi"]),
        }
        passed = passed and dpi_passed

        # Sharpness check
        sharpness_passed = metrics.sharpness >= self.config["quality"]["min_sharpness"]
        validation_results["sharpness"] = {
            "passed": sharpness_passed,
            "value": f"{metrics.sharpness:.2f}",
            "required": str(self.config["quality"]["min_sharpness"]),
        }
        passed = passed and sharpness_passed

        # Noise level check
        noise_passed = metrics.noise_level <= self.config["quality"]["max_noise_level"]
        validation_results["noise_level"] = {
            "passed": noise_passed,
            "value": f"{metrics.noise_level:.2f}",
            "required": f"<= {self.config['quality']['max_noise_level']}",
        }
        passed = passed and noise_passed

        # Dynamic range check
        range_passed = (
            metrics.dynamic_range >= self.config["color"]["dynamic_range"]["min"]
        )
        validation_results["dynamic_range"] = {
            "passed": range_passed,
            "value": str(metrics.dynamic_range),
            "required": str(self.config["color"]["dynamic_range"]["min"]),
        }
        passed = passed and range_passed

        # Color depth check
        depth_passed = metrics.color_depth >= self.config["color"]["bit_depth"]
        validation_results["color_depth"] = {
            "passed": depth_passed,
            "value": f"{metrics.color_depth}-bit",
            "required": f"{self.config['color']['bit_depth']}-bit",
        }
        passed = passed and depth_passed

        # File size check
        size_passed = metrics.file_size_mb >= self.config["quality"]["min_file_size_mb"]
        validation_results["file_size"] = {
            "passed": size_passed,
            "value": f"{metrics.file_size_mb:.2f}MB",
            "required": f">= {self.config['quality']['min_file_size_mb']}MB",
        }
        passed = passed and size_passed

        return passed, validation_results

    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _compute_noise_level(self, img: np.ndarray) -> float:
        """Compute image noise level"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Use wavelet transform to estimate noise
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        diff = cv2.absdiff(gray, noise)
        return np.mean(diff)

    def _compute_dynamic_range(self, img: np.ndarray) -> int:
        """Compute image dynamic range"""
        return int(np.max(img) - np.min(img))

    def _compute_psnr(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        return psnr(original, enhanced)

    def _compute_ssim(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Compute Structural Similarity Index"""
        return ssim(original, enhanced, multichannel=True)

    def _compute_color_accuracy(
        self, original: np.ndarray, enhanced: np.ndarray
    ) -> float:
        """Compute color accuracy between original and enhanced images"""
        if len(original.shape) != 3 or len(enhanced.shape) != 3:
            return 0.0

        # Convert to LAB color space
        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)

        # Calculate color difference
        diff = np.mean(np.abs(original_lab - enhanced_lab))
        max_diff = 255.0  # Maximum possible difference in LAB space

        # Convert to similarity score (0-1)
        return 1.0 - (diff / max_diff)

    def _compute_contrast_score(self, img: np.ndarray) -> float:
        """Compute image contrast score"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Calculate contrast using histogram spread
        bins = np.arange(256)
        mean = np.sum(bins * hist)
        variance = np.sum(((bins - mean) ** 2) * hist)

        # Normalize score
        return min(1.0, np.sqrt(variance) / 128.0)

    def _compute_detail_score(self, img: np.ndarray) -> float:
        """Compute image detail preservation score"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize score
        return min(1.0, np.mean(gradient_magnitude) / 128.0)

    def suggest_improvements(self, metrics: QualityMetrics) -> Dict:
        """Suggest improvements based on quality metrics"""
        suggestions = {}

        # Resolution suggestions
        if metrics.resolution[0] < self.config["resolution"]["width"]:
            suggestions["resolution"] = (
                f"Increase resolution to at least {self.config['resolution']['width']}x"
                f"{self.config['resolution']['height']}"
            )

        # DPI suggestions
        if metrics.dpi[0] < self.config["quality"]["dpi"]:
            suggestions["dpi"] = (
                f"Increase DPI to at least {self.config['quality']['dpi']}"
            )

        # Sharpness suggestions
        if metrics.sharpness < self.config["quality"]["min_sharpness"]:
            suggestions["sharpness"] = (
                "Increase image sharpness using detail enhancement with higher "
                "sharpening parameters"
            )

        # Noise suggestions
        if metrics.noise_level > self.config["quality"]["max_noise_level"]:
            suggestions["noise"] = (
                "Reduce image noise using noise reduction with appropriate "
                "strength to preserve details"
            )

        # Dynamic range suggestions
        if metrics.dynamic_range < self.config["color"]["dynamic_range"]["min"]:
            suggestions["dynamic_range"] = (
                "Improve dynamic range using color enhancement with increased "
                "contrast and proper white balance"
            )

        return suggestions
