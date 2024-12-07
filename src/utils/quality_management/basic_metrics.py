"""Basic image quality metrics module."""

import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class BasicMetricsCalculator:
    """Calculator for basic image quality metrics."""

    def calculate_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate basic quality metrics for an image.

        Args:
            image: Input image array

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        try:
            metrics["sharpness"] = self._calculate_sharpness(image)
            metrics["contrast"] = self._calculate_contrast(image)
            metrics["detail"] = self._calculate_detail_level(image)
            metrics["color"] = self._calculate_color_quality(image)
            metrics["noise"] = self._calculate_noise_level(image)
            metrics["texture"] = self._calculate_texture_preservation(image)
            metrics["pattern"] = self._calculate_pattern_retention(image)
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            metrics = {
                "sharpness": 0.0,
                "contrast": 0.0,
                "detail": 0.0,
                "color": 0.0,
                "noise": 0.0,
                "texture": 0.0,
                "pattern": 0.0,
            }
        return metrics

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = np.var(laplacian)
            return min(1.0, score / 500.0)
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            mean = np.sum(hist * np.arange(256))
            var = np.sum(hist * (np.arange(256) - mean) ** 2)
            std = np.sqrt(var)
            return min(1.0, std / 128.0)
        except Exception as e:
            logger.error(f"Error calculating contrast: {e}")
            return 0.0

    def _calculate_detail_level(self, image: np.ndarray) -> float:
        """Calculate image detail level."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            detail_score = np.mean(gradient_magnitude)
            return min(1.0, detail_score / 100.0)
        except Exception as e:
            logger.error(f"Error calculating detail level: {e}")
            return 0.0

    def _calculate_color_quality(self, image: np.ndarray) -> float:
        """Calculate color quality."""
        try:
            if len(image.shape) != 3:
                return 0.0
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            sat_mean = np.mean(hsv[:, :, 1])
            val_mean = np.mean(hsv[:, :, 2])
            sat_score = sat_mean / 255.0
            val_score = val_mean / 255.0
            return (sat_score + val_score) / 2.0
        except Exception as e:
            logger.error(f"Error calculating color quality: {e}")
            return 0.0

    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate image noise level."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise)
            return max(0.0, 1.0 - noise_level / 50.0)
        except Exception as e:
            logger.error(f"Error calculating noise level: {e}")
            return 0.0

    def _calculate_texture_preservation(self, image: np.ndarray) -> float:
        """Calculate texture preservation score."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            kernel = np.ones((5, 5), np.float32) / 25
            texture = cv2.filter2D(gray, -1, kernel)
            texture_score = np.mean(np.abs(gray - texture))
            return min(1.0, texture_score / 50.0)
        except Exception as e:
            logger.error(f"Error calculating texture preservation: {e}")
            return 0.0

    def _calculate_pattern_retention(self, image: np.ndarray) -> float:
        """Calculate pattern retention score."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            ksize = 31
            sigma = 4.0
            theta = 0
            lambd = 10.0
            gamma = 0.5
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
            filtered = cv2.filter2D(gray, -1, kernel)
            pattern_score = np.mean(filtered)
            return min(1.0, pattern_score / 128.0)
        except Exception as e:
            logger.error(f"Error calculating pattern retention: {e}")
            return 0.0
