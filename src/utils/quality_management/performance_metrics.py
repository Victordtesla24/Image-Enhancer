"""Performance-related metrics module."""

import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """Calculator for performance-related image metrics."""

    def calculate_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for an image.

        Args:
            image: Input image array

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        try:
            metrics["edge_preservation"] = self._analyze_edge_preservation(image)
            metrics["color_consistency"] = self._analyze_color_consistency(image)
            metrics["local_contrast"] = self._analyze_local_contrast(image)
            metrics["artifact_level"] = self._detect_artifacts(image)
            metrics["dynamic_range"] = self._calculate_dynamic_range(image)
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            metrics = {
                "edge_preservation": 0.0,
                "color_consistency": 0.0,
                "local_contrast": 0.0,
                "artifact_level": 0.0,
                "dynamic_range": 0.0,
            }
        return metrics

    def _analyze_edge_preservation(self, image: np.ndarray) -> float:
        """Analyze edge preservation quality."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges > 0)

            # Apply Sobel operators
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_strength = np.mean(gradient_magnitude)

            # Combine metrics
            edge_score = (edge_density + min(1.0, gradient_strength / 100.0)) / 2.0

            return edge_score
        except Exception as e:
            logger.error(f"Error analyzing edge preservation: {e}")
            return 0.0

    def _analyze_color_consistency(self, image: np.ndarray) -> float:
        """Analyze color consistency."""
        try:
            if len(image.shape) != 3:
                return 0.0

            # Convert to Lab color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Calculate color statistics per channel
            channel_stats = []
            for channel in range(3):
                values = lab[:, :, channel].flatten()
                stats = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "range": np.ptp(values),
                }
                channel_stats.append(stats)

            # Calculate consistency metrics
            consistency_scores = []

            # Color spread
            spread_score = 1.0 - min(1.0, np.mean([s["std"] for s in channel_stats]) / 50.0)
            consistency_scores.append(spread_score)

            # Color balance
            means = [s["mean"] for s in channel_stats]
            balance_score = 1.0 - min(1.0, np.std(means) / 20.0)
            consistency_scores.append(balance_score)

            # Color range utilization
            range_score = min(1.0, np.mean([s["range"] for s in channel_stats]) / 100.0)
            consistency_scores.append(range_score)

            # Combine scores
            return np.mean(consistency_scores)
        except Exception as e:
            logger.error(f"Error analyzing color consistency: {e}")
            return 0.0

    def _analyze_local_contrast(self, image: np.ndarray) -> float:
        """Analyze local contrast."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Calculate local standard deviation
            local_std = cv2.GaussianBlur(gray, (3, 3), 0)
            local_std = cv2.subtract(gray, local_std)
            local_std = np.std(local_std)

            # Normalize score
            return min(1.0, local_std / 50.0)
        except Exception as e:
            logger.error(f"Error analyzing local contrast: {e}")
            return 0.0

    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect image artifacts."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else gray

            # Apply bilateral filter
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Calculate difference
            diff = cv2.absdiff(gray, filtered)

            # Calculate artifact score
            artifact_level = np.mean(diff)

            # Normalize and invert score
            return max(0.0, 1.0 - artifact_level / 30.0)
        except Exception as e:
            logger.error(f"Error detecting artifacts: {e}")
            return 0.0

    def _calculate_dynamic_range(self, image: np.ndarray) -> float:
        """Calculate image dynamic range."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()

            # Find non-zero range
            non_zero = np.where(hist > 0.001)[0]
            if len(non_zero) < 2:
                return 0.0

            dynamic_range = (non_zero[-1] - non_zero[0]) / 255.0

            return dynamic_range
        except Exception as e:
            logger.error(f"Error calculating dynamic range: {e}")
            return 0.0
