"""Processing accuracy analysis module."""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

logger = logging.getLogger(__name__)


class ProcessingAccuracyAnalyzer:
    """Analyzer for processing accuracy metrics."""

    def __init__(self):
        """Initialize the analyzer."""
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def calculate_accuracy_scores(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        original_metrics: Dict[str, float],
        processed_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate accuracy scores for processing results.

        Args:
            original: Original image array
            processed: Processed image array
            original_metrics: Original image metrics
            processed_metrics: Processed image metrics

        Returns:
            Dictionary of accuracy scores
        """
        scores = {}
        try:
            # Structural similarity
            scores["structural_similarity"] = self._calculate_ssim(original, processed)

            # Peak signal-to-noise ratio
            scores["psnr"] = self._calculate_psnr(original, processed)

            # Feature preservation
            scores["feature_preservation"] = self._calculate_feature_preservation(
                original, processed
            )

            # Color accuracy
            scores["color_accuracy"] = self._calculate_color_accuracy(original, processed)

            # Overall accuracy score
            scores["overall_accuracy"] = np.mean([
                scores["structural_similarity"],
                scores["feature_preservation"],
                scores["color_accuracy"],
            ])
        except Exception as e:
            logger.error(f"Error calculating accuracy scores: {e}")
            scores = {
                "structural_similarity": 0.0,
                "psnr": 0.0,
                "feature_preservation": 0.0,
                "color_accuracy": 0.0,
                "overall_accuracy": 0.0,
            }

        return scores

    def _calculate_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate structural similarity index."""
        try:
            return structural_similarity(
                original,
                processed,
                channel_axis=2 if len(original.shape) == 3 else None,
            )
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return 0.0

    def _calculate_psnr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate peak signal-to-noise ratio."""
        try:
            return peak_signal_noise_ratio(original, processed)
        except Exception as e:
            logger.error(f"Error calculating PSNR: {e}")
            return 0.0

    def _calculate_feature_preservation(
        self, original: np.ndarray, processed: np.ndarray
    ) -> float:
        """Calculate feature preservation score."""
        try:
            # Extract features
            original_features = self._extract_features(original)
            processed_features = self._extract_features(processed)

            # Calculate matching score
            return self._match_features(original_features, processed_features)
        except Exception as e:
            logger.error(f"Error calculating feature preservation: {e}")
            return 0.0

    def _calculate_color_accuracy(
        self, original: np.ndarray, processed: np.ndarray
    ) -> float:
        """Calculate color accuracy score."""
        try:
            # Convert to Lab color space
            original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
            processed_lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)

            # Calculate color difference
            delta_e = np.mean(
                np.sqrt(np.sum((original_lab - processed_lab) ** 2, axis=2))
            )

            # Convert to score (0-1)
            return max(0, 1 - delta_e / 100)
        except Exception as e:
            logger.error(f"Error calculating color accuracy: {e}")
            return 0.0

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract image features for comparison."""
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            return descriptors if descriptors is not None else np.array([])
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])

    def _match_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Match features between two feature sets."""
        try:
            if len(features1) == 0 or len(features2) == 0:
                return 0.0

            # Match descriptors
            matches = self.matcher.match(features1, features2)

            if not matches:
                return 0.0

            # Calculate matching score
            distances = [m.distance for m in matches]
            avg_distance = np.mean(distances)
            max_distance = max(distances)

            return 1.0 - (avg_distance / max_distance) if max_distance > 0 else 1.0
        except Exception as e:
            logger.error(f"Error matching features: {e}")
            return 0.0
