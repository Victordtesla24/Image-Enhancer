"""Quality management module."""

import numpy as np
from typing import Dict, Any, Optional

from .basic_metrics import BasicMetricsCalculator
from .processing_accuracy import ProcessingAccuracyAnalyzer
from .quality_improvement import QualityImprovementAnalyzer
from .configuration import ConfigurationManager
from .performance_metrics import PerformanceMetricsCalculator


class QualityManager:
    """Manager for image quality analysis and improvement tracking."""

    def __init__(self):
        """Initialize quality manager."""
        self.config_manager = ConfigurationManager()
        self.basic_metrics = BasicMetricsCalculator()
        self.processing_accuracy = ProcessingAccuracyAnalyzer()
        self.quality_improvement = QualityImprovementAnalyzer()
        self.performance_metrics = PerformanceMetricsCalculator()
        self.initialized = False

    def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize the quality manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_manager.initialize(config_path)
        self.initialized = True

    @property
    def metrics_history(self):
        """Get metrics history."""
        return self.config_manager.get_metrics_history()

    @property
    def quality_thresholds(self):
        """Get quality thresholds."""
        return self.config_manager.quality_thresholds

    @property
    def accuracy_thresholds(self):
        """Get accuracy thresholds."""
        return self.config_manager.accuracy_thresholds

    def calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for an image.

        Args:
            image: Input image array

        Returns:
            Dictionary of quality metrics
        """
        if not self.initialized:
            self.initialize()

        if image is None:
            raise ValueError("Input image cannot be None")

        # Calculate basic metrics
        metrics = self.basic_metrics.calculate_metrics(image)

        # Calculate performance metrics
        performance_metrics = self.performance_metrics.calculate_metrics(image)
        metrics.update(performance_metrics)

        # Store metrics history
        self.config_manager.add_to_metrics_history(metrics.copy())

        return metrics

    def analyze_processing_accuracy(
        self, original: np.ndarray, processed: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze processing accuracy between original and processed images.

        Args:
            original: Original image array
            processed: Processed image array

        Returns:
            Analysis results dictionary
        """
        if not self.initialized:
            self.initialize()

        if original.shape != processed.shape:
            raise ValueError("Input images must have the same dimensions")

        # Check if images are identical
        if np.array_equal(original, processed):
            return {
                "metrics_comparison": {},
                "accuracy_scores": {"overall_accuracy": 1.0},
                "quality_improvement": {
                    "improvements": [],
                    "degradations": [],
                    "stable_metrics": [],
                    "significant_improvements": [],
                    "overall_score": 0.0,
                    "overall_improvement": 0.0,
                },
                "warnings": [],
                "recommendations": [],
            }

        # Calculate metrics for both images
        original_metrics = self.calculate_quality_metrics(original)
        processed_metrics = self.calculate_quality_metrics(processed)

        # Compare metrics
        metrics_comparison = self.quality_improvement.compare_metrics(
            original_metrics, processed_metrics
        )

        # Calculate accuracy scores
        accuracy_scores = self.processing_accuracy.calculate_accuracy_scores(
            original, processed, original_metrics, processed_metrics
        )

        # Analyze quality improvement
        quality_improvement = self.quality_improvement.analyze_quality_improvement(
            metrics_comparison
        )

        # Prepare analysis results
        analysis = {
            "metrics_comparison": metrics_comparison,
            "accuracy_scores": accuracy_scores,
            "quality_improvement": quality_improvement,
            "warnings": [],
            "recommendations": [],
        }

        # Generate feedback
        self.quality_improvement.generate_analysis_feedback(analysis)

        return analysis

    def _calculate_accuracy_scores(
        self, original: np.ndarray, processed: np.ndarray, original_metrics: Dict[str, float], processed_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate accuracy scores between original and processed images.
        
        Args:
            original: Original image array
            processed: Processed image array
            original_metrics: Original image metrics
            processed_metrics: Processed image metrics
            
        Returns:
            Dictionary of accuracy scores
        """
        return self.processing_accuracy.calculate_accuracy_scores(
            original, processed, original_metrics, processed_metrics
        )

    def _analyze_quality_improvement(self, metrics_comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze quality improvement from metrics comparison.
        
        Args:
            metrics_comparison: Metrics comparison dictionary
            
        Returns:
            Quality improvement analysis
        """
        return self.quality_improvement.analyze_quality_improvement(metrics_comparison)

    def _generate_analysis_feedback(self, analysis: Dict[str, Any]) -> None:
        """Generate warnings and recommendations based on analysis.
        
        Args:
            analysis: Processing analysis dictionary
        """
        self.quality_improvement.generate_analysis_feedback(analysis)
