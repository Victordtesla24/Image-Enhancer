"""Configuration management module."""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manager for quality configuration and state."""

    def __init__(self):
        """Initialize configuration manager."""
        self.initialized = False
        self.quality_thresholds: Dict[str, float] = {}
        self.accuracy_thresholds: Dict[str, float] = {}
        self.metrics_history: List[Dict[str, float]] = []
        self.config: Dict[str, Any] = {}

    def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration with optional custom config.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_thresholds()
        self.initialized = True

    def _setup_thresholds(self) -> None:
        """Setup quality and accuracy thresholds from config."""
        self.quality_thresholds = self.config["quality"]["thresholds"]
        self.accuracy_thresholds = {
            "structural_similarity": 0.85,
            "feature_preservation": 0.75,
            "color_accuracy": 0.8,
            "overall_accuracy": 0.8,
        }

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")

        # Use default configuration
        return {
            "quality": {
                "thresholds": {
                    "sharpness": 0.7,
                    "contrast": 0.65,
                    "detail": 0.6,
                    "color": 0.8,
                    "noise": 0.2,
                    "texture": 0.75,
                    "pattern": 0.7,
                },
                "weights": {
                    "sharpness": 1.0,
                    "contrast": 1.0,
                    "detail": 1.0,
                    "color": 1.0,
                    "noise": 0.8,
                    "texture": 0.9,
                    "pattern": 0.7,
                },
            },
            "processing": {
                "batch_size": 1,
                "use_gpu": True,
                "precision": "float32",
                "max_iterations": 5,
            },
            "analysis": {
                "min_quality_score": 0.8,
                "max_quality_variance": 0.1,
                "min_improvement_rate": 0.05,
            },
        }

    def add_to_metrics_history(self, metrics: Dict[str, float]) -> None:
        """Add metrics to history.

        Args:
            metrics: Quality metrics dictionary
        """
        self.metrics_history.append(metrics.copy())

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get metrics history.

        Returns:
            List of historical metrics
        """
        return self.metrics_history

    def get_quality_threshold(self, metric: str) -> float:
        """Get quality threshold for specific metric.

        Args:
            metric: Metric name

        Returns:
            Threshold value
        """
        return self.quality_thresholds.get(metric, 0.0)

    def get_accuracy_threshold(self, metric: str) -> float:
        """Get accuracy threshold for specific metric.

        Args:
            metric: Metric name

        Returns:
            Threshold value
        """
        return self.accuracy_thresholds.get(metric, 0.0)

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration.

        Returns:
            Processing configuration dictionary
        """
        return self.config["processing"]

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration.

        Returns:
            Analysis configuration dictionary
        """
        return self.config["analysis"]
