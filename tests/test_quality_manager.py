"""Test suite for quality management."""

import numpy as np
import pytest
from PIL import Image

from src.utils.quality_management.quality_manager import QualityManager


@pytest.fixture
def quality_manager():
    """Create quality manager instance."""
    return QualityManager()


@pytest.fixture
def test_image():
    """Create test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # Add white square
    return img


@pytest.fixture
def processed_image():
    """Create processed test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 200  # Add lighter square
    return img


def test_quality_metrics_calculation(quality_manager, test_image):
    """Test quality metrics calculation."""
    metrics = quality_manager.calculate_quality_metrics(test_image)

    # Check basic metrics
    assert "sharpness" in metrics
    assert "contrast" in metrics
    assert "detail" in metrics
    assert "color" in metrics
    assert "noise" in metrics
    assert "texture" in metrics
    assert "pattern" in metrics

    # Check advanced metrics
    assert "edge_preservation" in metrics
    assert "color_consistency" in metrics
    assert "local_contrast" in metrics
    assert "artifact_level" in metrics
    assert "dynamic_range" in metrics

    # Check value ranges
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {metric} out of range: {value}"


def test_processing_accuracy_analysis(quality_manager, test_image, processed_image):
    """Test processing accuracy analysis."""
    analysis = quality_manager.analyze_processing_accuracy(test_image, processed_image)

    # Check analysis structure
    assert "metrics_comparison" in analysis
    assert "accuracy_scores" in analysis
    assert "quality_improvement" in analysis
    assert "warnings" in analysis
    assert "recommendations" in analysis

    # Check metrics comparison
    for metric, comparison in analysis["metrics_comparison"].items():
        assert "original" in comparison
        assert "processed" in comparison
        assert "improvement" in comparison
        assert "improvement_percentage" in comparison

    # Check accuracy scores
    scores = analysis["accuracy_scores"]
    assert "structural_similarity" in scores
    assert "psnr" in scores
    assert "feature_preservation" in scores
    assert "color_accuracy" in scores
    assert "overall_accuracy" in scores

    # Check quality improvement
    improvement = analysis["quality_improvement"]
    assert "overall_improvement" in improvement
    assert "significant_improvements" in improvement
    assert "degradations" in improvement
    assert "stable_metrics" in improvement


def test_accuracy_scores_calculation(quality_manager, test_image, processed_image):
    """Test accuracy scores calculation."""
    scores = quality_manager._calculate_accuracy_scores(
        test_image,
        processed_image,
        quality_manager.calculate_quality_metrics(test_image),
        quality_manager.calculate_quality_metrics(processed_image),
    )

    # Check all scores
    assert 0 <= scores["structural_similarity"] <= 1
    assert scores["psnr"] >= 0
    assert 0 <= scores["feature_preservation"] <= 1
    assert 0 <= scores["color_accuracy"] <= 1
    assert 0 <= scores["overall_accuracy"] <= 1


def test_quality_improvement_analysis(quality_manager):
    """Test quality improvement analysis."""
    metrics_comparison = {
        "sharpness": {
            "original": 0.5,
            "processed": 0.6,
            "improvement": 0.1,
            "improvement_percentage": 20.0,
        },
        "contrast": {
            "original": 0.7,
            "processed": 0.65,
            "improvement": -0.05,
            "improvement_percentage": -7.14,
        },
        "detail": {
            "original": 0.8,
            "processed": 0.805,
            "improvement": 0.005,
            "improvement_percentage": 0.625,
        },
    }

    analysis = quality_manager._analyze_quality_improvement(metrics_comparison)

    # Check analysis structure
    assert isinstance(analysis["overall_improvement"], float)
    assert isinstance(analysis["significant_improvements"], list)
    assert isinstance(analysis["degradations"], list)
    assert isinstance(analysis["stable_metrics"], list)

    # Check categorization
    assert len(analysis["significant_improvements"]) == 1  # sharpness
    assert len(analysis["degradations"]) == 1  # contrast
    assert len(analysis["stable_metrics"]) == 1  # detail


def test_analysis_feedback_generation(quality_manager):
    """Test analysis feedback generation."""
    analysis = {
        "accuracy_scores": {
            "structural_similarity": 0.8,
            "color_accuracy": 0.85,
            "overall_accuracy": 0.75,
        },
        "quality_improvement": {
            "degradations": [
                {"metric": "sharpness", "degradation": 10.5},
                {"metric": "contrast", "degradation": 7.2},
            ]
        },
        "warnings": [],
        "recommendations": [],
    }

    quality_manager._generate_analysis_feedback(analysis)

    # Check warnings
    assert len(analysis["warnings"]) >= 3  # 2 degradations + low overall accuracy

    # Check recommendations
    assert (
        len(analysis["recommendations"]) >= 2
    )  # General adjustment + specific parameters


def test_edge_cases(quality_manager):
    """Test edge cases in quality analysis."""
    # Test with empty image
    empty_image = np.zeros((1, 1, 3), dtype=np.uint8)
    metrics = quality_manager.calculate_quality_metrics(empty_image)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with saturated image
    saturated_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    metrics = quality_manager.calculate_quality_metrics(saturated_image)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with same image (perfect accuracy)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    analysis = quality_manager.analyze_processing_accuracy(test_image, test_image)
    assert analysis["accuracy_scores"]["overall_accuracy"] == 1.0
    assert not analysis["warnings"]


def test_metrics_history(quality_manager, test_image):
    """Test metrics history tracking."""
    # Calculate metrics multiple times
    for _ in range(3):
        quality_manager.calculate_quality_metrics(test_image)

    assert len(quality_manager.metrics_history) == 3
    assert all(isinstance(m, dict) for m in quality_manager.metrics_history)


def test_quality_thresholds(quality_manager):
    """Test quality thresholds configuration."""
    assert all(0 <= v <= 1 for v in quality_manager.quality_thresholds.values())
    assert all(
        isinstance(v, float) for v in quality_manager.quality_thresholds.values()
    )


def test_accuracy_thresholds(quality_manager):
    """Test accuracy thresholds configuration."""
    assert all(0 <= v <= 1 for v in quality_manager.accuracy_thresholds.values())
    assert all(
        isinstance(v, float) for v in quality_manager.accuracy_thresholds.values()
    )


@pytest.mark.parametrize(
    "metric",
    ["sharpness", "contrast", "detail", "color", "noise", "texture", "pattern"],
)
def test_individual_metrics(quality_manager, test_image, metric):
    """Test individual metric calculations."""
    metrics = quality_manager.calculate_quality_metrics(test_image)
    assert metric in metrics
    assert 0 <= metrics[metric] <= 1


def test_error_handling(quality_manager):
    """Test error handling in quality analysis."""
    # Test with invalid image
    with pytest.raises(ValueError):
        quality_manager.calculate_quality_metrics(None)

    # Test with mismatched image sizes
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        quality_manager.analyze_processing_accuracy(img1, img2)


def test_performance_metrics(quality_manager, test_image):
    """Test performance-related metrics calculation."""
    metrics = quality_manager.calculate_quality_metrics(test_image)

    # Check performance-related metrics
    assert "artifact_level" in metrics
    assert "dynamic_range" in metrics
    assert "local_contrast" in metrics

    # Check value ranges
    assert 0 <= metrics["artifact_level"] <= 1
    assert 0 <= metrics["dynamic_range"] <= 1
    assert 0 <= metrics["local_contrast"] <= 1
