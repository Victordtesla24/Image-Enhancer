"""Tests for basic quality metrics calculation."""

import pytest


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


@pytest.mark.parametrize(
    "metric",
    ["sharpness", "contrast", "detail", "color", "noise", "texture", "pattern"],
)
def test_individual_metrics(quality_manager, test_image, metric):
    """Test individual metric calculations."""
    metrics = quality_manager.calculate_quality_metrics(test_image)
    assert metric in metrics
    assert 0 <= metrics[metric] <= 1
