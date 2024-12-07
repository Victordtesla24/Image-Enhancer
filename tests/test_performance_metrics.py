"""Tests for performance-related metrics."""


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
