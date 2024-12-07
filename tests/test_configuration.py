"""Tests for configuration and history tracking."""


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
