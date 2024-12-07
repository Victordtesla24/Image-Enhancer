"""Tests for edge cases and error handling."""

import numpy as np
import pytest


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
