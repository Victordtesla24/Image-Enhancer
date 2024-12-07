"""Shared fixtures for quality management tests."""

import numpy as np
import pytest

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
