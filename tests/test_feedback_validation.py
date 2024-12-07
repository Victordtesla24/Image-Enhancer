"""Test suite for feedback validation."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.user_interface import FeedbackUI
from src.utils.model_learning.learning_manager import LearningManager


@pytest.fixture
def learning_manager():
    """Create learning manager instance."""
    return LearningManager()


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with patch("src.components.user_interface.st") as mock_st:
        mock_st.slider = MagicMock(return_value=3)
        mock_st.multiselect = MagicMock(return_value=["Noise"])
        mock_st.text_area = MagicMock(return_value="Test comment")
        yield mock_st


def test_feedback_validation_required_fields(learning_manager):
    """Test validation of required feedback fields."""
    # Test with missing fields
    invalid_feedback = {"quality_rating": 4}
    learning_manager.process_feedback(invalid_feedback)
    assert len(learning_manager.feedback_history) == 0

    # Test with all required fields
    valid_feedback = {
        "quality_rating": 4,
        "aspect_ratings": {"sharpness": 4, "color": 3},
        "enhancement_params": {"sharpness": 1.2},
    }
    learning_manager.process_feedback(valid_feedback)
    assert len(learning_manager.feedback_history) == 1


def test_feedback_validation_field_types(learning_manager):
    """Test validation of feedback field types."""
    # Test with invalid types
    invalid_feedback = {
        "quality_rating": "4",  # Should be number
        "aspect_ratings": [1, 2, 3],  # Should be dict
        "enhancement_params": "invalid",  # Should be dict
    }
    learning_manager.process_feedback(invalid_feedback)
    assert len(learning_manager.feedback_history) == 0

    # Test with valid types
    valid_feedback = {
        "quality_rating": 4,
        "aspect_ratings": {"sharpness": 4},
        "enhancement_params": {"contrast": 1.1},
    }
    learning_manager.process_feedback(valid_feedback)
    assert len(learning_manager.feedback_history) == 1


def test_feedback_validation_value_ranges(learning_manager):
    """Test validation of feedback value ranges."""
    # Test with out-of-range values
    invalid_feedback = {
        "quality_rating": 6,  # Should be 1-5
        "aspect_ratings": {"sharpness": -1},  # Should be 1-5
        "enhancement_params": {"contrast": 3.0},  # Should be 0.5-2.0
    }

    params = learning_manager._validate_parameters(
        invalid_feedback["enhancement_params"]
    )
    assert params["contrast"] == 2.0  # Clamped to max

    # Test with valid ranges
    valid_feedback = {
        "quality_rating": 4,
        "aspect_ratings": {"sharpness": 3},
        "enhancement_params": {"contrast": 1.5},
    }
    learning_manager.process_feedback(valid_feedback)
    assert len(learning_manager.feedback_history) == 1


def test_feedback_validation_timestamp(learning_manager):
    """Test validation and addition of timestamps."""
    # Test without timestamp
    feedback = {
        "quality_rating": 4,
        "aspect_ratings": {"sharpness": 4},
        "enhancement_params": {"contrast": 1.1},
    }
    learning_manager.process_feedback(feedback)
    assert len(learning_manager.feedback_history) == 1
    assert "timestamp" in learning_manager.feedback_history[0]

    # Verify timestamp format
    timestamp = learning_manager.feedback_history[0]["timestamp"]
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        pytest.fail("Invalid timestamp format")


def test_feedback_validation_metrics_correlation(learning_manager):
    """Test correlation between feedback and metrics."""
    metrics = {"sharpness": 0.7, "contrast": 0.8, "detail": 0.6}

    feedback = {
        "quality_rating": 4,
        "aspect_ratings": {"sharpness": 4, "contrast": 4, "detail": 3},
        "enhancement_params": {"sharpness": 1.2, "contrast": 1.1, "detail": 1.0},
    }

    # Test parameter adaptation
    adapted_params = learning_manager.adapt_enhancement_parameters(metrics, feedback)
    assert isinstance(adapted_params, dict)
    assert all(0.5 <= v <= 2.0 for v in adapted_params.values())


def test_feedback_validation_ui_integration(mock_streamlit):
    """Test feedback validation in UI."""
    ui = FeedbackUI()

    # Collect feedback through UI
    feedback = ui.collect_feedback(None, {"sharpness": 1.0})

    # Verify feedback structure
    assert isinstance(feedback, dict)
    assert feedback["quality_rating"] == 3  # From mock slider
    assert feedback["issues"] == ["Noise"]  # From mock multiselect
    assert feedback["comments"] == "Test comment"  # From mock text area
    assert isinstance(feedback["timestamp"], float)


def test_feedback_validation_history(learning_manager):
    """Test feedback history validation."""
    feedbacks = [
        {
            "quality_rating": i,
            "aspect_ratings": {"sharpness": i},
            "enhancement_params": {"contrast": 1.0},
        }
        for i in range(1, 4)
    ]

    # Add multiple feedbacks
    for feedback in feedbacks:
        learning_manager.process_feedback(feedback)

    assert len(learning_manager.feedback_history) == 3
    assert all("timestamp" in f for f in learning_manager.feedback_history)


def test_feedback_validation_preference_calculation(learning_manager):
    """Test validation of preference calculations."""
    feedback = {
        "aspect_ratings": {
            "sharpness": 4,
            "detail": 3,
            "color": 5,
            "noise": 2,
            "texture": 4,
        }
    }

    adjustments = learning_manager._calculate_preference_adjustments(feedback)

    # Verify adjustment ranges
    assert all(0.5 <= v <= 1.5 for v in adjustments.values())

    # Verify parameter mapping
    assert "sharpness" in adjustments
    assert "color_boost" in adjustments
    assert "noise_reduction" in adjustments
