"""Test suite for model learning system."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.utils.model_learning.learning_manager import LearningManager


@pytest.fixture
def learning_manager(tmp_path):
    """Create learning manager with temporary directories."""
    feedback_dir = tmp_path / "feedback"
    return LearningManager(feedback_dir=str(feedback_dir))


@pytest.fixture
def sample_feedback():
    """Create sample feedback data."""
    return {
        "quality_rating": 4,
        "aspect_ratings": {"detail": 4, "color": 3, "noise": 4, "sharpness": 5},
        "enhancement_params": {
            "sharpness": 1.2,
            "contrast": 1.1,
            "brightness": 1.0,
            "detail": 1.3,
            "noise_reduction": 0.4,
        },
        "comments": "Good enhancement",
    }


def test_feedback_processing(learning_manager, sample_feedback):
    """Test feedback processing."""
    learning_manager.process_feedback(sample_feedback)

    # Verify feedback is stored
    assert len(learning_manager.feedback_history) == 1
    assert learning_manager.feedback_history[0]["quality_rating"] == 4

    # Verify feedback file is created
    feedback_files = list(Path(learning_manager.feedback_dir).glob("feedback_*.json"))
    assert len(feedback_files) == 1


def test_style_profile_creation(learning_manager, sample_feedback):
    """Test style profile creation."""
    profile_name = learning_manager.create_style_profile(sample_feedback)

    # Verify profile is created
    assert profile_name in learning_manager.style_profiles
    profile = learning_manager.style_profiles[profile_name]

    # Verify profile contents
    assert profile["parameters"] == sample_feedback["enhancement_params"]
    assert profile["ratings"] == sample_feedback["aspect_ratings"]
    assert profile["quality_rating"] == sample_feedback["quality_rating"]
    assert profile["feedback_count"] == 1


def test_style_recommendation(learning_manager, sample_feedback):
    """Test style recommendation."""
    # Create profile
    learning_manager.create_style_profile(sample_feedback)

    # Get recommendation
    metrics = {"detail": 0.7, "color": 0.6, "noise": 0.8, "sharpness": 0.9}
    params = learning_manager.get_style_recommendation(metrics)

    # Verify recommendation
    assert isinstance(params, dict)
    assert all(isinstance(v, (int, float)) for v in params.values())


def test_session_management(learning_manager):
    """Test learning session management."""
    # Start session
    session_id = learning_manager.start_session()
    assert learning_manager.current_session is not None
    assert learning_manager.current_session["id"] == session_id

    # End session
    learning_manager.end_session()
    assert learning_manager.current_session is None

    # Verify session file
    session_files = list(Path(learning_manager.feedback_dir).glob("session_*.json"))
    assert len(session_files) == 1


def test_profile_updating(learning_manager):
    """Test style profile updating."""
    # Create multiple feedback entries
    feedbacks = []
    for i in range(5):
        feedback = {
            "quality_rating": 4 + (i % 2),
            "aspect_ratings": {
                "detail": 4,
                "color": 3 + (i % 3),
                "noise": 4,
                "sharpness": 5,
            },
            "enhancement_params": {
                "sharpness": 1.2 + (i * 0.1),
                "contrast": 1.1,
                "brightness": 1.0,
                "detail": 1.3,
                "noise_reduction": 0.4,
            },
        }
        feedbacks.append(feedback)
        learning_manager.process_feedback(feedback)

    # Verify profiles are created and updated
    assert len(learning_manager.style_profiles) > 0

    # Verify profile parameters are averaged
    for profile in learning_manager.style_profiles.values():
        assert isinstance(profile["parameters"], dict)
        assert isinstance(profile["ratings"], dict)
        assert isinstance(profile["quality_rating"], (int, float))
        assert profile["feedback_count"] > 0


def test_learning_persistence(learning_manager, sample_feedback):
    """Test learning data persistence."""
    # Process feedback and create profile
    learning_manager.process_feedback(sample_feedback)
    learning_manager.create_style_profile(sample_feedback)

    # Verify files are created
    assert len(list(Path(learning_manager.feedback_dir).glob("feedback_*.json"))) > 0
    assert (
        len(list(Path(learning_manager.feedback_dir).glob("learning_state.json"))) == 1
    )


def test_edge_cases(learning_manager):
    """Test edge cases in learning system."""
    # Test empty feedback
    empty_feedback = {
        "quality_rating": 0,
        "aspect_ratings": {},
        "enhancement_params": {},
        "comments": "",
    }
    learning_manager.process_feedback(empty_feedback)

    # Test invalid metrics
    invalid_metrics = {"invalid": -1, "another_invalid": 2.5}
    params = learning_manager.get_style_recommendation(invalid_metrics)
    assert isinstance(params, dict)

    # Test missing feedback directory cleanup
    feedback_dir = learning_manager.feedback_dir
    learning_manager._cleanup_feedback_dir()
    assert not Path(feedback_dir).exists()


def test_concurrent_sessions(learning_manager, sample_feedback):
    """Test handling of concurrent sessions."""
    # Start first session
    session1_id = learning_manager.start_session()
    learning_manager.process_feedback(sample_feedback)

    # Try starting second session without ending first
    session2_id = learning_manager.start_session()
    assert session2_id != session1_id

    # Verify only latest session is active
    assert learning_manager.current_session["id"] == session2_id


def test_profile_matching(learning_manager, sample_feedback):
    """Test profile matching algorithm."""
    # Create diverse profiles
    feedbacks = []
    for i in range(3):
        feedback = sample_feedback.copy()
        feedback["aspect_ratings"] = {
            k: v + i for k, v in feedback["aspect_ratings"].items()
        }
        feedbacks.append(feedback)
        learning_manager.create_style_profile(feedback)

    # Test matching with different metrics
    test_metrics = {"detail": 0.8, "color": 0.7, "noise": 0.6, "sharpness": 0.9}
    params = learning_manager.get_style_recommendation(test_metrics)

    # Verify recommendation is within bounds
    assert all(0 <= v <= 2.0 for v in params.values())
