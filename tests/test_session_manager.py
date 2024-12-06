"""Test suite for session management system"""

import os
import pytest
import json
from datetime import datetime
from pathlib import Path
from src.utils.session_management.session_manager import (
    SessionManager,
    QualityPreferences,
    EnhancementAttempt
)

@pytest.fixture
def session_manager():
    """Create SessionManager instance with test session ID"""
    manager = SessionManager("test_session")
    yield manager
    # Cleanup test files
    session_file = Path("sessions") / "session_test_session.json"
    if session_file.exists():
        session_file.unlink()
    if Path("sessions").exists():
        Path("sessions").rmdir()

@pytest.fixture
def test_image_hash():
    """Create test image hash"""
    return "test_hash_123"

def test_session_initialization(session_manager):
    """Test session manager initialization"""
    assert session_manager.session_id == "test_session"
    assert isinstance(session_manager.quality_preferences, QualityPreferences)
    assert session_manager.enhancement_attempts == []
    assert session_manager.current_image_hash is None

def test_quality_preferences_update(session_manager):
    """Test updating quality preferences"""
    new_preferences = {
        "min_resolution": (3840, 2160),
        "min_dpi": 200,
        "min_sharpness": 60.0
    }
    session_manager.update_quality_preferences(new_preferences)
    
    assert session_manager.quality_preferences.min_resolution == (3840, 2160)
    assert session_manager.quality_preferences.min_dpi == 200
    assert session_manager.quality_preferences.min_sharpness == 60.0

def test_enhancement_attempt_recording(session_manager, test_image_hash):
    """Test recording enhancement attempts"""
    models_used = ["Super Resolution", "Color Enhancement"]
    parameters = {
        "super_resolution": {"scale_factor": 2.0},
        "color_enhancement": {"saturation": 1.2}
    }
    quality_metrics = {
        "resolution": "5120x2880",
        "sharpness": 75.0
    }
    
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=models_used,
        parameters=parameters,
        quality_metrics=quality_metrics,
        success=True,
        feedback="Good enhancement"
    )
    
    assert len(session_manager.enhancement_attempts) == 1
    attempt = session_manager.enhancement_attempts[0]
    assert attempt.input_image_hash == test_image_hash
    assert attempt.models_used == models_used
    assert attempt.parameters == parameters
    assert attempt.quality_metrics == quality_metrics
    assert attempt.success is True
    assert attempt.feedback == "Good enhancement"

def test_enhancement_history_retrieval(session_manager, test_image_hash):
    """Test retrieving enhancement history"""
    # Record multiple attempts
    for i in range(3):
        session_manager.record_enhancement_attempt(
            input_image_hash=test_image_hash,
            models_used=["Super Resolution"],
            parameters={"super_resolution": {"scale_factor": 2.0}},
            quality_metrics={"resolution": "5120x2880"},
            success=True
        )
    
    # Get history for specific image
    history = session_manager.get_enhancement_history(test_image_hash)
    assert len(history) == 3
    assert all(attempt.input_image_hash == test_image_hash for attempt in history)
    
    # Get all history
    all_history = session_manager.get_enhancement_history()
    assert len(all_history) == 3

def test_successful_parameters_retrieval(session_manager, test_image_hash):
    """Test retrieving successful enhancement parameters"""
    # Record successful and failed attempts
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics={"resolution": "5120x2880"},
        success=True
    )
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 3.0}},
        quality_metrics={"resolution": "5120x2880"},
        success=False
    )
    
    successful_params = session_manager.get_successful_parameters(test_image_hash)
    assert "super_resolution" in successful_params
    assert successful_params["super_resolution"]["scale_factor"] == 2.0

def test_enhancement_suggestions(session_manager, test_image_hash):
    """Test getting enhancement suggestions"""
    # Record multiple successful attempts with different parameters
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics={"resolution": "5120x2880"},
        success=True
    )
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.5}},
        quality_metrics={"resolution": "5120x2880"},
        success=True
    )
    
    suggestions = session_manager.get_enhancement_suggestions(test_image_hash)
    assert "super_resolution" in suggestions
    assert 2.0 <= suggestions["super_resolution"]["scale_factor"] <= 2.5

def test_feedback_application(session_manager, test_image_hash):
    """Test applying user feedback"""
    feedback = {
        "super_resolution": {
            "scale_factor": 1  # Increase scale factor
        }
    }
    session_manager.apply_feedback(test_image_hash, feedback)
    
    # Verify feedback was recorded
    history = session_manager.get_enhancement_history(test_image_hash)
    metrics_summary = session_manager.get_quality_metrics_summary(test_image_hash)
    assert isinstance(metrics_summary, dict)

def test_session_persistence(session_manager, test_image_hash):
    """Test session state persistence"""
    # Record an attempt
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics={"resolution": "5120x2880"},
        success=True
    )
    
    # Create new session manager with same ID
    new_session = SessionManager("test_session")
    
    # Verify state was loaded
    assert len(new_session.enhancement_attempts) == 1
    attempt = new_session.enhancement_attempts[0]
    assert attempt.input_image_hash == test_image_hash
    assert attempt.success is True

def test_quality_metrics_summary(session_manager, test_image_hash):
    """Test quality metrics summary generation"""
    # Record attempts with different metrics
    metrics1 = {"sharpness": 70.0, "noise_level": 100.0}
    metrics2 = {"sharpness": 80.0, "noise_level": 90.0}
    
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics=metrics1,
        success=True
    )
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics=metrics2,
        success=True
    )
    
    summary = session_manager.get_quality_metrics_summary(test_image_hash)
    assert "sharpness" in summary
    assert "noise_level" in summary
    assert summary["sharpness"]["mean"] == 75.0
    assert summary["sharpness"]["min"] == 70.0
    assert summary["sharpness"]["max"] == 80.0

def test_error_handling(session_manager):
    """Test error handling"""
    # Test invalid image hash
    with pytest.raises(Exception):
        session_manager.get_successful_parameters(None)
    
    # Test invalid feedback
    with pytest.raises(Exception):
        session_manager.apply_feedback(None, None)

def test_session_cleanup(session_manager, test_image_hash):
    """Test session cleanup"""
    # Record some attempts
    session_manager.record_enhancement_attempt(
        input_image_hash=test_image_hash,
        models_used=["Super Resolution"],
        parameters={"super_resolution": {"scale_factor": 2.0}},
        quality_metrics={"resolution": "5120x2880"},
        success=True
    )
    
    # Verify session file exists
    session_file = Path("sessions") / f"session_{session_manager.session_id}.json"
    assert session_file.exists()
    
    # Cleanup happens in fixture
