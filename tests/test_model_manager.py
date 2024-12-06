"""Test suite for model management system"""

import os
import pytest
import json
from pathlib import Path
import numpy as np
from src.utils.model_management.model_manager import (
    ModelManager,
    ModelState,
    EnhancementHistory
)

@pytest.fixture
def model_manager():
    """Create ModelManager instance with test session ID"""
    manager = ModelManager("test_session")
    yield manager
    # Cleanup test files
    history_file = Path("models/history") / "history_test_session.json"
    if history_file.exists():
        history_file.unlink()
    if Path("models/history").exists():
        Path("models/history").rmdir()
    if Path("models").exists():
        Path("models").rmdir()

def test_model_initialization(model_manager):
    """Test model manager initialization"""
    assert model_manager.session_id == "test_session"
    assert isinstance(model_manager.models_state, dict)
    assert "super_resolution" in model_manager.models_state
    assert "color_enhancement" in model_manager.models_state
    assert "detail_enhancement" in model_manager.models_state

def test_model_parameters(model_manager):
    """Test model parameter management"""
    # Get initial parameters
    params = model_manager.get_model_parameters("super_resolution")
    assert isinstance(params, dict)
    assert "scale_factor" in params
    assert "denoise_strength" in params
    
    # Update parameters
    new_params = {"scale_factor": 3.0}
    model_manager.update_model_parameters("super_resolution", new_params)
    
    # Verify update
    updated_params = model_manager.get_model_parameters("super_resolution")
    assert updated_params["scale_factor"] == 3.0

def test_enhancement_history_recording(model_manager):
    """Test recording enhancement history"""
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.0},
        quality_metrics={"psnr": 30.0},
        success=True,
        feedback="Good enhancement"
    )
    
    state = model_manager.models_state["super_resolution"]
    assert len(state.enhancement_history) == 1
    history = state.enhancement_history[0]
    assert history.model_name == "super_resolution"
    assert history.parameters["scale_factor"] == 2.0
    assert history.quality_metrics["psnr"] == 30.0
    assert history.success is True
    assert history.feedback == "Good enhancement"

def test_performance_metrics_update(model_manager):
    """Test performance metrics updating"""
    # Record multiple attempts
    for psnr in [28.0, 30.0, 32.0]:
        model_manager.record_enhancement_attempt(
            model_name="super_resolution",
            parameters={"scale_factor": 2.0},
            quality_metrics={"psnr": psnr},
            success=True
        )
    
    metrics = model_manager.models_state["super_resolution"].performance_metrics
    assert metrics["success_rate"] == 1.0
    assert metrics["psnr"] == 30.0  # Average of last 10 attempts

def test_parameter_adaptation(model_manager):
    """Test parameter adaptation based on feedback"""
    # Record successful attempts
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.0},
        quality_metrics={"psnr": 30.0},
        success=True
    )
    
    # Apply feedback
    feedback = {"scale_factor": 1}  # Increase scale factor
    model_manager.adapt_parameters("super_resolution", feedback)
    
    # Verify adaptation
    params = model_manager.get_model_parameters("super_resolution")
    assert params["scale_factor"] > 2.0

def test_history_persistence(model_manager):
    """Test history persistence"""
    # Record attempt
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.0},
        quality_metrics={"psnr": 30.0},
        success=True
    )
    
    # Create new manager with same session
    new_manager = ModelManager("test_session")
    new_manager.load_history("test_session")
    
    state = new_manager.models_state["super_resolution"]
    assert len(state.enhancement_history) == 1
    history = state.enhancement_history[0]
    assert history.parameters["scale_factor"] == 2.0

def test_enhancement_suggestions(model_manager):
    """Test enhancement suggestions generation"""
    # Record successful attempts with different parameters
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.0},
        quality_metrics={"psnr": 30.0},
        success=True
    )
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.5},
        quality_metrics={"psnr": 32.0},
        success=True
    )
    
    suggestions = model_manager.get_enhancement_suggestions("super_resolution")
    assert isinstance(suggestions, dict)
    assert "scale_factor" in suggestions
    assert 2.0 <= suggestions["scale_factor"] <= 2.5

def test_learning_rate_adjustment(model_manager):
    """Test learning rate adjustment"""
    initial_lr = model_manager.models_state["super_resolution"].learning_rate
    
    # Record multiple successful attempts
    for _ in range(5):
        model_manager.record_enhancement_attempt(
            model_name="super_resolution",
            parameters={"scale_factor": 2.0},
            quality_metrics={"psnr": 30.0},
            success=True
        )
    
    # Apply feedback multiple times
    feedback = {"scale_factor": 1}
    for _ in range(3):
        model_manager.adapt_parameters("super_resolution", feedback)
    
    current_lr = model_manager.models_state["super_resolution"].learning_rate
    assert current_lr != initial_lr

def test_model_state_management(model_manager):
    """Test model state management"""
    # Initialize new model state
    state = ModelState(
        name="test_model",
        parameters={"param1": 1.0},
        performance_metrics={"metric1": 0.0},
        enhancement_history=[]
    )
    
    # Add to manager
    model_manager.models_state["test_model"] = state
    
    # Verify state
    assert "test_model" in model_manager.models_state
    assert model_manager.models_state["test_model"].parameters["param1"] == 1.0

def test_error_handling(model_manager):
    """Test error handling"""
    # Test invalid model name
    with pytest.raises(KeyError):
        model_manager.get_model_parameters("invalid_model")
    
    # Test invalid parameters
    with pytest.raises(Exception):
        model_manager.update_model_parameters("super_resolution", None)

def test_performance_tracking(model_manager):
    """Test performance tracking"""
    # Record attempts with varying success
    successes = [True, True, False, True]
    for success in successes:
        model_manager.record_enhancement_attempt(
            model_name="super_resolution",
            parameters={"scale_factor": 2.0},
            quality_metrics={"psnr": 30.0},
            success=success
        )
    
    metrics = model_manager.models_state["super_resolution"].performance_metrics
    assert metrics["success_rate"] == 0.75  # 3/4 successful attempts

def test_model_adaptation_limits(model_manager):
    """Test model parameter adaptation limits"""
    # Try to adapt parameters beyond reasonable limits
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 4.5},  # Already high
        quality_metrics={"psnr": 30.0},
        success=True
    )
    
    feedback = {"scale_factor": 1}  # Try to increase further
    model_manager.adapt_parameters("super_resolution", feedback)
    
    params = model_manager.get_model_parameters("super_resolution")
    assert params["scale_factor"] <= 5.0  # Should be capped

def test_history_cleanup(model_manager):
    """Test history cleanup"""
    # Record some attempts
    model_manager.record_enhancement_attempt(
        model_name="super_resolution",
        parameters={"scale_factor": 2.0},
        quality_metrics={"psnr": 30.0},
        success=True
    )
    
    # Verify history file exists
    history_file = Path("models/history") / f"history_{model_manager.session_id}.json"
    assert history_file.exists()
    
    # Cleanup happens in fixture
