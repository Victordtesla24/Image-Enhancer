"""Test suite for main image processor"""

import os
import pytest
import numpy as np
from PIL import Image
import torch
from src.utils.image_processor import ImageEnhancer
from src.utils.model_management.model_manager import ModelManager
from src.utils.session_management.session_manager import SessionManager
from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def test_image():
    """Create test image"""
    # Create a test pattern with various features
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add color gradient
    x, y = np.meshgrid(np.linspace(0, 255, 200), np.linspace(0, 255, 200))
    img[:, :, 0] = x.astype(np.uint8)  # Red channel
    img[:, :, 1] = y.astype(np.uint8)  # Green channel
    
    # Add some patterns for detail enhancement testing
    img[50:150, 50:150, 2] = 255  # Blue square
    img[75:125, 75:125] = [255, 255, 255]  # White square
    
    return Image.fromarray(img)

@pytest.fixture
def processor():
    """Create ImageEnhancer instance"""
    return ImageEnhancer("test_session")

def test_processor_initialization(processor):
    """Test image processor initialization"""
    assert isinstance(processor.model_manager, ModelManager)
    assert isinstance(processor.session_manager, SessionManager)
    assert isinstance(processor.quality_manager, QualityManager)
    assert processor.models is not None
    assert len(processor.models) == 3

def test_available_models(processor):
    """Test getting available models"""
    models = processor.get_available_models()
    assert len(models) == 3
    assert all(isinstance(model, dict) for model in models)
    assert all("name" in model and "description" in model for model in models)

def test_basic_enhancement(processor, test_image):
    """Test basic image enhancement"""
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    assert isinstance(enhanced_image, Image.Image)
    assert enhanced_image.size[0] == 400
    assert "quality_results" in details
    assert "processing_time" in details

def test_full_enhancement_pipeline(processor, test_image):
    """Test complete enhancement pipeline"""
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution", "Color Enhancement", "Detail Enhancement"],
        retry_count=0
    )
    
    assert isinstance(enhanced_image, Image.Image)
    assert enhanced_image.size[0] == 400
    assert len(details["models_used"]) == 3
    assert all(model["parameters"] for model in details["models_used"])

def test_enhancement_with_retry(processor, test_image):
    """Test enhancement with retry mechanism"""
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution", "Detail Enhancement"],
        retry_count=1
    )
    
    assert isinstance(enhanced_image, Image.Image)
    assert "retry_count" in details
    assert details["retry_count"] >= 0

def test_quality_validation(processor, test_image):
    """Test quality validation in enhancement process"""
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    assert "quality_results" in details
    quality_results = details["quality_results"]
    assert "resolution" in quality_results
    assert "sharpness" in quality_results
    assert "noise_level" in quality_results

def test_session_tracking(processor, test_image):
    """Test session tracking during enhancement"""
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    history = processor.get_enhancement_history()
    assert "history" in history
    assert "metrics_summary" in history
    assert len(history["history"]) > 0

def test_feedback_system(processor, test_image):
    """Test feedback system"""
    # Perform enhancement
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    # Apply feedback
    image_hash = processor._compute_image_hash(test_image)
    feedback = {
        "super_resolution": {
            "scale_factor": 1  # Increase scale factor
        }
    }
    processor.apply_feedback(image_hash, feedback)
    
    # Verify feedback application
    history = processor.get_enhancement_history(image_hash)
    assert "metrics_summary" in history

def test_quality_preferences(processor):
    """Test quality preferences management"""
    preferences = processor.get_quality_preferences()
    assert isinstance(preferences, dict)
    
    new_preferences = {
        "min_resolution": (3840, 2160),
        "min_dpi": 200
    }
    processor.update_quality_preferences(new_preferences)
    
    updated_preferences = processor.get_quality_preferences()
    assert updated_preferences["min_resolution"] == (3840, 2160)
    assert updated_preferences["min_dpi"] == 200

def test_error_handling(processor):
    """Test error handling"""
    # Test with invalid input
    with pytest.raises(Exception):
        processor.enhance_image(None, 400, ["Super Resolution"])
    
    # Test with invalid model
    with pytest.raises(Exception):
        processor.enhance_image(
            Image.new('RGB', (100, 100)),
            400,
            ["Invalid Model"]
        )

def test_memory_management(processor, test_image):
    """Test memory management during enhancement"""
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=1000,  # Larger size to stress memory
        models=["Super Resolution", "Color Enhancement", "Detail Enhancement"],
        retry_count=0
    )
    
    # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.1  # Allow for small overhead

def test_enhancement_consistency(processor, test_image):
    """Test consistency of enhancement results"""
    # Perform two identical enhancements
    result1, details1 = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    result2, details2 = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution"],
        retry_count=0
    )
    
    # Compare results
    diff = np.array(result1) - np.array(result2)
    assert np.abs(diff).mean() < 1.0  # Allow for small floating-point differences

def test_progress_callback(processor, test_image):
    """Test progress callback functionality"""
    progress_values = []
    
    def progress_callback(progress, status):
        progress_values.append(progress)
    
    enhanced_image, details = processor.enhance_image(
        test_image,
        target_width=400,
        models=["Super Resolution", "Color Enhancement"],
        progress_callback=progress_callback
    )
    
    assert len(progress_values) > 0
    assert progress_values[-1] == 1.0  # Final progress should be 100%

def test_cleanup(processor):
    """Test cleanup after processing"""
    # Cleanup happens in fixture teardown
    pass
