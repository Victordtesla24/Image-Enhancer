"""Test suite for quality management system"""

import os
import pytest
import numpy as np
from PIL import Image
import cv2
from src.utils.quality_management.quality_manager import QualityManager, QualityMetrics

@pytest.fixture
def test_image():
    """Create test image"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # White square in center
    return Image.fromarray(img)

@pytest.fixture
def quality_manager():
    """Create QualityManager instance"""
    config = {
        'resolution': {'width': 5120, 'height': 2880},
        'quality': {
            'dpi': 300,
            'min_sharpness': 70,
            'max_noise_level': 120,
            'min_file_size_mb': 1.5
        },
        'color': {
            'bit_depth': 24,
            'dynamic_range': {'min': 220, 'max': 255}
        }
    }
    return QualityManager(config)

def test_quality_metrics_computation(quality_manager, test_image):
    """Test computation of quality metrics"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    
    assert isinstance(metrics, QualityMetrics)
    assert metrics.resolution == (100, 100)
    assert metrics.color_depth == 24
    assert metrics.dynamic_range > 0
    assert metrics.sharpness >= 0
    assert metrics.noise_level >= 0

def test_quality_validation(quality_manager, test_image):
    """Test quality validation"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    passed, results = quality_manager.validate_quality(metrics)
    
    assert isinstance(passed, bool)
    assert isinstance(results, dict)
    assert "resolution" in results
    assert "sharpness" in results
    assert "noise_level" in results
    assert "color_depth" in results

def test_sharpness_computation(quality_manager, test_image):
    """Test sharpness computation"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    assert metrics.sharpness >= 0
    assert isinstance(metrics.sharpness, float)

def test_noise_level_computation(quality_manager, test_image):
    """Test noise level computation"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    assert metrics.noise_level >= 0
    assert isinstance(metrics.noise_level, float)

def test_dynamic_range_computation(quality_manager, test_image):
    """Test dynamic range computation"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    assert metrics.dynamic_range > 0
    assert isinstance(metrics.dynamic_range, int)
    assert metrics.dynamic_range <= 255

def test_comparison_metrics(quality_manager, test_image):
    """Test comparison metrics computation"""
    # Create slightly modified image for comparison
    modified = test_image.copy()
    modified_array = np.array(modified)
    modified_array = cv2.GaussianBlur(modified_array, (3, 3), 0)
    modified = Image.fromarray(modified_array)
    
    metrics = quality_manager.compute_quality_metrics(modified, original=test_image)
    assert metrics.psnr is not None
    assert metrics.ssim is not None
    assert metrics.color_accuracy is not None

def test_quality_validation_thresholds(quality_manager, test_image):
    """Test quality validation thresholds"""
    metrics = quality_manager.compute_quality_metrics(test_image)
    passed, results = quality_manager.validate_quality(metrics)
    
    # Resolution check
    assert results["resolution"]["passed"] == (
        metrics.resolution[0] >= quality_manager.config["resolution"]["width"] and
        metrics.resolution[1] >= quality_manager.config["resolution"]["height"]
    )
    
    # Color depth check
    assert results["color_depth"]["passed"] == (
        metrics.color_depth >= quality_manager.config["color"]["bit_depth"]
    )
    
    # Dynamic range check
    assert results["dynamic_range"]["passed"] == (
        metrics.dynamic_range >= quality_manager.config["color"]["dynamic_range"]["min"]
    )

def test_error_handling(quality_manager):
    """Test error handling for invalid inputs"""
    with pytest.raises(Exception):
        quality_manager.compute_quality_metrics(None)
    
    with pytest.raises(Exception):
        quality_manager.validate_quality(None)

def test_high_quality_image(quality_manager):
    """Test with high quality image"""
    # Create high quality test image
    img = np.zeros((5120, 2880, 3), dtype=np.uint8)
    img[1000:4000, 500:2000] = 255
    high_quality = Image.fromarray(img)
    
    metrics = quality_manager.compute_quality_metrics(high_quality)
    passed, results = quality_manager.validate_quality(metrics)
    
    assert passed
    assert results["resolution"]["passed"]
    assert results["color_depth"]["passed"]
    assert results["dynamic_range"]["passed"]

def test_low_quality_image(quality_manager):
    """Test with low quality image"""
    # Create low quality test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 128  # Gray square with low contrast
    low_quality = Image.fromarray(img)
    
    metrics = quality_manager.compute_quality_metrics(low_quality)
    passed, results = quality_manager.validate_quality(metrics)
    
    assert not passed
    assert not results["resolution"]["passed"]
    assert results["dynamic_range"]["passed"] == (
        metrics.dynamic_range >= quality_manager.config["color"]["dynamic_range"]["min"]
    )

def test_quality_metrics_consistency(quality_manager, test_image):
    """Test consistency of quality metrics"""
    metrics1 = quality_manager.compute_quality_metrics(test_image)
    metrics2 = quality_manager.compute_quality_metrics(test_image)
    
    assert metrics1.resolution == metrics2.resolution
    assert metrics1.color_depth == metrics2.color_depth
    assert abs(metrics1.sharpness - metrics2.sharpness) < 1e-6
    assert abs(metrics1.noise_level - metrics2.noise_level) < 1e-6
    assert metrics1.dynamic_range == metrics2.dynamic_range
