"""Test suite for AI enhancement quality management"""

import pytest
import numpy as np
from PIL import Image
import cv2
from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def test_image():
    """Create test image"""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add color gradient
    x, y = np.meshgrid(np.linspace(0, 255, 512), np.linspace(0, 255, 512))
    img[:, :, 0] = x.astype(np.uint8)  # Red channel
    img[:, :, 1] = y.astype(np.uint8)  # Green channel
    
    # Add patterns for detail enhancement testing
    img[128:384, 128:384, 2] = 255  # Blue square
    img[192:320, 192:320] = [255, 255, 255]  # White square
    
    return Image.fromarray(img)

@pytest.fixture
def test_5k_image():
    """Create 5K test image"""
    img = np.zeros((2880, 5120, 3), dtype=np.uint8)
    img[1000:2000, 2000:3000] = [255, 255, 255]  # Add some content
    return Image.fromarray(img)

@pytest.fixture
def quality_manager():
    """Create QualityManager instance"""
    return QualityManager()

class TestEnhancementQuality:
    def test_quality_metrics_calculation(self, quality_manager, test_image):
        """Test quality metrics calculation"""
        metrics = quality_manager.calculate_metrics(np.array(test_image))
        
        # Verify basic metrics
        assert isinstance(metrics, dict)
        assert 'sharpness' in metrics
        assert 'color_accuracy' in metrics
        assert 'detail_preservation' in metrics
        assert 'noise_level' in metrics
        
        # Verify metric ranges
        for key in ['sharpness', 'color_accuracy', 'detail_preservation', 'noise_level']:
            assert 0 <= metrics[key] <= 1

    def test_quality_parameter_adjustment(self, quality_manager, test_image):
        """Test quality parameter adjustment"""
        # Initial metrics
        initial_metrics = quality_manager.calculate_metrics(np.array(test_image))
        
        # Update parameters
        quality_manager.update_parameters({
            'sharpness': 0.8,
            'color_boost': 0.7,
            'detail_level': 0.6
        })
        
        # Calculate new metrics
        new_metrics = quality_manager.calculate_metrics(np.array(test_image))
        
        # Verify metrics changed
        assert new_metrics != initial_metrics

    def test_5k_quality_verification(self, quality_manager, test_5k_image):
        """Test 5K quality verification"""
        metrics = quality_manager.calculate_metrics(np.array(test_5k_image))
        passed, issues = quality_manager.verify_5k_quality(metrics)
        
        # Basic verification
        assert isinstance(passed, bool)
        assert isinstance(issues, dict)
        assert 'resolution_maintained' in metrics

    def test_enhancement_suggestions(self, quality_manager, test_image):
        """Test enhancement suggestions"""
        metrics = quality_manager.calculate_metrics(np.array(test_image))
        suggestions = quality_manager.get_enhancement_suggestions(metrics)
        
        assert isinstance(suggestions, dict)
        for key in ['sharpness', 'color', 'detail', 'noise']:
            if key in suggestions:
                assert isinstance(suggestions[key], str)

    def test_feedback_integration(self, quality_manager, test_image):
        """Test feedback integration"""
        feedback = [{
            'sharpness_satisfaction': 0.8,
            'color_satisfaction': 0.7,
            'detail_satisfaction': 0.6
        }]
        
        # Initial metrics
        initial_metrics = quality_manager.calculate_metrics(np.array(test_image))
        
        # Apply feedback
        quality_manager.adapt_to_feedback(feedback)
        
        # New metrics
        new_metrics = quality_manager.calculate_metrics(np.array(test_image))
        
        # Verify adaptation occurred
        assert quality_manager.current_parameters != {}

    def test_metric_consistency(self, quality_manager, test_image):
        """Test metric calculation consistency"""
        metrics1 = quality_manager.calculate_metrics(np.array(test_image))
        metrics2 = quality_manager.calculate_metrics(np.array(test_image))
        
        # Verify consistent results
        for key in metrics1:
            if isinstance(metrics1[key], (int, float)):
                assert abs(metrics1[key] - metrics2[key]) < 0.001

    def test_enhancement_history(self, quality_manager, test_image):
        """Test enhancement history tracking"""
        # Record multiple measurements
        for _ in range(3):
            metrics = quality_manager.calculate_metrics(np.array(test_image))
            quality_manager.enhancement_history.append(metrics)
        
        assert len(quality_manager.enhancement_history) > 0
        assert isinstance(quality_manager.enhancement_history[0], dict)
