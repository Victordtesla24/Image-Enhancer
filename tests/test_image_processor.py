"""Test suite for AI-powered image enhancement processor"""

import pytest
import numpy as np
from PIL import Image
import cv2
import torch
from src.utils.image_processor import ImageEnhancer
from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def test_image():
    """Create test image with various features for enhancement testing"""
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
    
    # Add test patterns
    img[1000:2000, 2000:3000] = [255, 255, 255]  # White rectangle
    img[500:1500, 1000:2000, 0] = 255  # Red rectangle
    img[1500:2500, 3000:4000, 1] = 255  # Green rectangle
    
    return Image.fromarray(img)

@pytest.fixture
def processor():
    """Create ImageEnhancer instance"""
    return ImageEnhancer()

@pytest.fixture
def quality_manager():
    """Create QualityManager instance"""
    return QualityManager()

class TestImageEnhancement:
    def test_enhancement_quality(self, processor, quality_manager, test_image):
        """Test image enhancement quality"""
        # Get initial quality metrics
        initial_metrics = quality_manager.calculate_metrics(np.array(test_image))

        # Enhance image
        enhanced_image = processor.enhance(np.array(test_image))

        # Get enhanced quality metrics
        enhanced_metrics = quality_manager.calculate_metrics(enhanced_image)

        # Verify quality improvements
        assert enhanced_metrics['sharpness'] >= initial_metrics['sharpness']
        assert enhanced_metrics['color_accuracy'] >= initial_metrics['color_accuracy']
        assert enhanced_metrics['detail_preservation'] >= initial_metrics['detail_preservation']
        assert enhanced_metrics['noise_level'] <= initial_metrics['noise_level']

    def test_recursive_enhancement(self, processor, quality_manager, test_image):
        """Test recursive enhancement capabilities"""
        enhanced = np.array(test_image)
        metrics_history = []

        # Perform multiple enhancement iterations
        for iteration in range(3):
            enhanced = processor.enhance(enhanced)
            metrics = quality_manager.calculate_metrics(enhanced)
            metrics_history.append(metrics)

            # Update enhancement parameters based on metrics
            processor.update_parameters({
                'sharpness': min(1.0, metrics['sharpness'] + 0.1),
                'color_boost': min(1.0, metrics['color_accuracy'] + 0.1),
                'detail_level': min(1.0, metrics['detail_preservation'] + 0.1)
            })

        # Verify progressive improvement or maintenance of high quality
        for i in range(1, len(metrics_history)):
            assert metrics_history[i]['sharpness'] >= metrics_history[i-1]['sharpness'] or metrics_history[i]['sharpness'] >= 0.9
            assert metrics_history[i]['color_accuracy'] >= metrics_history[i-1]['color_accuracy'] or metrics_history[i]['color_accuracy'] >= 0.9
            assert metrics_history[i]['detail_preservation'] >= metrics_history[i-1]['detail_preservation'] or metrics_history[i]['detail_preservation'] >= 0.9

    def test_5k_enhancement(self, processor, quality_manager, test_5k_image):
        """Test 5K image enhancement capabilities"""
        # Set 5K optimization parameters
        processor.update_parameters({
            'resolution_target': '5k',
            'quality_preset': 'ultra',
            'detail_preservation': 0.9,
            'sharpness': 0.9,
            'color_boost': 0.9
        })

        # Get initial metrics
        initial_metrics = quality_manager.calculate_metrics(np.array(test_5k_image))

        # Enhance 5K image
        enhanced = processor.enhance(np.array(test_5k_image))

        # Get enhanced metrics
        enhanced_metrics = quality_manager.calculate_metrics(enhanced)

        # Verify quality improvements
        assert enhanced_metrics['sharpness'] >= initial_metrics['sharpness']
        assert enhanced_metrics['detail_preservation'] >= initial_metrics['detail_preservation']
        assert enhanced_metrics['color_accuracy'] >= initial_metrics['color_accuracy']
        assert enhanced_metrics['resolution_maintained']

    def test_quality_feedback_integration(self, processor, quality_manager, test_image):
        """Test enhancement adaptation based on quality feedback"""
        enhanced = np.array(test_image)
        feedback_history = []

        # Perform multiple enhancement iterations with feedback
        for iteration in range(3):
            enhanced = processor.enhance(enhanced)
            metrics = quality_manager.calculate_metrics(enhanced)

            # Simulate user feedback
            feedback = {
                'sharpness_satisfaction': 0.7 + iteration * 0.1,
                'color_satisfaction': 0.8,
                'detail_satisfaction': 0.7 + iteration * 0.1
            }
            feedback_history.append(feedback)

            # Apply feedback
            processor.adapt_to_feedback(feedback_history)

            # Verify metrics improve or maintain high quality
            if iteration > 0:
                assert metrics['sharpness'] >= previous_metrics['sharpness'] or metrics['sharpness'] >= 0.9
                assert metrics['color_accuracy'] >= previous_metrics['color_accuracy'] or metrics['color_accuracy'] >= 0.9
                assert metrics['detail_preservation'] >= previous_metrics['detail_preservation'] or metrics['detail_preservation'] >= 0.9

            previous_metrics = metrics

    def test_enhancement_consistency(self, processor, quality_manager, test_image):
        """Test consistency of enhancement results"""
        results = []
        
        # Perform multiple enhancements with same parameters
        for _ in range(3):
            enhanced = processor.enhance(np.array(test_image))
            metrics = quality_manager.calculate_metrics(enhanced)
            results.append(metrics)
        
        # Verify consistency
        for i in range(1, len(results)):
            assert abs(results[i]['sharpness'] - results[0]['sharpness']) < 0.1
            assert abs(results[i]['color_accuracy'] - results[0]['color_accuracy']) < 0.1
            assert abs(results[i]['detail_preservation'] - results[0]['detail_preservation']) < 0.1

    def test_enhancement_suggestions(self, processor, quality_manager, test_image):
        """Test enhancement suggestion system"""
        enhanced = processor.enhance(np.array(test_image))
        metrics = quality_manager.calculate_metrics(enhanced)
        suggestions = quality_manager.get_enhancement_suggestions(metrics)
        
        assert isinstance(suggestions, dict)
        for key in ['sharpness', 'color', 'detail']:
            if key in suggestions:
                assert isinstance(suggestions[key], str)
                assert len(suggestions[key]) > 0
