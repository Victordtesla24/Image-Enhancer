"""Tests for AI-powered image enhancers with quality feedback"""

import pytest
import numpy as np
import torch
from PIL import Image
from src.utils.enhancers.super_resolution import SuperResolutionEnhancer
from src.utils.enhancers.detail_enhancement import DetailEnhancer
from src.utils.enhancers.color_enhancement import ColorEnhancer
from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def quality_manager():
    """Create QualityManager instance"""
    return QualityManager()

@pytest.fixture
def test_image():
    """Create a test image"""
    return np.random.rand(100, 100, 3).astype(np.float32)

@pytest.fixture
def enhancers():
    """Create instances of all enhancers"""
    return {
        'super_res': SuperResolutionEnhancer(),
        'detail': DetailEnhancer(),
        'color': ColorEnhancer()
    }

class TestImageEnhancement:
    def test_quality_improvement(self, enhancers, test_image, quality_manager):
        """Test quality improvement through enhancement pipeline"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get initial quality metrics
        initial_metrics = quality_manager.calculate_metrics(test_image)
        enhanced = test_image

        # Apply each enhancement in sequence
        for enhancer in enhancers.values():
            enhanced = enhancer.enhance(enhanced)

        # Get final quality metrics
        final_metrics = quality_manager.calculate_metrics(enhanced)

        # Verify quality improvements
        assert final_metrics['sharpness'] > initial_metrics['sharpness']
        assert final_metrics['color_accuracy'] > initial_metrics['color_accuracy']
        assert final_metrics['noise_level'] < initial_metrics['noise_level']
        assert final_metrics['detail_preservation'] > initial_metrics['detail_preservation']

    def test_recursive_enhancement(self, enhancers, test_image, quality_manager):
        """Test recursive enhancement with quality feedback"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        enhanced = test_image
        metrics_history = []

        # Perform multiple enhancement iterations
        for _ in range(3):
            # Apply enhancements
            for enhancer in enhancers.values():
                enhanced = enhancer.enhance(enhanced)

            # Calculate and store metrics
            metrics = quality_manager.calculate_metrics(enhanced)
            metrics_history.append(metrics)

        # Verify progressive improvement
        for i in range(1, len(metrics_history)):
            assert metrics_history[i]['sharpness'] >= metrics_history[i-1]['sharpness']
            assert metrics_history[i]['color_accuracy'] >= metrics_history[i-1]['color_accuracy']
            assert metrics_history[i]['detail_preservation'] >= metrics_history[i-1]['detail_preservation']

    def test_quality_parameter_adjustment(self, enhancers, test_image, quality_manager):
        """Test enhancement response to quality parameter adjustments"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test different quality configurations
        quality_configs = [
            {'sharpness': 0.8, 'color_boost': 0.7, 'detail_level': 0.6},
            {'sharpness': 0.6, 'color_boost': 0.8, 'detail_level': 0.7},
            {'sharpness': 0.7, 'color_boost': 0.6, 'detail_level': 0.8}
        ]

        for config in quality_configs:
            # Update enhancer parameters
            for enhancer in enhancers.values():
                enhancer.update_parameters(config)

            # Apply enhancements
            enhanced = test_image
            for enhancer in enhancers.values():
                enhanced = enhancer.enhance(enhanced)

            # Verify metrics align with configuration
            metrics = quality_manager.calculate_metrics(enhanced)
            assert abs(metrics['sharpness'] - config['sharpness']) < 0.2
            assert abs(metrics['color_accuracy'] - config['color_boost']) < 0.2
            assert abs(metrics['detail_preservation'] - config['detail_level']) < 0.2

    def test_user_feedback_integration(self, enhancers, test_image, quality_manager):
        """Test enhancement adaptation based on user feedback"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        feedback_history = []
        enhanced = test_image

        # Simulate multiple rounds of feedback
        for _ in range(3):
            # Apply current enhancement settings
            for enhancer in enhancers.values():
                enhanced = enhancer.enhance(enhanced)

            # Simulate user feedback
            user_feedback = {
                'sharpness_satisfaction': 0.7,
                'color_satisfaction': 0.8,
                'detail_satisfaction': 0.6
            }
            feedback_history.append(user_feedback)

            # Update enhancers based on feedback
            for enhancer in enhancers.values():
                enhancer.adapt_to_feedback(feedback_history)

        # Verify final enhancement quality
        final_metrics = quality_manager.calculate_metrics(enhanced)
        assert final_metrics['sharpness'] > 0.7
        assert final_metrics['color_accuracy'] > 0.7
        assert final_metrics['detail_preservation'] > 0.6

    def test_5k_image_enhancement(self, enhancers, quality_manager):
        """Test enhancement quality for 5K resolution images"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create a 5K test image (5120x2880)
        test_5k = np.random.rand(2880, 5120, 3).astype(np.float32)
        enhanced = test_5k

        # Apply enhancements with 5K-specific parameters
        for enhancer in enhancers.values():
            enhancer.update_parameters({
                'resolution_target': '5k',
                'quality_preset': 'ultra',
                'detail_preservation': 0.9
            })
            enhanced = enhancer.enhance(enhanced)

        # Verify 5K-specific quality metrics
        metrics = quality_manager.calculate_metrics(enhanced)
        assert metrics['resolution_maintained']
        assert metrics['detail_preservation'] > 0.8
        assert metrics['sharpness'] > 0.8
        assert metrics['color_accuracy'] > 0.8

    def test_real_time_quality_adjustment(self, enhancers, test_image, quality_manager):
        """Test real-time quality adjustment capabilities"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        enhanced = test_image
        
        # Test real-time parameter adjustments
        adjustments = [
            {'sharpness': '+0.1', 'color': '+0.2', 'detail': '-0.1'},
            {'sharpness': '-0.1', 'color': '+0.1', 'detail': '+0.2'},
            {'sharpness': '+0.2', 'color': '-0.1', 'detail': '+0.1'}
        ]

        for adjustment in adjustments:
            # Apply current enhancement
            for enhancer in enhancers.values():
                enhancer.adjust_parameters(adjustment)
                enhanced = enhancer.enhance(enhanced)

            # Verify immediate effect of adjustments
            metrics = quality_manager.calculate_metrics(enhanced)
            assert metrics['enhancement_responsiveness'] > 0.7
            assert metrics['adjustment_accuracy'] > 0.8

    def test_enhancement_consistency(self, enhancers, test_image, quality_manager):
        """Test consistency of enhancement quality across multiple runs"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        results = []
        
        # Perform multiple enhancement runs
        for _ in range(3):
            enhanced = test_image
            for enhancer in enhancers.values():
                enhanced = enhancer.enhance(enhanced)
            results.append(quality_manager.calculate_metrics(enhanced))

        # Verify consistency across runs
        for i in range(1, len(results)):
            assert abs(results[i]['sharpness'] - results[0]['sharpness']) < 0.1
            assert abs(results[i]['color_accuracy'] - results[0]['color_accuracy']) < 0.1
            assert abs(results[i]['detail_preservation'] - results[0]['detail_preservation']) < 0.1
