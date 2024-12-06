"""Test suite for AI enhancement model management"""

import pytest
import numpy as np
import torch
from pathlib import Path
import shutil
from src.utils.model_management.model_manager import ModelManager
from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def test_image():
    """Create test image"""
    return np.random.rand(512, 512, 3).astype(np.float32)

@pytest.fixture
def test_5k_image():
    """Create 5K test image"""
    return np.random.rand(2880, 5120, 3).astype(np.float32)

@pytest.fixture
def model_manager():
    """Create ModelManager instance"""
    manager = ModelManager()
    yield manager
    # Cleanup is handled by the manager's __del__ method
    manager.cleanup()

@pytest.fixture
def quality_manager():
    """Create QualityManager instance"""
    return QualityManager()

class TestEnhancementModels:
    def test_model_initialization(self, model_manager):
        """Test enhancement model initialization"""
        # Verify models are initialized
        assert 'super_resolution' in model_manager.models
        assert 'detail' in model_manager.models
        assert 'color' in model_manager.models
        
        # Verify model states
        assert 'super_resolution' in model_manager.models_state
        assert 'detail' in model_manager.models_state
        assert 'color' in model_manager.models_state

    def test_enhancement_quality(self, model_manager, quality_manager, test_image):
        """Test enhancement quality for each model"""
        for model_name in model_manager.models:
            # Get initial quality metrics
            initial_metrics = quality_manager.calculate_metrics(test_image)
            
            # Apply enhancement
            enhanced = model_manager.enhance(model_name, test_image)
            
            # Get enhanced quality metrics
            enhanced_metrics = quality_manager.calculate_metrics(enhanced)
            
            # Verify quality improvement
            assert enhanced_metrics['sharpness'] >= initial_metrics['sharpness']
            assert enhanced_metrics['color_accuracy'] >= initial_metrics['color_accuracy']
            assert enhanced_metrics['detail_preservation'] >= initial_metrics['detail_preservation']

    def test_parameter_adaptation(self, model_manager, quality_manager, test_image):
        """Test model parameter adaptation based on quality feedback"""
        feedback_history = [
            {
                'sharpness_satisfaction': 0.8,
                'color_satisfaction': 0.7,
                'detail_satisfaction': 0.6
            }
        ]
        
        # Record initial parameters
        initial_params = {
            name: model_manager.get_model_parameters(name)
            for name in model_manager.models
        }
        
        # Apply feedback
        model_manager.adapt_to_feedback(feedback_history)
        
        # Verify parameters were updated
        for name in model_manager.models:
            new_params = model_manager.get_model_parameters(name)
            assert new_params != initial_params[name]

    def test_enhancement_history(self, model_manager, quality_manager, test_image):
        """Test enhancement history tracking"""
        model_name = 'super_resolution'
        
        # Record multiple enhancement attempts
        for _ in range(3):
            enhanced = model_manager.enhance(model_name, test_image)
            metrics = quality_manager.calculate_metrics(enhanced)
            
            model_manager.record_enhancement_attempt(
                model_name=model_name,
                parameters=model_manager.get_model_parameters(model_name),
                quality_metrics=metrics,
                success=True
            )
        
        # Verify history
        state = model_manager.models_state[model_name]
        assert len(state.enhancement_history) == 3
        assert state.performance_metrics['success_rate'] == 1.0

    def test_real_time_adjustment(self, model_manager, quality_manager, test_image):
        """Test real-time parameter adjustment"""
        model_name = 'super_resolution'
        
        # Test different parameter adjustments
        adjustments = [
            {'sharpness': 0.8, 'detail_level': 0.7},
            {'sharpness': 0.6, 'detail_level': 0.9},
            {'sharpness': 0.7, 'detail_level': 0.8}
        ]
        
        for params in adjustments:
            # Update parameters
            model_manager.update_parameters(model_name, params)
            
            # Verify parameters were updated
            current_params = model_manager.get_model_parameters(model_name)
            for key, value in params.items():
                assert abs(current_params[key] - value) < 0.01

    def test_enhancement_consistency(self, model_manager, quality_manager, test_image):
        """Test consistency of enhancement results"""
        model_name = 'super_resolution'
        results = []
        
        # Perform multiple enhancements
        for _ in range(3):
            enhanced = model_manager.enhance(model_name, test_image)
            metrics = quality_manager.calculate_metrics(enhanced)
            results.append(metrics)
        
        # Verify consistency
        for i in range(1, len(results)):
            assert abs(results[i]['sharpness'] - results[0]['sharpness']) < 0.1
            assert abs(results[i]['color_accuracy'] - results[0]['color_accuracy']) < 0.1
            assert abs(results[i]['detail_preservation'] - results[0]['detail_preservation']) < 0.1

    def test_history_persistence(self, model_manager, quality_manager, test_image):
        """Test history saving and loading"""
        model_name = 'super_resolution'
        
        # Record enhancement attempt
        enhanced = model_manager.enhance(model_name, test_image)
        metrics = quality_manager.calculate_metrics(enhanced)
        
        model_manager.record_enhancement_attempt(
            model_name=model_name,
            parameters=model_manager.get_model_parameters(model_name),
            quality_metrics=metrics,
            success=True
        )
        
        # Create new manager and load history
        new_manager = ModelManager(model_manager.session_id)
        success = new_manager.load_history(model_manager.session_id)
        
        assert success
        assert len(new_manager.models_state[model_name].enhancement_history) == 1
