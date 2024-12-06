"""AI Model Management System"""

import logging
import torch
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from ..models.super_resolution import SuperResolutionModel
from ..models.detail_enhancement import DetailEnhancementModel
from ..models.color_enhancement import ColorEnhancementModel

logger = logging.getLogger(__name__)

@dataclass
class EnhancementHistory:
    """Track enhancement attempts and results"""
    timestamp: str
    model_name: str
    parameters: Dict
    quality_metrics: Dict
    success: bool
    feedback: Optional[str] = None

@dataclass
class ModelState:
    """Track model state and performance"""
    name: str
    parameters: Dict
    performance_metrics: Dict
    enhancement_history: List[EnhancementHistory]
    learning_rate: float = 0.001

class ModelManager:
    """Manages AI models, their states, and learning processes"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.models = {
            'super_resolution': SuperResolutionModel(),
            'detail': DetailEnhancementModel(),
            'color': ColorEnhancementModel()
        }
        
        # Load models
        for model in self.models.values():
            model.load()
        
        # Initialize model states
        self.models_state: Dict[str, ModelState] = {}
        self._initialize_models()
        
        # Setup history directory
        self.history_dir = Path("models/history")
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_models(self):
        """Initialize model states with default parameters"""
        default_models = {
            "super_resolution": {
                "parameters": {
                    "sharpness": 0.7,
                    "detail_level": 0.7,
                    "color_boost": 0.7,
                    "scale_factor": 4.0,
                    "denoise_strength": 0.5,
                    "detail_preservation": 0.8,
                },
                "performance_metrics": {"psnr": 0.0, "ssim": 0.0, "success_rate": 0.0},
            },
            "detail": {
                "parameters": {
                    "sharpness": 0.7,
                    "detail_level": 0.7,
                    "color_boost": 0.7,
                    "noise_reduction": 0.5,
                    "detail_boost": 1.2,
                    "edge_preservation": 0.8,
                },
                "performance_metrics": {
                    "sharpness_score": 0.0,
                    "noise_level": 0.0,
                    "success_rate": 0.0,
                },
            },
            "color": {
                "parameters": {
                    "sharpness": 0.7,
                    "detail_level": 0.7,
                    "color_boost": 0.7,
                    "saturation": 1.2,
                    "contrast": 1.15,
                    "brightness": 1.1,
                },
                "performance_metrics": {
                    "color_accuracy": 0.0,
                    "contrast_score": 0.0,
                    "success_rate": 0.0,
                },
            }
        }

        for model_name, config in default_models.items():
            self.models_state[model_name] = ModelState(
                name=model_name,
                parameters=config["parameters"],
                performance_metrics=config["performance_metrics"],
                enhancement_history=[],
            )
            # Initialize model parameters
            if model_name in self.models:
                self.models[model_name].update_parameters(config["parameters"])

    def get_model_parameters(self, model_name: str) -> Dict:
        """Get current parameters for a model"""
        return self.models_state[model_name].parameters.copy()

    def update_parameters(self, model_name: str, parameters: Dict):
        """Update model parameters"""
        current_params = self.models_state[model_name].parameters
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                current_params[key] = value
            elif isinstance(value, str) and value.startswith(('+', '-')):
                # Handle incremental adjustments
                try:
                    delta = float(value)
                    current_value = current_params.get(key, 0.0)
                    current_params[key] = max(0.0, min(1.0, current_value + delta))
                except ValueError:
                    pass
        
        # Update model parameters
        if model_name in self.models:
            self.models[model_name].update_parameters(current_params)

    def enhance(self, model_name: str, image: torch.Tensor) -> torch.Tensor:
        """Enhance image using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        try:
            return self.models[model_name].enhance(image)
        except Exception as e:
            logger.error(f"Error in {model_name} enhancement: {str(e)}")
            return image

    def record_enhancement_attempt(
        self,
        model_name: str,
        parameters: Dict,
        quality_metrics: Dict,
        success: bool,
        feedback: Optional[str] = None,
    ):
        """Record an enhancement attempt and its results"""
        history = EnhancementHistory(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            parameters=parameters,
            quality_metrics=quality_metrics,
            success=success,
            feedback=feedback,
        )
        self.models_state[model_name].enhancement_history.append(history)
        self._update_performance_metrics(model_name)
        self._save_history()

    def _update_performance_metrics(self, model_name: str):
        """Update model performance metrics based on history"""
        history = self.models_state[model_name].enhancement_history
        if not history:
            return

        # Calculate success rate
        success_rate = sum(1 for h in history if h.success) / len(history)
        self.models_state[model_name].performance_metrics["success_rate"] = success_rate

        # Update model-specific metrics
        recent_metrics = [h.quality_metrics for h in history[-10:]]  # Last 10 attempts
        for metric_name in self.models_state[model_name].performance_metrics:
            if metric_name != "success_rate":
                metric_values = [m.get(metric_name, 0) for m in recent_metrics]
                if metric_values:
                    self.models_state[model_name].performance_metrics[metric_name] = np.mean(metric_values)

    def adapt_to_feedback(self, feedback_history: List[Dict]):
        """Adapt model parameters based on feedback history"""
        if not feedback_history:
            return
            
        for model_name in self.models:
            model_state = self.models_state[model_name]
            
            # Calculate average feedback scores
            avg_feedback = {
                key: np.mean([f[key] for f in feedback_history if key in f])
                for key in ['sharpness_satisfaction', 'color_satisfaction', 'detail_satisfaction']
            }
            
            # Adjust parameters based on feedback
            params = model_state.parameters.copy()
            if 'sharpness_satisfaction' in avg_feedback:
                params['sharpness'] = min(1.0, params.get('sharpness', 0.7) * 
                                        (1 + (avg_feedback['sharpness_satisfaction'] - 0.5) * 0.2))
                
            if 'color_satisfaction' in avg_feedback:
                params['color_boost'] = min(1.0, params.get('color_boost', 0.7) * 
                                          (1 + (avg_feedback['color_satisfaction'] - 0.5) * 0.2))
                
            if 'detail_satisfaction' in avg_feedback:
                params['detail_level'] = min(1.0, params.get('detail_level', 0.7) * 
                                           (1 + (avg_feedback['detail_satisfaction'] - 0.5) * 0.2))
            
            # Update model parameters
            self.update_parameters(model_name, params)

    def _save_history(self):
        """Save enhancement history to disk"""
        history_file = self.history_dir / f"history_{self.session_id}.json"
        history_data = {
            model_name: {
                "parameters": state.parameters,
                "performance_metrics": state.performance_metrics,
                "history": [asdict(h) for h in state.enhancement_history],
            }
            for model_name, state in self.models_state.items()
        }

        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

    def load_history(self, session_id: str):
        """Load enhancement history from disk"""
        history_file = self.history_dir / f"history_{session_id}.json"
        if not history_file.exists():
            return False

        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)

            for model_name, data in history_data.items():
                if model_name in self.models_state:
                    self.models_state[model_name].parameters = data["parameters"]
                    self.models_state[model_name].performance_metrics = data["performance_metrics"]
                    self.models_state[model_name].enhancement_history = [
                        EnhancementHistory(**h) for h in data["history"]
                    ]
                    # Update model parameters
                    if model_name in self.models:
                        self.models[model_name].update_parameters(data["parameters"])
            return True
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return False

    def cleanup(self):
        """Clean up model resources and history"""
        try:
            if self.history_dir.exists():
                shutil.rmtree(self.history_dir)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()
