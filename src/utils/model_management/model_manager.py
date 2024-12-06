"""AI Model Management System"""

import logging
import torch
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

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
        self.models_state: Dict[str, ModelState] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_dir = Path("models/history")
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model states
        self._initialize_models()

    def _initialize_models(self):
        """Initialize model states with default parameters"""
        default_models = {
            "super_resolution": {
                "parameters": {
                    "scale_factor": 4.0,
                    "denoise_strength": 0.5,
                    "detail_preservation": 0.8,
                },
                "performance_metrics": {"psnr": 0.0, "ssim": 0.0, "success_rate": 0.0},
            },
            "color_enhancement": {
                "parameters": {
                    "saturation": 1.2,
                    "contrast": 1.15,
                    "brightness": 1.1,
                    "color_balance": 1.0,
                },
                "performance_metrics": {
                    "color_accuracy": 0.0,
                    "contrast_score": 0.0,
                    "success_rate": 0.0,
                },
            },
            "detail_enhancement": {
                "parameters": {
                    "sharpness": 1.3,
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
        }

        for model_name, config in default_models.items():
            self.models_state[model_name] = ModelState(
                name=model_name,
                parameters=config["parameters"],
                performance_metrics=config["performance_metrics"],
                enhancement_history=[],
            )

    def get_model_parameters(self, model_name: str) -> Dict:
        """Get current parameters for a model"""
        return self.models_state[model_name].parameters.copy()

    def update_model_parameters(self, model_name: str, parameters: Dict):
        """Update model parameters"""
        self.models_state[model_name].parameters.update(parameters)

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
                    self.models_state[model_name].performance_metrics[metric_name] = (
                        np.mean(metric_values)
                    )

    def adapt_parameters(self, model_name: str, feedback: Dict):
        """Adapt model parameters based on feedback"""
        model_state = self.models_state[model_name]
        history = model_state.enhancement_history

        if not history:
            return

        # Get recent successful attempts
        successful_attempts = [h for h in history[-5:] if h.success]
        if not successful_attempts:
            return

        # Calculate parameter adjustments based on feedback
        for param_name, current_value in model_state.parameters.items():
            if param_name in feedback:
                direction = feedback[param_name]  # 1 for increase, -1 for decrease

                # Calculate adjustment based on successful parameters
                successful_values = [
                    h.parameters.get(param_name, current_value)
                    for h in successful_attempts
                ]
                mean_successful = np.mean(successful_values)

                # Adjust parameter with learning rate
                adjustment = direction * model_state.learning_rate * abs(current_value)
                new_value = current_value + adjustment

                # Apply bounds (assuming parameters should stay positive)
                new_value = max(0.1, min(5.0, new_value))

                model_state.parameters[param_name] = new_value

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
                    self.models_state[model_name].performance_metrics = data[
                        "performance_metrics"
                    ]
                    self.models_state[model_name].enhancement_history = [
                        EnhancementHistory(**h) for h in data["history"]
                    ]
            return True
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return False

    def get_enhancement_suggestions(self, model_name: str) -> Dict:
        """Get parameter suggestions based on successful enhancements"""
        history = self.models_state[model_name].enhancement_history
        if not history:
            return self.get_model_parameters(model_name)

        successful_attempts = [h for h in history if h.success]
        if not successful_attempts:
            return self.get_model_parameters(model_name)

        # Calculate average parameters from successful attempts
        suggested_params = {}
        for param_name in self.models_state[model_name].parameters:
            values = [h.parameters.get(param_name, 0) for h in successful_attempts]
            suggested_params[param_name] = np.mean(values)

        return suggested_params
