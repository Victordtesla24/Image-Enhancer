"""Session Management System"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from ..model_management.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class QualityPreferences:
    """User's quality preferences"""

    min_resolution: tuple = (5120, 2880)  # 5K
    min_dpi: int = 300
    min_sharpness: float = 70.0
    max_noise_level: float = 120.0
    min_dynamic_range: int = 220
    color_depth: int = 24
    min_file_size_mb: float = 1.5


@dataclass
class EnhancementAttempt:
    """Record of an enhancement attempt"""

    timestamp: str
    input_image_hash: str
    models_used: List[str]
    parameters: Dict
    quality_metrics: Dict
    success: bool
    feedback: Optional[str] = None


class SessionManager:
    """Manages user sessions and enhancement context"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path("sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session state
        self.quality_preferences = QualityPreferences()
        self.enhancement_attempts: List[EnhancementAttempt] = []
        self.current_image_hash: Optional[str] = None
        self.model_manager = ModelManager(self.session_id)

        # Load or create session file
        self._load_or_create_session()

    def _load_or_create_session(self):
        """Load existing session or create new one"""
        session_file = self.session_dir / f"session_{self.session_id}.json"
        if session_file.exists():
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                self.quality_preferences = QualityPreferences(
                    **data.get("quality_preferences", {})
                )
                self.enhancement_attempts = [
                    EnhancementAttempt(**attempt)
                    for attempt in data.get("enhancement_attempts", [])
                ]
                self.current_image_hash = data.get("current_image_hash")
            except Exception as e:
                logger.error(f"Error loading session: {str(e)}")
                self._create_new_session()
        else:
            self._create_new_session()

    def _create_new_session(self):
        """Create new session with default settings"""
        self.quality_preferences = QualityPreferences()
        self.enhancement_attempts = []
        self.current_image_hash = None
        self._save_session()

    def _save_session(self):
        """Save session state to disk"""
        session_file = self.session_dir / f"session_{self.session_id}.json"
        session_data = {
            "session_id": self.session_id,
            "quality_preferences": asdict(self.quality_preferences),
            "enhancement_attempts": [
                asdict(attempt) for attempt in self.enhancement_attempts
            ],
            "current_image_hash": self.current_image_hash,
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def update_quality_preferences(self, preferences: Dict):
        """Update quality preferences"""
        current_prefs = asdict(self.quality_preferences)
        current_prefs.update(preferences)
        self.quality_preferences = QualityPreferences(**current_prefs)
        self._save_session()

    def record_enhancement_attempt(
        self,
        input_image_hash: str,
        models_used: List[str],
        parameters: Dict,
        quality_metrics: Dict,
        success: bool,
        feedback: Optional[str] = None,
    ):
        """Record an enhancement attempt"""
        attempt = EnhancementAttempt(
            timestamp=datetime.now().isoformat(),
            input_image_hash=input_image_hash,
            models_used=models_used,
            parameters=parameters,
            quality_metrics=quality_metrics,
            success=success,
            feedback=feedback,
        )
        self.enhancement_attempts.append(attempt)
        self.current_image_hash = input_image_hash

        # Update model manager with attempt results
        for model_name in models_used:
            self.model_manager.record_enhancement_attempt(
                model_name=model_name,
                parameters=parameters.get(model_name, {}),
                quality_metrics=quality_metrics,
                success=success,
                feedback=feedback,
            )

        self._save_session()

    def get_enhancement_history(
        self, input_image_hash: Optional[str] = None
    ) -> List[EnhancementAttempt]:
        """Get enhancement history for an image or all attempts"""
        if input_image_hash:
            return [
                attempt
                for attempt in self.enhancement_attempts
                if attempt.input_image_hash == input_image_hash
            ]
        return self.enhancement_attempts

    def get_successful_parameters(self, input_image_hash: str) -> Dict:
        """Get parameters from successful enhancement attempts"""
        successful_attempts = [
            attempt
            for attempt in self.enhancement_attempts
            if attempt.input_image_hash == input_image_hash and attempt.success
        ]

        if not successful_attempts:
            return {}

        # Aggregate parameters from successful attempts
        aggregated_params = {}
        for model_name in set().union(
            *(attempt.parameters.keys() for attempt in successful_attempts)
        ):
            model_params = {}
            for param_name in set().union(
                *(
                    attempt.parameters[model_name].keys()
                    for attempt in successful_attempts
                    if model_name in attempt.parameters
                )
            ):
                values = [
                    attempt.parameters[model_name][param_name]
                    for attempt in successful_attempts
                    if model_name in attempt.parameters
                    and param_name in attempt.parameters[model_name]
                ]
                if values:
                    model_params[param_name] = np.mean(values)
            aggregated_params[model_name] = model_params

        return aggregated_params

    def get_enhancement_suggestions(self, input_image_hash: str) -> Dict:
        """Get enhancement suggestions based on history"""
        # Get successful parameters for this image
        image_params = self.get_successful_parameters(input_image_hash)

        # Get model-based suggestions
        model_suggestions = {
            model_name: self.model_manager.get_enhancement_suggestions(model_name)
            for model_name in [
                "super_resolution",
                "color_enhancement",
                "detail_enhancement",
            ]
        }

        # Combine image-specific and model-based suggestions
        suggestions = {}
        for model_name in model_suggestions:
            if model_name in image_params:
                # Weighted average of image-specific and model-based parameters
                suggestions[model_name] = {
                    param: 0.7 * image_params[model_name].get(param, value)
                    + 0.3 * value
                    for param, value in model_suggestions[model_name].items()
                }
            else:
                suggestions[model_name] = model_suggestions[model_name]

        return suggestions

    def apply_feedback(self, input_image_hash: str, feedback: Dict):
        """Apply user feedback to model parameters"""
        # Update model parameters based on feedback
        for model_name, model_feedback in feedback.items():
            self.model_manager.adapt_parameters(model_name, model_feedback)

        self._save_session()

    def get_quality_metrics_summary(
        self, input_image_hash: Optional[str] = None
    ) -> Dict:
        """Get summary of quality metrics"""
        attempts = self.get_enhancement_history(input_image_hash)
        if not attempts:
            return {}

        metrics_summary = {}
        for attempt in attempts:
            for metric, value in attempt.quality_metrics.items():
                if metric not in metrics_summary:
                    metrics_summary[metric] = []
                metrics_summary[metric].append(value)

        return {
            metric: {
                "mean": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "latest": values[-1],
            }
            for metric, values in metrics_summary.items()
        }
