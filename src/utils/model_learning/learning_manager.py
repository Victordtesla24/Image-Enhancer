"""Learning manager for model adaptation based on user feedback."""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LearningManager:
    """Manages model learning and adaptation."""

    def __init__(self, feedback_dir: Optional[str] = None):
        """Initialize learning manager.

        Args:
            feedback_dir: Optional directory for feedback storage
        """
        self.feedback_dir = Path(feedback_dir) if feedback_dir else Path("feedback")
        self.feedback_dir.mkdir(exist_ok=True)

        self.style_profiles = {}
        self.active_sessions = {}
        self.feedback_history = []
        self.current_session = None
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
        self.preferences = {
            "feedback_frequency": "low_quality",
            "auto_enhance": True,
            "save_history": True,
            "learning_enabled": True,
        }
        
        # Load existing state if available
        self._load_state()

    def adapt_enhancement_parameters(
        self, metrics: Dict[str, float], feedback: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adapt enhancement parameters based on metrics and feedback.

        Args:
            metrics: Current quality metrics
            feedback: User feedback

        Returns:
            Adapted enhancement parameters
        """
        adapted_params = {}
        
        # Get aspect ratings and current parameters
        aspect_ratings = feedback.get("aspect_ratings", {})
        current_params = feedback.get("enhancement_params", {})
        
        # Parameter mapping
        param_mapping = {
            "sharpness": "sharpness",
            "contrast": "contrast",
            "detail": "detail_enhancement",
            "color": "color_boost",
            "noise": "noise_reduction"
        }
        
        for aspect, rating in aspect_ratings.items():
            if aspect not in param_mapping:
                continue
                
            param_name = param_mapping[aspect]
            current_value = current_params.get(param_name, 1.0)
            metric_value = metrics.get(aspect, 0.5)
            
            # Calculate adjustment based on rating and metric
            target = rating / 5.0  # Convert rating to 0-1 scale
            diff = target - metric_value
            
            # Adjust parameter based on difference
            adjustment = diff * 0.2  # 20% adjustment per full difference
            new_value = current_value * (1.0 + adjustment)
            
            # Clamp value to valid range
            adapted_params[param_name] = max(0.5, min(2.0, new_value))
            
        return adapted_params

    def _calculate_preference_adjustments(
        self, feedback: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate preference adjustments based on feedback.

        Args:
            feedback: User feedback dictionary

        Returns:
            Dictionary of preference adjustments
        """
        adjustments = {}
        aspect_ratings = feedback.get("aspect_ratings", {})
        
        # Parameter mapping
        param_mapping = {
            "sharpness": "sharpness",
            "detail": "detail_enhancement",
            "color": "color_boost",
            "noise": "noise_reduction",
            "texture": "texture_preservation"
        }
        
        for aspect, rating in aspect_ratings.items():
            if aspect not in param_mapping:
                continue
                
            param_name = param_mapping[aspect]
            
            # Convert rating to adjustment factor
            # Rating of 3 = no adjustment (1.0)
            # Rating of 1 = decrease (0.5)
            # Rating of 5 = increase (1.5)
            adjustment = 0.5 + (rating / 5.0)
            
            # Ensure adjustment is in valid range
            adjustments[param_name] = max(0.5, min(1.5, adjustment))
            
        return adjustments

    def process_feedback(self, feedback: Dict[str, Any]) -> None:
        """Process user feedback.

        Args:
            feedback: User feedback dictionary
        """
        if not feedback:
            return

        try:
            # Validate required fields
            required_fields = ["quality_rating", "aspect_ratings", "enhancement_params"]
            for field in required_fields:
                if field not in feedback:
                    logger.warning(f"Missing required field in feedback: {field}")
                    return

            # Validate field types
            if not isinstance(feedback["quality_rating"], (int, float)):
                logger.warning("Invalid quality_rating type")
                return
            if not isinstance(feedback["aspect_ratings"], dict):
                logger.warning("Invalid aspect_ratings type")
                return
            if not isinstance(feedback["enhancement_params"], dict):
                logger.warning("Invalid enhancement_params type")
                return

            # Add timestamp if not present
            if "timestamp" not in feedback:
                feedback["timestamp"] = datetime.now().isoformat()

            # Add session info
            if self.current_session:
                feedback["session_id"] = self.current_session["id"]
                self.current_session["feedback"].append(feedback)

            # Store feedback
            self.feedback_history.append(feedback)
            self._save_feedback(feedback)

            # Create or update style profile
            self.create_style_profile(feedback)

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")

    def create_style_profile(self, feedback: Dict[str, Any]) -> str:
        """Create new style profile.

        Args:
            feedback: User feedback dictionary

        Returns:
            Profile identifier
        """
        if "aspect_ratings" not in feedback:
            raise ValueError("Missing aspect ratings")

        ratings = feedback["aspect_ratings"]
        required_fields = ["sharpness", "detail", "noise"]

        # Add missing fields with default values
        for field in required_fields:
            if field not in ratings:
                ratings[field] = 3.0  # Default to neutral rating

        # Clamp ratings to valid range
        for field, value in ratings.items():
            if isinstance(value, (int, float)):
                ratings[field] = max(1, min(5, value))

        # Validate ratings
        for field in required_fields:
            value = ratings[field]
            if not isinstance(value, (int, float)):
                msg = (
                    f"Invalid type for rating {field}: "
                    f"expected numeric, got {type(value).__name__}"
                )
                raise TypeError(msg)

            if value < 1 or value > 5:
                msg = f"Invalid value for rating {field}: must be between 1 and 5"
                raise ValueError(msg)

        profile_id = self._generate_session_id()
        
        # Check if similar profile exists
        best_match = None
        best_score = 0.0
        for pid, profile in self.style_profiles.items():
            score = self._calculate_profile_match(profile, {k: v/5.0 for k, v in ratings.items()})
            if score > 0.8:  # 80% similarity threshold
                best_match = pid
                best_score = score
                break

        if best_match:
            # Update existing profile
            self._update_style_profile(feedback, best_match)
            return best_match
        else:
            # Create new profile
            self.style_profiles[profile_id] = {
                "preferences": {
                    field: ratings[field] / 5.0  # Convert to 0-1 range
                    for field in required_fields
                },
                "parameters": feedback.get("enhancement_params", {}),
                "ratings": ratings,  # Store original ratings
                "quality_rating": feedback.get("quality_rating", 3),
                "created": time.time(),
                "last_update": time.time(),
                "feedback_count": 1,
                "success_rate": (1.0 if feedback.get("quality_rating", 0) >= 4 else 0.0),
                "weights": {field: 1.0 for field in required_fields},
            }
            return profile_id

    def get_style_recommendation(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Get style recommendations based on metrics.

        Args:
            metrics: Current quality metrics

        Returns:
            Enhancement parameters
        """
        if not metrics or not self.style_profiles:
            return {}

        # Find best matching profile
        best_match = None
        best_score = 0.0

        for profile_id, profile in self.style_profiles.items():
            score = self._calculate_profile_match(profile, metrics)
            if score > best_score:
                best_score = score
                best_match = profile_id

        if best_match:
            profile = self.style_profiles[best_match]
            return self._calculate_adjustments(profile, metrics)
        
        return {}

    def _generate_session_id(self) -> str:
        """Generate unique session identifier.

        Returns:
            Session identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{random_suffix}"

    def _save_feedback(self, feedback: Dict[str, Any]) -> None:
        """Save feedback to file.

        Args:
            feedback: Feedback dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        feedback_file = self.feedback_dir / f"feedback_{timestamp}.json"
        try:
            with open(feedback_file, "w") as f:
                json.dump(feedback, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

    def _update_style_profile(self, feedback: Dict[str, Any], profile_id: str) -> None:
        """Update style profile based on feedback.

        Args:
            feedback: Feedback dictionary
            profile_id: Profile identifier
        """
        if profile_id not in self.style_profiles:
            return

        profile = self.style_profiles[profile_id]
        profile["last_update"] = time.time()
        profile["feedback_count"] += 1

        # Update success rate
        quality_rating = feedback.get("quality_rating", 0)
        new_success = 1.0 if quality_rating >= 4 else 0.0
        old_rate = profile["success_rate"]
        old_count = profile["feedback_count"] - 1
        profile["success_rate"] = (old_rate * old_count + new_success) / profile[
            "feedback_count"
        ]

        # Update preferences
        if "aspect_ratings" in feedback:
            ratings = feedback["aspect_ratings"]
            for field in profile["preferences"]:
                if field in ratings:
                    value = min(5, max(1, ratings[field])) / 5.0  # Clamp and convert to 0-1 range
                    # Use exponential moving average for smooth updates
                    alpha = 0.3
                    old_value = profile["preferences"][field]
                    profile["preferences"][field] = (
                        alpha * value + (1 - alpha) * old_value
                    )

        # Update parameters
        if "enhancement_params" in feedback and quality_rating >= 4:
            new_params = feedback["enhancement_params"]
            old_params = profile["parameters"]
            
            # Use exponential moving average for parameter updates
            alpha = 0.3
            for param, value in new_params.items():
                if param in old_params:
                    old_params[param] = alpha * value + (1 - alpha) * old_params[param]
                else:
                    old_params[param] = value

    def _calculate_profile_match(
        self, profile: Dict[str, Any], metrics: Dict[str, float]
    ) -> float:
        """Calculate match score between profile and metrics.

        Args:
            profile: Style profile
            metrics: Image metrics

        Returns:
            Match score between 0 and 1
        """
        if not profile["preferences"] or not metrics:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in profile["preferences"]:
                weight = profile["weights"].get(metric, 1.0)
                target = profile["preferences"][metric]
                score = 1.0 - min(abs(value - target), 1.0)
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_adjustments(
        self, profile: Dict[str, Any], metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate parameter adjustments based on profile and metrics.

        Args:
            profile: Style profile
            metrics: Current quality metrics

        Returns:
            Dictionary of parameter adjustments
        """
        return profile["parameters"].copy()

    def _validate_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Validate enhancement parameters.

        Args:
            params: Enhancement parameters

        Returns:
            Validated parameters
        """
        validated = {}

        for param, value in params.items():
            if not isinstance(value, (int, float)):
                continue

            # Validate value ranges
            if param in ["contrast", "brightness", "saturation"]:
                # These parameters should be between 0.5 and 2.0
                validated[param] = max(0.5, min(2.0, value))
            else:
                # All other parameters should be between 0.0 and 1.0
                validated[param] = max(0.0, min(1.0, value))

        return validated

    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user preferences.

        Args:
            preferences: User preferences dictionary
        """
        # Validate preferences
        valid_frequencies = ["always", "low_quality", "never"]
        if "feedback_frequency" in preferences:
            freq = preferences["feedback_frequency"]
            if freq not in valid_frequencies:
                msg = (
                    f"Invalid feedback frequency: "
                    f"must be one of {valid_frequencies}"
                )
                raise ValueError(msg)

        self.preferences.update(preferences)
        self._save_state()

    def cleanup(self) -> None:
        """Clean up resources."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        # Remove old profiles
        inactive_profiles = [
            profile_id
            for profile_id, profile in self.style_profiles.items()
            if (
                current_time - profile["last_update"] > self.cleanup_interval
                and profile["feedback_count"] < 5
            )
        ]

        for profile_id in inactive_profiles:
            del self.style_profiles[profile_id]

        self._cleanup_feedback_dir()
        self.last_cleanup = current_time
        self._save_state()

    def _cleanup_feedback_dir(self) -> None:
        """Clean up feedback directory."""
        if self.feedback_dir.exists():
            shutil.rmtree(self.feedback_dir)

    def start_session(self) -> str:
        """Start new learning session.

        Returns:
            Session identifier
        """
        # End current session if exists
        if self.current_session:
            self.end_session()
            
        session_id = self._generate_session_id()
        session = {
            "id": session_id,
            "start_time": time.time(),
            "last_update": time.time(),
            "feedback": [],
            "metrics": [],
            "status": "active",
        }
        self.active_sessions[session_id] = session
        self.current_session = session
        
        # Save session to file
        self._save_session(session)
        return session_id

    def end_session(self) -> None:
        """End current learning session."""
        if self.current_session:
            session_id = self.current_session["id"]
            session = self.active_sessions[session_id]
            session["status"] = "completed"
            session["end_time"] = time.time()
            self._save_session(session)
            self.current_session = None

    def _save_session(self, session: Dict[str, Any]) -> None:
        """Save session to file.

        Args:
            session: Session dictionary
        """
        session_file = self.feedback_dir / f"session_{session['id']}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(session, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def _save_state(self) -> None:
        """Save learning state to file."""
        state = {
            "style_profiles": self.style_profiles,
            "preferences": self.preferences,
            "last_cleanup": self.last_cleanup,
        }
        
        state_file = self.feedback_dir / "learning_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def _load_state(self) -> None:
        """Load learning state from file."""
        state_file = self.feedback_dir / "learning_state.json"
        if not state_file.exists():
            return
            
        try:
            with open(state_file) as f:
                state = json.load(f)
                self.style_profiles = state.get("style_profiles", {})
                self.preferences = state.get("preferences", self.preferences)
                self.last_cleanup = state.get("last_cleanup", time.time())
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
