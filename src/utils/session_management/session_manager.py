"""Session Management System"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Manages processing sessions."""

    def __init__(self):
        """Initialize session manager."""
        self.sessions = {}
        self.session_history = []
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()

    def start_session(self) -> str:
        """Start new processing session.

        Returns:
            Session identifier
        """
        session_id = self._generate_session_id()
        self.sessions[session_id] = {
            "status": "active",
            "start_time": time.time(),
            "last_update": time.time(),
            "metrics": {
                "initial": {},
                "final": {},
                "improvement": {},
            },
            "error": None,
        }
        return session_id

    def update_session_metrics(
        self,
        session_id: str,
        initial_metrics: Dict[str, float],
        final_metrics: Dict[str, float],
        improvement_metrics: Dict[str, float],
    ) -> None:
        """Update session metrics.

        Args:
            session_id: Session identifier
            initial_metrics: Initial quality metrics
            final_metrics: Final quality metrics
            improvement_metrics: Improvement metrics
        """
        if session_id in self.sessions:
            self.sessions[session_id]["metrics"] = {
                "initial": initial_metrics,
                "final": final_metrics,
                "improvement": improvement_metrics,
            }
            self.sessions[session_id]["last_update"] = time.time()

    def mark_session_error(self, session_id: str, error_message: str) -> None:
        """Mark session as error.

        Args:
            session_id: Session identifier
            error_message: Error message
        """
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "error"
            self.sessions[session_id]["error"] = error_message
            self.sessions[session_id]["end_time"] = time.time()

    def end_session(self, session_id: str) -> None:
        """End processing session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "completed"
            self.sessions[session_id]["end_time"] = time.time()
            self.session_history.append(self.sessions[session_id])
            del self.sessions[session_id]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Session information or None if not found
        """
        return self.sessions.get(session_id)

    def get_active_sessions(self) -> List[str]:
        """Get active session IDs.

        Returns:
            List of active session IDs
        """
        return [
            session_id
            for session_id, info in self.sessions.items()
            if info["status"] == "active"
        ]

    def cleanup_sessions(self) -> None:
        """Clean up inactive sessions."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        inactive_sessions = [
            session_id
            for session_id, info in self.sessions.items()
            if current_time - info["last_update"] > self.cleanup_interval
        ]

        for session_id in inactive_sessions:
            self.end_session(session_id)

        self.last_cleanup = current_time

    def _generate_session_id(self) -> str:
        """Generate unique session identifier.

        Returns:
            Session identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = uuid.uuid4().hex[:8]
        return f"{timestamp}_{random_suffix}"
