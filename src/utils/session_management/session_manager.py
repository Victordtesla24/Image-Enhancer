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

    def __init__(self) -> None:
        """Initialize session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        
    def initialize(self) -> None:
        """Initialize the session manager."""
        if self.initialized:
            return
            
        logger.info("Initializing session manager")
        self.sessions = {}
        self.initialized = True
        
    def create_session(self, session_id: str, config: Dict[str, Any]) -> bool:
        """Create new session.
        
        Args:
            session_id: Session identifier
            config: Session configuration
            
        Returns:
            True if session created, False if exists
        """
        if session_id in self.sessions:
            return False
            
        self.sessions[session_id] = {
            'config': config,
            'start_time': time.time(),
            'status': 'created'
        }
        return True
        
    def update_session(self, session_id: str, progress: float) -> None:
        """Update session progress.
        
        Args:
            session_id: Session identifier
            progress: Progress value (0-1)
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id].update({
            'progress': progress,
            'last_update': time.time()
        })
        
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        return self.sessions[session_id]
        
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = 'cleaned'
            del self.sessions[session_id]
            
    def cleanup(self) -> None:
        """Clean up all sessions."""
        for session_id in list(self.sessions.keys()):
            self.cleanup_session(session_id)

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

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session information
        
        Raises:
            ValueError: If session not found
        """
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            return {
                'id': session_id,
                'start_time': session['start_time'],
                'end_time': session.get('end_time'),
                'status': session['status'],
                'processed_items': session.get('processed_items', 0),
                'total_items': session.get('total_items', 0),
                'errors': session.get('errors', []),
                'metrics': session.get('metrics', {}),
                'config': session.get('config', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session info: {str(e)}")
            raise

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

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources.
        
        Args:
            session_id: Session identifier
        
        Raises:
            ValueError: If session not found
        """
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Clean up resources
            session = self.sessions[session_id]
            if 'resources' in session:
                for resource in session['resources']:
                    try:
                        if hasattr(resource, 'cleanup'):
                            resource.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up resource: {str(e)}")
            
            # Remove session
            del self.sessions[session_id]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up session {session_id}: {str(e)}")
            raise
