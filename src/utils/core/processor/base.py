"""Base processor module."""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..error_handler import ErrorHandler
from ..device_manager import DeviceManager
from ...model_management.model_manager import ModelManager
from ...session_management.session_manager import SessionManager

logger = logging.getLogger(__name__)

class BaseProcessor:
    """Base class for processing functionality."""

    def __init__(self) -> None:
        """Initialize processor."""
        self.initialized = False
        self.config: Dict[str, Any] = {}
        
        # Initialize components
        self.model_manager = ModelManager()
        self.device_manager = DeviceManager()
        self.session_manager = SessionManager()
        self.error_handler = ErrorHandler()
        
        self.initialize()

    def initialize(self) -> None:
        """Initialize processor."""
        if self.initialized:
            return
            
        # Initialize components
        self.model_manager.initialize()
        self.device_manager.initialize()
        self.session_manager.initialize()
        self.error_handler.initialize()
        
        # Set default config
        self.config = {
            'batch_size': 4,
            'max_memory_usage': 0.8,
            'max_cpu_usage': 0.9,
            'timeout': 30,
            'retry_attempts': 3,
            'log_level': 'INFO'
        }
        
        self.initialized = True

    def get_error_info(self) -> Dict[str, Any]:
        """Get error information.
        
        Returns:
            Error information dictionary
        """
        return {
            'error': self.error_handler.last_error,
            'timestamp': self.error_handler.last_error_time,
            'count': len(self.error_handler.error_history)
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.device_manager.cleanup()
        self.session_manager.cleanup() 