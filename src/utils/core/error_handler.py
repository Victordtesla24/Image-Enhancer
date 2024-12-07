"""Advanced error handling and recovery system"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Handles errors and exceptions in the enhancement pipeline."""

    def __init__(self):
        """Initialize error handler."""
        self.logger = logging.getLogger(__name__)
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.error_thresholds = {
            "memory": 0.9,  # 90% memory usage threshold
            "gpu_memory": 0.85,  # 85% GPU memory threshold
            "processing_time": 30.0,  # 30 seconds timeout
        }

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Union[str, int, float]],
    ) -> Optional[Dict]:
        """Handle an error with context and attempt recovery.

        Args:
            error: The exception that occurred
            context: Dictionary with error context

        Returns:
            Recovery instructions if available, None otherwise
        """
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "timestamp": time.time(),
            "recovered": False,
        }

        # Generate error ID for tracking recovery attempts
        error_id = f"{error_info['type']}_{context.get('operation', 'unknown')}"

        # Track recovery attempts
        if error_id in self.recovery_attempts:
            self.recovery_attempts[error_id] += 1
        else:
            self.recovery_attempts[error_id] = 1

        # Attempt recovery if within limits
        if self.recovery_attempts[error_id] <= self.max_recovery_attempts:
            recovery_instructions = self._attempt_recovery(error, context)
            if recovery_instructions:
                error_info["recovered"] = True
                error_info["recovery"] = recovery_instructions

        self.error_history.append(error_info)
        self.logger.error(
            "Error occurred: {} - {}".format(
                error_info['type'], 
                error_info['message']
            )
        )

        return error_info.get("recovery")

    def _attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
    ) -> Optional[Dict]:
        """Attempt to recover from error.

        Args:
            error: The exception that occurred
            context: Error context

        Returns:
            Recovery instructions if available
        """
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return self._handle_gpu_memory_error(context)
        elif isinstance(error, MemoryError):
            return self._handle_memory_error(context)
        elif isinstance(error, TimeoutError):
            return self._handle_timeout_error(context)
        elif isinstance(error, ValueError):
            return self._handle_validation_error(context)
        return None

    def _handle_gpu_memory_error(self, context: Dict[str, Any]) -> Dict:
        """Handle GPU memory errors.

        Args:
            context: Error context

        Returns:
            Recovery instructions
        """
        current_batch = context.get("batch_size", 1)
        return {
            "action": "reduce_batch_size",
            "current_batch_size": current_batch,
            "recommended_batch_size": max(1, current_batch // 2),
            "clear_cache": True,
        }

    def _handle_memory_error(self, context: Dict[str, Any]) -> Dict:
        """Handle system memory errors.

        Args:
            context: Error context

        Returns:
            Recovery instructions
        """
        return {
            "action": "optimize_memory",
            "clear_cache": True,
            "reduce_precision": True,
            "enable_checkpointing": True,
        }

    def _handle_timeout_error(self, context: Dict[str, Any]) -> Dict:
        """Handle timeout errors.

        Args:
            context: Error context

        Returns:
            Recovery instructions
        """
        current_timeout = context.get("timeout", 30)
        new_timeout = min(current_timeout * 1.5, 120)
        return {
            "action": "optimize_processing",
            "timeout": new_timeout,
            "enable_async": True,
            "reduce_quality": True,
        }

    def _handle_validation_error(self, context: Dict[str, Any]) -> Dict:
        """Handle validation errors.

        Args:
            context: Error context

        Returns:
            Recovery instructions
        """
        return {
            "action": "validate_input",
            "validation_rules": self._get_validation_rules(context),
            "sanitize_input": True,
        }

    def _get_validation_rules(self, context: Dict[str, Any]) -> Dict:
        """Get validation rules based on context.

        Args:
            context: Error context

        Returns:
            Validation rules
        """
        return {
            "image": {
                "min_size": (32, 32),
                "max_size": (8192, 8192),
                "formats": ["jpg", "png", "jpeg"],
            },
            "parameters": {
                "range": (0.0, 2.0),
                "types": ["float", "int"],
                "required": ["quality", "format"],
            },
        }

    def verify_tensor(self, tensor: torch.Tensor) -> bool:
        """Verify tensor validity.

        Args:
            tensor: Tensor to verify

        Returns:
            True if tensor is valid
        """
        try:
            if not isinstance(tensor, torch.Tensor):
                return False
            if torch.isnan(tensor).any():
                return False
            if torch.isinf(tensor).any():
                return False
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return True
        except Exception as e:
            shape = (
                str(tensor.shape) if hasattr(tensor, "shape") else "unknown"
            )
            self.handle_error(e, {"tensor_shape": shape})
            return False

    def get_error_history(self) -> List[Dict]:
        """Get the error history.

        Returns:
            List of error dictionaries
        """
        return self.error_history

    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history = []
        self.recovery_attempts = {}
