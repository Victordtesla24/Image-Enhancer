"""Advanced cache management system."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of model outputs and intermediate results."""

    def __init__(self, cache_dir="cache", max_memory_usage=0.8, cleanup_interval=3600):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            max_memory_usage: Maximum memory usage in bytes
            cleanup_interval: Cache cleanup interval in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_usage = max_memory_usage * self._get_available_memory()
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()

        # Create cache directories
        self.models_dir = self.cache_dir / "models"
        self.results_dir = self.cache_dir / "results"
        self.temp_dir = self.cache_dir / "temp"

        for directory in [self.models_dir, self.results_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize memory tracking
        self.memory_usage = {}
        self.access_history = {}

        # Perform initial cleanup
        self.cleanup_cache()

    def save_model_cache(self, model_id: str, data: Dict[str, Any]) -> bool:
        """Save model cache data.

        Args:
            model_id: Model identifier
            data: Data to cache

        Returns:
            bool: Success status
        """
        try:
            cache_path = self.models_dir / f"{model_id}.json"

            # Convert data to JSON string to get accurate size
            json_data = json.dumps(data)
            data_size = len(json_data.encode("utf-8"))

            # Check memory usage before saving
            if not self._check_memory_available(data_size):
                self.cleanup_cache()
                if not self._check_memory_available(data_size):
                    logger.warning(f"Insufficient memory to cache model {model_id}")
                    return False

            # Save data
            with open(cache_path, "w") as f:
                f.write(json_data)

            # Update tracking
            self._update_tracking(str(cache_path), data_size)
            return True

        except Exception as e:
            logger.error(f"Error saving model cache: {str(e)}")
            return False

    def load_model_cache(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model cache data.

        Args:
            model_id: Model identifier

        Returns:
            Cached data or None if not found
        """
        try:
            cache_path = self.models_dir / f"{model_id}.json"
            if not cache_path.exists():
                return None

            with open(cache_path) as f:
                data = json.load(f)

            # Update access history
            self._update_access(str(cache_path))
            return data

        except Exception as e:
            logger.error(f"Error loading model cache: {str(e)}")
            return None

    def save_result(
        self,
        result_id: str,
        data: Union[Dict, np.ndarray, torch.Tensor, Image.Image],
    ) -> bool:
        """Save processing result.

        Args:
            result_id: Result identifier
            data: Result data

        Returns:
            bool: Success status
        """
        try:
            # Convert data to appropriate format and get size
            if isinstance(data, (np.ndarray, torch.Tensor)):
                result_path = self.results_dir / f"{result_id}.npy"
                data_size = self._get_array_size(data)
                if self._check_memory_available(data_size):
                    np.save(result_path, self._convert_to_numpy(data))
                else:
                    return False
            elif isinstance(data, Image.Image):
                result_path = self.results_dir / f"{result_id}.png"
                data_size = data.size[0] * data.size[1] * len(data.getbands())
                if self._check_memory_available(data_size):
                    data.save(result_path)
                else:
                    return False
            else:
                result_path = self.results_dir / f"{result_id}.json"
                json_data = json.dumps(data)
                data_size = len(json_data.encode("utf-8"))
                if self._check_memory_available(data_size):
                    with open(result_path, "w") as f:
                        f.write(json_data)
                else:
                    return False

            # Update tracking
            self._update_tracking(str(result_path), data_size)
            return True

        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            return False

    def load_result(
        self, result_id: str
    ) -> Optional[Union[Dict, np.ndarray, Image.Image]]:
        """Load processing result.

        Args:
            result_id: Result identifier

        Returns:
            Result data or None if not found
        """
        try:
            # Check different formats
            paths = [
                self.results_dir / f"{result_id}.{ext}"
                for ext in ["json", "npy", "png"]
            ]

            for path in paths:
                if path.exists():
                    # Update access history
                    self._update_access(str(path))

                    # Load based on format
                    if path.suffix == ".json":
                        with open(path) as f:
                            return json.load(f)
                    elif path.suffix == ".npy":
                        return np.load(path)
                    else:
                        return Image.open(path)

            return None

        except Exception as e:
            logger.error(f"Error loading result: {str(e)}")
            return None

    def cleanup_cache(self):
        """Clean up cached data based on memory usage and access patterns."""
        current_time = time.time()

        # Check if cleanup is needed
        if (
            current_time - self.last_cleanup < self.cleanup_interval
            and self._get_total_memory_usage() < self.max_memory_usage
        ):
            return

        try:
            # Sort files by last access time
            files = sorted(self.access_history.items(), key=lambda x: x[1])

            # Remove files until memory usage is acceptable
            target_usage = self.max_memory_usage * 0.8
            for filepath, _ in files:
                if self._get_total_memory_usage() <= target_usage:
                    break

                if os.path.exists(filepath):
                    os.remove(filepath)
                    self._remove_tracking(filepath)

            self.last_cleanup = current_time

        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def _convert_to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _check_memory_available(self, size: int) -> bool:
        """Check if memory is available for new data."""
        return (self._get_total_memory_usage() + size) <= self.max_memory_usage

    def _get_array_size(self, data: Union[np.ndarray, torch.Tensor]) -> int:
        """Get size of array in bytes."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        return data.nbytes

    def _get_total_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        return sum(self.memory_usage.values())

    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        try:
            import psutil

            return psutil.virtual_memory().available
        except ImportError:
            # Default to 1GB if psutil not available
            return 1024 * 1024 * 1024

    def _update_tracking(self, filepath: str, size: int):
        """Update memory tracking for file."""
        self.memory_usage[filepath] = size
        self._update_access(filepath)

    def _update_access(self, filepath: str):
        """Update access history for file."""
        self.access_history[filepath] = time.time()

    def _remove_tracking(self, filepath: str):
        """Remove file from tracking."""
        self.memory_usage.pop(filepath, None)
        self.access_history.pop(filepath, None)
