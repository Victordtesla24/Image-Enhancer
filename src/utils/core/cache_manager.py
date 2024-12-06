import os
from typing import Any, Dict, Optional
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
from diskcache import Cache

class CacheManager:
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 10.0):
        """
        Initialize the cache manager using diskcache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in gigabytes
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.cache = Cache(directory=str(self.cache_dir), size_limit=int(self.max_size))
        self.metadata_key = "cache_metadata"
        self._init_metadata()
        
    def _init_metadata(self) -> None:
        """Initialize or load cache metadata."""
        if self.metadata_key not in self.cache:
            self.cache[self.metadata_key] = {
                "entries": {},
                "size": 0
            }
    
    def _get_metadata(self) -> Dict:
        """Get current metadata."""
        return self.cache[self.metadata_key]
    
    def _update_metadata(self, metadata: Dict) -> None:
        """Update metadata in cache."""
        self.cache[self.metadata_key] = metadata
    
    def cache_model_state(self, model_id: str, state: Dict[str, torch.Tensor]) -> bool:
        """Cache model state dictionary."""
        try:
            key = f"model_{model_id}"
            self.cache[key] = state
            
            metadata = self._get_metadata()
            metadata["entries"][key] = {
                "type": "model_state",
                "model_id": model_id,
                "timestamp": datetime.now().isoformat()
            }
            self._update_metadata(metadata)
            return True
        except Exception as e:
            print(f"Error caching model state: {e}")
            return False
    
    def cache_enhancement_result(self, image_id: str, result: np.ndarray) -> bool:
        """Cache image enhancement result."""
        try:
            key = f"result_{image_id}"
            self.cache[key] = result
            
            metadata = self._get_metadata()
            metadata["entries"][key] = {
                "type": "enhancement_result",
                "image_id": image_id,
                "timestamp": datetime.now().isoformat()
            }
            self._update_metadata(metadata)
            return True
        except Exception as e:
            print(f"Error caching enhancement result: {e}")
            return False
    
    def get_cached_model_state(self, model_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached model state."""
        try:
            key = f"model_{model_id}"
            if key in self.cache:
                return self.cache[key]
            return None
        except Exception as e:
            print(f"Error loading cached model state: {e}")
            return None
    
    def get_cached_result(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieve cached enhancement result."""
        try:
            key = f"result_{image_id}"
            if key in self.cache:
                return self.cache[key]
            return None
        except Exception as e:
            print(f"Error loading cached result: {e}")
            return None
    
    def optimize_cache_usage(self) -> None:
        """
        Optimize cache usage by removing old entries when size limit is exceeded.
        Note: diskcache handles size limits automatically.
        """
        pass  # diskcache handles this automatically
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        try:
            self.cache.clear()
            self._init_metadata()
        except Exception as e:
            print(f"Error clearing cache: {e}")
