"""Batch processing functionality."""

import logging
from typing import List, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import BaseProcessor

logger = logging.getLogger(__name__)

class BatchProcessor(BaseProcessor):
    """Handles batch processing of data."""

    def _process_batch(self, batch: List[NDArray[np.uint8]]) -> List[NDArray[np.uint8]]:
        """Process a batch of data.
        
        Args:
            batch: List of arrays to process
            
        Returns:
            List of processed arrays
        """
        results = []
        for item in batch:
            try:
                # Apply processing
                processed = self._apply_processing(item)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error processing batch item: {str(e)}")
                continue
        return results

    def _apply_processing(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply processing to single item.
        
        Args:
            data: Array to process
            
        Returns:
            Processed array
        """
        # This should be implemented by subclasses
        return data.copy()

    def process_dataset(self, dataset_path: Union[str, List[str]], session_id: str) -> None:
        """Process a dataset of items.
        
        Args:
            dataset_path: Path to dataset or list of paths
            session_id: Session identifier
            
        Raises:
            ValueError: If dataset is empty or invalid
            RuntimeError: If resource limits are exceeded
        """
        if not dataset_path:
            raise ValueError("Empty dataset")
            
        if not session_id:
            raise ValueError("Missing session ID")
            
        # Check resource availability
        if not self.device_manager.check_resource_availability():
            raise RuntimeError("Resource limits exceeded")
            
        try:
            # Create session
            if not self.session_manager.create_session(session_id, {}):
                raise ValueError(f"Session {session_id} already exists")
                
            # Process dataset
            if isinstance(dataset_path, str):
                dataset_path = [dataset_path]
                
            for path in dataset_path:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Path not found: {path}")
                    
            # Process in batches
            batch_size = self.config.get('batch_size', 4)
            for i in range(0, len(dataset_path), batch_size):
                # Check resource availability before each batch
                if not self.device_manager.check_resource_availability():
                    raise RuntimeError("Resource limits exceeded during batch processing")
                    
                batch = dataset_path[i:i + batch_size]
                self._process_batch(batch)
                
        except Exception as e:
            self.error_handler.handle_error(e, {'session_id': session_id})
            raise
        finally:
            self.session_manager.cleanup_session(session_id) 