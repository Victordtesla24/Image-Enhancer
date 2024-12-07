"""GPU acceleration module for image processing."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU devices and memory."""

    def __init__(self):
        """Initialize device manager."""
        self.devices = self._get_available_devices()
        self.current_device = 0

    def get_device(self) -> torch.device:
        """Get current device.

        Returns:
            Current device
        """
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.current_device}")
        return torch.device("cpu")

    def get_next_device(self) -> torch.device:
        """Get next available device.

        Returns:
            Next available device
        """
        if torch.cuda.is_available():
            self.current_device = (self.current_device + 1) % torch.cuda.device_count()
            return torch.device(f"cuda:{self.current_device}")
        return torch.device("cpu")

    def get_optimal_device(self) -> torch.device:
        """Get device with most available memory.

        Returns:
            Device with most available memory
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")

        stats = self.memory_stats()
        if not stats:
            return torch.device("cpu")

        # Find device with most available memory
        def get_available_memory(item):
            return item[1]["total"] - item[1]["allocated"]

        optimal_device = max(stats.items(), key=get_available_memory)[0]
        self.current_device = int(optimal_device.split(":")[-1])
        return torch.device(f"cuda:{self.current_device}")

    def _get_available_devices(self) -> List[dict]:
        """Get available GPU devices."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": props.total_memory,
                        "memory_used": 0,
                    }
                )
            logger.info(f"Initialized {device_count} GPU devices")
            return devices
        return [{"index": -1, "name": "CPU", "memory_total": 0, "memory_used": 0}]

    def memory_stats(self) -> dict:
        """Get memory statistics for all devices."""
        stats = {}
        for device in self.devices:
            if device["index"] >= 0:
                with torch.cuda.device(device["index"]):
                    stats[device["name"]] = {
                        "allocated": torch.cuda.memory_allocated(),
                        "cached": torch.cuda.memory_reserved(),
                        "total": device["memory_total"],
                    }
        return stats

    def clear_memory(self):
        """Clear memory on all devices."""
        if torch.cuda.is_available():
            for device in self.devices:
                if device["index"] >= 0:
                    with torch.cuda.device(device["index"]):
                        torch.cuda.empty_cache()
                        device["memory_used"] = 0


class GPUAccelerator:
    """Manages GPU acceleration and device management."""

    def __init__(self):
        """Initialize GPU accelerator."""
        self.device_manager = DeviceManager()
        self.initialized = False
        self.batch_size = self._calculate_optimal_batch_size()

    def initialize(self):
        """Initialize GPU acceleration."""
        if self.initialized:
            return
            
        self.initialized = True

    def get_device(self) -> torch.device:
        """Get current device.
        
        Returns:
            Current device
        """
        return self.device_manager.get_device()

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return 4  # Default CPU batch size

        # Get memory of largest GPU
        max_memory = 0
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory
            max_memory = max(max_memory, memory)

        # Estimate memory per image (assuming 1024x1024 RGB float32)
        mem_per_image = 1024 * 1024 * 3 * 4  # Height * Width * Channels * Bytes

        # Use 70% of available memory
        usable_memory = max_memory * 0.7

        # Calculate batch size
        optimal_size = max(1, int(usable_memory / (mem_per_image * 2)))
        return min(optimal_size, 32)  # Cap at 32 to prevent excessive memory usage

    def process_batch(self, batch: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Process a batch of tensors.
        
        Args:
            batch: Input batch tensor or list of tensors
            
        Returns:
            List of processed tensors
        """
        try:
            # Convert list to tensor if needed
            if isinstance(batch, list):
                if not batch:
                    return []
                if isinstance(batch[0], np.ndarray):
                    batch = [torch.from_numpy(arr) for arr in batch]
                batch = torch.stack(batch)

            # Move to GPU if available
            device = self.get_device()
            batch = batch.to(device)
            
            # Process batch
            processed = []
            for i in range(len(batch)):
                # Get individual tensor
                tensor = batch[i].unsqueeze(0)  # Add batch dimension
                
                # Apply processing
                enhanced = self._apply_processing(tensor)
                
                # Remove batch dimension and append
                processed.append(enhanced.squeeze(0))
            
            # Move back to CPU
            processed = [p.cpu() for p in processed]
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return []

    def _process_batch_single(self, tensors: List[Union[np.ndarray, torch.Tensor]]) -> List[torch.Tensor]:
        """Process batch on single device.
        
        Args:
            tensors: List of input arrays or tensors
            
        Returns:
            List of processed tensors
        """
        device = self.get_device()
        results = []
        
        for tensor in tensors:
            # Convert numpy array to tensor if needed
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            
            # Move to device
            tensor = tensor.to(device)
            
            # Process tensor
            processed = self._apply_processing(tensor)
            
            # Move back to CPU
            results.append(processed.cpu())
            
        return results

    def _apply_processing(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply processing to a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Processed tensor
        """
        # Placeholder for actual processing
        # For now, just return the input tensor
        return tensor

    def to_gpu(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor.
        
        Args:
            array: Input numpy array
            
        Returns:
            PyTorch tensor on GPU
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Ensure float32 and correct shape
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0

        # Add batch dimension if needed
        if len(array.shape) == 3:
            array = np.expand_dims(array, 0)

        # Convert to tensor
        tensor = torch.from_numpy(array)

        # Ensure channel dimension is second
        if tensor.shape[1] != 3 and len(tensor.shape) == 4:
            tensor = tensor.permute(0, 3, 1, 2)

        return tensor.to(self.get_device())

    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        # Move to CPU and convert to numpy
        array = tensor.detach().cpu().numpy()

        # Remove batch dimension
        if array.shape[0] == 1:
            array = array[0]

        # Move channels to last dimension
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))

        return array

    def clear_memory(self):
        """Clear GPU memory cache."""
        self.device_manager.clear_memory()

    def get_memory_stats(self) -> dict:
        """Get GPU memory statistics."""
        return self.device_manager.memory_stats()
