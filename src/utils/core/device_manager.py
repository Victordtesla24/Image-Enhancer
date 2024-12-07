import os
import psutil
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device resources."""
    
    def __init__(self) -> None:
        """Initialize device manager."""
        self.initialized = False
        self.device = 'cpu'
        self.resource_limits = {
            'max_memory_usage': 0.8,  # 80%
            'max_cpu_usage': 0.9      # 90%
        }
        
    def initialize(self) -> None:
        """Initialize device manager."""
        if self.initialized:
            return
            
        logger.info("Initializing device manager")
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.initialized = True
        
    def check_resource_availability(self) -> bool:
        """Check if resources are available.
        
        Returns:
            True if resources available
        """
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.resource_limits['max_cpu_usage'] * 100:
                logger.warning(f"CPU usage too high: {cpu_percent}%")
                return False
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.resource_limits['max_memory_usage'] * 100:
                logger.warning(f"Memory usage too high: {memory.percent}%")
                return False
                
            # Check GPU memory if available
            if self.device == 'cuda':
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory > self.resource_limits['max_memory_usage']:
                    logger.warning(f"GPU memory usage too high: {gpu_memory * 100}%")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error checking resources: {str(e)}")
            return False
            
    def get_device(self) -> str:
        """Get current device.
        
        Returns:
            Device name
        """
        return self.device
        
    def set_device(self, device: str) -> None:
        """Set device.
        
        Args:
            device: Device name
        """
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {device}")
            
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        self.device = device
        logger.info(f"Device set to: {device}")
        
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()