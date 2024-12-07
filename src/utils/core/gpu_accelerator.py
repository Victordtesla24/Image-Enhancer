"""GPU acceleration utilities."""

import logging
import torch
import warnings
import os

# Suppress all torch warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT to prevent path warnings

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Handles GPU acceleration for image processing."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(GPUAccelerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the accelerator if not already initialized."""
        if not self._initialized:
            self._initialized = True
            # Initialize device without logging
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    def get_device(self):
        """Get the current device."""
        return self.device
        
    def to_device(self, tensor):
        """Move tensor to device."""
        return tensor.to(self.device)
        
    def from_device(self, tensor):
        """Move tensor from device to CPU."""
        return tensor.cpu()
        
    def is_gpu_available(self):
        """Check if GPU is available."""
        return torch.cuda.is_available()
        
    def get_device_properties(self):
        """Get properties of current device."""
        if self.is_gpu_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'memory': {
                    'allocated': torch.cuda.memory_allocated(0),
                    'cached': torch.cuda.memory_reserved(0)
                }
            }
        return {'name': 'CPU', 'capability': None, 'memory': None}
