"""Super Resolution Enhancer with GPU acceleration"""

import torch
import numpy as np
from typing import Dict
import logging
from ..core.gpu_accelerator import GPUAccelerator
from ..models.super_resolution import SuperResolutionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperResolutionEnhancer:
    """GPU-accelerated super resolution enhancement"""
    
    def __init__(self):
        """Initialize the enhancer with GPU acceleration"""
        self.gpu_accelerator = GPUAccelerator()
        self.device = self.gpu_accelerator.device_manager.get_device()
        self.model = SuperResolutionModel()
        self.model.load()  # Load model weights
        
    def enhance(self, img: np.ndarray, target_width: int) -> np.ndarray:
        """
        Enhance image resolution using GPU acceleration.
        
        Args:
            img: Input image as numpy array (H, W, C)
            target_width: Target width for super resolution
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Calculate memory requirements (input + processing buffer + output)
            required_memory = img.nbytes * 4  # Buffer for processing
            
            # Allocate GPU resources
            resources = self.gpu_accelerator.allocate_resources({
                "task_id": "super_res",
                "memory": required_memory
            })
            
            if resources["status"] != "allocated":
                logger.error(f"Failed to allocate GPU resources: {resources}")
                raise RuntimeError("GPU resource allocation failed")
            
            # Convert to tensor and optimize
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            operations = [img_tensor]
            optimized_ops = self.gpu_accelerator.optimize_compute(operations)
            
            # Process with model
            with torch.no_grad():
                enhanced = self.model.enhance(optimized_ops[0])
            
            # Convert back to numpy array
            result = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Release GPU resources
            self.gpu_accelerator.release_resources("super_res")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in super resolution enhancement: {str(e)}")
            # Ensure resources are released even if an error occurs
            self.gpu_accelerator.release_resources("super_res")
            raise
            
    def _validate_input(self, img: np.ndarray) -> bool:
        """
        Validate input image.
        
        Args:
            img: Input image to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(img, np.ndarray):
            return False
        if len(img.shape) != 3:
            return False
        if img.shape[2] != 3:  # Must be RGB
            return False
        return True
