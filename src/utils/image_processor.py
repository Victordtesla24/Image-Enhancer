"""Main image processor integrating all enhancement systems"""

import logging
import time
import hashlib
import cv2
from PIL import Image
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from .model_management.model_manager import ModelManager
from .session_management.session_manager import SessionManager
from .quality_management.quality_manager import QualityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEnhancer:
    """Main image enhancement coordinator with advanced management systems"""

    def __init__(self, session_id: Optional[str] = None):
        logger.info("Initializing ImageEnhancer")

        # Load configuration
        self.config = self._load_config()

        # Initialize management systems
        self.session_manager = SessionManager(session_id)
        self.model_manager = ModelManager(session_id)
        self.quality_manager = QualityManager()

        # Initialize parameters with balanced values
        self.parameters = {
            'resolution_target': 'hd',
            'quality_preset': 'standard',
            'detail_preservation': 0.85,
            'sharpness': 0.85,
            'color_boost': 0.8,
            'detail_level': 0.85,
            'noise_reduction': 0.75  # Reduced to preserve more detail
        }

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_config(self):
        """Load configuration"""
        return {
            "resolution": {"width": 5120, "height": 2880},
            "quality": {
                "dpi": 300,
                "min_sharpness": 70,
                "max_noise_level": 120,
                "min_file_size_mb": 1.5,
            },
            "color": {"bit_depth": 24, "dynamic_range": {"min": 220, "max": 255}},
        }

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        if isinstance(image, np.ndarray):
            # Ensure float32 and correct shape
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
            
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = np.expand_dims(image, 0)
            
            # Convert to tensor
            tensor = torch.from_numpy(image).to(self.device)
            
            # Ensure channel dimension is second
            if tensor.shape[1] != 3:
                tensor = tensor.permute(0, 3, 1, 2)
            
            return tensor
        return image

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            # Move to CPU and convert to numpy
            array = tensor.detach().cpu().numpy()
            
            # Remove batch dimension
            if array.shape[0] == 1:
                array = array[0]
            
            # Move channels to last dimension
            if array.shape[0] == 3:
                array = np.transpose(array, (1, 2, 0))
            
            return array
        return tensor

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive sharpening"""
        if len(image.shape) == 3:
            # Convert to uint8 for processing
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * self.parameters['sharpness']
            
            # Apply sharpening
            sharpened = cv2.filter2D(img_uint8, -1, kernel)
            
            # Blend with original based on detail preservation parameter
            blend_factor = self.parameters['detail_preservation']
            result = cv2.addWeighted(img_uint8, 1-blend_factor, sharpened, blend_factor, 0)
            
            return result.astype(np.float32) / 255.0
        return image

    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply balanced noise reduction"""
        if len(image.shape) == 3:
            # Convert to uint8 for denoising
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Apply denoising with moderate strength
            h_factor = 10 * self.parameters['noise_reduction']  # Reduced strength
            
            # Apply bilateral filter first for edge preservation
            bilateral = cv2.bilateralFilter(img_uint8, 5, 50, 50)
            
            # Apply non-local means denoising
            denoised = np.zeros_like(bilateral)
            for i in range(3):
                denoised[:,:,i] = cv2.fastNlMeansDenoising(
                    bilateral[:,:,i],
                    None,
                    h=h_factor,
                    templateWindowSize=5,  # Smaller window for detail preservation
                    searchWindowSize=15
                )
            
            # Blend result with original to preserve details
            blend_factor = self.parameters['noise_reduction']
            result = cv2.addWeighted(img_uint8, 1-blend_factor, denoised, blend_factor, 0)
            
            return result.astype(np.float32) / 255.0
        return image

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using all available models"""
        # Convert to tensor
        tensor = self._to_tensor(image)
        enhanced = tensor

        try:
            # Initial noise reduction with detail preservation
            image_np = self._to_numpy(tensor)
            denoised = self._apply_noise_reduction(image_np)
            enhanced = self._to_tensor(denoised)

            # Update model parameters before enhancement
            for model_name in ['super_resolution', 'detail', 'color']:
                self.model_manager.update_parameters(model_name, self.parameters)

            # Apply each enhancement model in sequence with quality checks
            for model_name in ['super_resolution', 'detail', 'color']:
                # Apply enhancement
                current = self.model_manager.enhance(model_name, enhanced)
                
                # Check if enhancement improved quality
                current_np = self._to_numpy(current)
                enhanced_np = self._to_numpy(enhanced)
                
                current_metrics = self.quality_manager.calculate_metrics(current_np)
                previous_metrics = self.quality_manager.calculate_metrics(enhanced_np)
                
                # Keep enhancement if quality improved
                if (current_metrics['sharpness'] >= previous_metrics['sharpness'] * 0.95 and  # Allow slight sharpness decrease
                    current_metrics['color_accuracy'] >= previous_metrics['color_accuracy'] and
                    current_metrics['detail_preservation'] >= previous_metrics['detail_preservation'] and
                    current_metrics['noise_level'] <= previous_metrics['noise_level'] * 1.1):  # Allow slight noise increase
                    enhanced = current
                else:
                    # If quality decreased, try with adjusted parameters
                    adjusted_params = self.parameters.copy()
                    adjusted_params.update({
                        'sharpness': min(1.0, self.parameters['sharpness'] * 1.2),
                        'detail_level': min(1.0, self.parameters['detail_level'] * 1.2),
                        'color_boost': min(1.0, self.parameters['color_boost'] * 1.1),
                        'noise_reduction': max(0.5, self.parameters['noise_reduction'] * 0.9)  # Reduce noise reduction if needed
                    })
                    self.model_manager.update_parameters(model_name, adjusted_params)
                    enhanced = self.model_manager.enhance(model_name, enhanced)

                # Apply sharpening if needed
                enhanced_np = self._to_numpy(enhanced)
                if current_metrics['sharpness'] < previous_metrics['sharpness']:
                    enhanced_np = self._apply_sharpening(enhanced_np)
                    enhanced = self._to_tensor(enhanced_np)

            # Final balanced enhancement
            enhanced_np = self._to_numpy(enhanced)
            
            # Check if final sharpening is needed
            final_metrics = self.quality_manager.calculate_metrics(enhanced_np)
            initial_metrics = self.quality_manager.calculate_metrics(image)
            
            if final_metrics['sharpness'] < initial_metrics['sharpness']:
                # Apply final sharpening with stronger parameters
                temp_params = self.parameters.copy()
                temp_params['sharpness'] = min(1.0, self.parameters['sharpness'] * 1.5)
                temp_params['detail_preservation'] = min(1.0, self.parameters['detail_preservation'] * 1.2)
                self.parameters = temp_params
                enhanced_np = self._apply_sharpening(enhanced_np)
                
                # Light noise reduction if needed
                if final_metrics['noise_level'] > initial_metrics['noise_level']:
                    enhanced_np = self._apply_noise_reduction(enhanced_np)
            
            return enhanced_np

        except Exception as e:
            logger.error(f"Error during enhancement: {str(e)}")
            return image

    def update_parameters(self, parameters: Dict):
        """Update enhancement parameters"""
        self.parameters.update(parameters)
        
        # Update parameters for all models
        for model_name in ['super_resolution', 'detail', 'color']:
            self.model_manager.update_parameters(model_name, parameters)

    def adapt_to_feedback(self, feedback_history: List[Dict]):
        """Adapt enhancement parameters based on feedback"""
        if not feedback_history:
            return
            
        # Calculate average feedback scores
        avg_feedback = {
            key: np.mean([f[key] for f in feedback_history if key in f])
            for key in ['sharpness_satisfaction', 'color_satisfaction', 'detail_satisfaction']
        }
        
        # Update parameters based on feedback with more aggressive adjustments
        if 'sharpness_satisfaction' in avg_feedback:
            self.parameters['sharpness'] = min(1.0, 
                self.parameters['sharpness'] * (1 + (avg_feedback['sharpness_satisfaction'] - 0.5) * 0.3))
            
        if 'color_satisfaction' in avg_feedback:
            self.parameters['color_boost'] = min(1.0,
                self.parameters['color_boost'] * (1 + (avg_feedback['color_satisfaction'] - 0.5) * 0.2))
            
        if 'detail_satisfaction' in avg_feedback:
            self.parameters['detail_level'] = min(1.0,
                self.parameters['detail_level'] * (1 + (avg_feedback['detail_satisfaction'] - 0.5) * 0.3))
            
            # Adjust noise reduction inversely to detail satisfaction
            self.parameters['noise_reduction'] = max(0.5,
                self.parameters['noise_reduction'] * (1 - (avg_feedback['detail_satisfaction'] - 0.5) * 0.2))
        
        # Update model parameters
        for model_name in ['super_resolution', 'detail', 'color']:
            self.model_manager.update_parameters(model_name, self.parameters)

    def _compute_image_hash(self, image: Image.Image) -> str:
        """Compute unique hash for image"""
        return hashlib.md5(np.array(image).tobytes()).hexdigest()

    def enhance_image(
        self,
        input_image: Image.Image,
        target_width: int,
        models: List[str],
        progress_callback=None,
        retry_count: int = 0,
    ) -> Tuple[Image.Image, Dict]:
        """Enhance image using selected models with quality validation"""
        try:
            logger.info(f"Starting image enhancement with models: {models}")
            start_time = time.time()

            # Convert PIL Image to numpy array
            img_array = np.array(input_image)
            if img_array.dtype != np.float32:
                img_array = img_array.astype(np.float32) / 255.0

            # Apply enhancements
            enhanced_array = self.enhance(img_array)

            # Convert back to uint8 for PIL
            enhanced_array = (enhanced_array * 255).astype(np.uint8)
            enhanced_img = Image.fromarray(enhanced_array)
            enhanced_img.info["dpi"] = (300, 300)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare enhancement details
            enhancement_details = {
                "source_size": f"{input_image.size[0]}x{input_image.size[1]}",
                "target_size": f"{enhanced_img.size[0]}x{enhanced_img.size[1]}",
                "processing_time": f"{processing_time:.2f} seconds",
                "parameters": self.parameters,
            }

            return enhanced_img, enhancement_details

        except Exception as e:
            logger.error(f"Error during image enhancement: {str(e)}")
            raise

    def get_quality_preferences(self) -> Dict:
        """Get current quality preferences"""
        return self.parameters.copy()

    def update_quality_preferences(self, preferences: Dict):
        """Update quality preferences"""
        self.update_parameters(preferences)

    def get_enhancement_history(self, image_hash: Optional[str] = None) -> Dict:
        """Get enhancement history"""
        return {
            "history": self.model_manager.models_state,
            "parameters": self.parameters
        }
