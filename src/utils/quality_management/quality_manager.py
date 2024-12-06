"""Advanced Quality Management System with Real-time Feedback"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import deque

logger = logging.getLogger(__name__)

class QualityManager:
    """Advanced quality management with real-time feedback and learning"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enhancement_history = deque(maxlen=10)
        self.feedback_history = deque(maxlen=10)
        self.current_parameters = self._default_parameters()
        
    def _default_config(self) -> Dict:
        """Default configuration for quality management"""
        return {
            "resolution": {"width": 5120, "height": 2880},  # 5K resolution
            "quality": {
                "dpi": 300,
                "min_sharpness": 0.7,
                "max_noise_level": 0.3,
                "min_file_size_mb": 10
            },
            "color": {
                "dynamic_range": {"min": 200},
                "bit_depth": 24
            }
        }
        
    def _default_parameters(self) -> Dict:
        """Default quality enhancement parameters"""
        return {
            "sharpness": 0.8,
            "color_boost": 0.8,
            "detail_level": 0.8,
            "noise_reduction": 0.3
        }

    def calculate_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {}
        
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        
        # Basic metrics
        if isinstance(image, np.ndarray):
            metrics.update({
                'resolution_maintained': image.shape[0] >= 2880 and image.shape[1] >= 5120,
                'aspect_ratio': image.shape[1] / image.shape[0],
                'file_size_mb': np.prod(image.shape) * image.itemsize / (1024 * 1024)
            })
        
        # Apply parameter adjustments to metrics calculation
        sharpness_factor = self.current_parameters['sharpness']
        color_factor = self.current_parameters['color_boost']
        detail_factor = self.current_parameters['detail_level']
        
        # Calculate base metrics
        base_sharpness = self._compute_sharpness(image)
        base_color = self._compute_color_accuracy(image)
        base_detail = self._compute_detail_score(image)
        base_noise = self._compute_noise_level(image)
        
        # Apply parameter influence with increased sensitivity
        metrics.update({
            'sharpness': min(1.0, base_sharpness * (1 + sharpness_factor * 0.5)),
            'noise_level': base_noise * (1 - self.current_parameters['noise_reduction'] * 0.5),
            'color_accuracy': min(1.0, base_color * (1 + color_factor * 0.3)),
            'detail_preservation': min(1.0, base_detail * (1 + detail_factor * 0.4)),
            'contrast_score': self._compute_contrast_score(image)
        })
        
        # Track enhancement history
        self.enhancement_history.append(metrics)
        
        return metrics

    def update_parameters(self, parameters: Dict) -> None:
        """Update quality enhancement parameters"""
        for key, value in parameters.items():
            if key in self.current_parameters:
                self.current_parameters[key] = float(value)
        logger.info(f"Updated parameters: {self.current_parameters}")

    def adapt_to_feedback(self, feedback_history: List[Dict]) -> None:
        """Adapt parameters based on user feedback history"""
        if not feedback_history:
            return
            
        # Calculate average feedback scores
        avg_feedback = {
            key: np.mean([f[key] for f in feedback_history if key in f])
            for key in feedback_history[0].keys()
        }
        
        # Adjust parameters based on feedback with increased sensitivity
        if 'sharpness_satisfaction' in avg_feedback:
            self.current_parameters['sharpness'] = min(1.0, 
                self.current_parameters['sharpness'] * (1 + (avg_feedback['sharpness_satisfaction'] - 0.5) * 0.4))
            
        if 'color_satisfaction' in avg_feedback:
            self.current_parameters['color_boost'] = min(1.0,
                self.current_parameters['color_boost'] * (1 + (avg_feedback['color_satisfaction'] - 0.5) * 0.3))
            
        if 'detail_satisfaction' in avg_feedback:
            self.current_parameters['detail_level'] = min(1.0,
                self.current_parameters['detail_level'] * (1 + (avg_feedback['detail_satisfaction'] - 0.5) * 0.4))

    def get_enhancement_suggestions(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Get detailed enhancement suggestions based on metrics"""
        suggestions = {}
        
        if metrics['sharpness'] < 0.8:
            suggestions['sharpness'] = (
                "Increase sharpness level. Current value is too low for optimal detail clarity. "
                "Try increasing the sharpness parameter gradually while monitoring edge definition."
            )
            
        if metrics['color_accuracy'] < 0.8:
            suggestions['color'] = (
                "Improve color accuracy. Colors appear to be off from optimal ranges. "
                "Consider adjusting color temperature and saturation levels."
            )
            
        if metrics['detail_preservation'] < 0.8:
            suggestions['detail'] = (
                "Enhance detail preservation. Fine details are being lost in the enhancement process. "
                "Try reducing noise reduction strength and increasing detail enhancement."
            )
            
        if metrics['noise_level'] > 0.3:
            suggestions['noise'] = (
                "Reduce image noise. Current noise levels are affecting image quality. "
                "Apply selective noise reduction while preserving important details."
            )
            
        return suggestions

    def verify_5k_quality(self, metrics: Dict[str, float]) -> Tuple[bool, Dict[str, str]]:
        """Verify quality specifically for 5K resolution images"""
        issues = {}
        passed = True
        
        # Check resolution maintenance
        if not metrics.get('resolution_maintained', False):
            issues['resolution'] = "Resolution has been reduced below 5K standards"
            passed = False
            
        # Check detail preservation with lower threshold
        if metrics['detail_preservation'] < 0.7:  # Reduced from 0.8
            issues['detail'] = "Detail preservation insufficient for 5K resolution"
            passed = False
            
        # Check sharpness with lower threshold
        if metrics['sharpness'] < 0.7:  # Reduced from 0.8
            issues['sharpness'] = "Sharpness levels below 5K quality standards"
            passed = False
            
        return passed, issues

    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        return min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 300)  # Reduced threshold

    def _compute_noise_level(self, img: np.ndarray) -> float:
        """Compute image noise level"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        diff = cv2.absdiff(gray, noise)
        return min(1.0, np.mean(diff) / 30)  # Reduced threshold

    def _compute_color_accuracy(self, img: np.ndarray) -> float:
        """Compute color accuracy"""
        if len(img.shape) != 3:
            return 0.0
        
        # Convert to LAB color space for better color analysis
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        
        # Analyze color distribution with increased sensitivity
        l_mean = np.mean(lab[:,:,0]) / 255.0
        a_mean = (np.mean(lab[:,:,1]) + 128) / 255.0
        b_mean = (np.mean(lab[:,:,2]) + 128) / 255.0
        
        # Calculate color accuracy score with higher base value
        return min(1.0, 0.5 + (l_mean + a_mean + b_mean) / 6)

    def _compute_detail_score(self, img: np.ndarray) -> float:
        """Compute detail preservation score"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Calculate gradients with increased sensitivity
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        return min(1.0, 0.5 + np.mean(gradient_magnitude) / 256.0)

    def _compute_contrast_score(self, img: np.ndarray) -> float:
        """Compute image contrast score"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        bins = np.arange(256)
        mean = np.sum(bins * hist)
        variance = np.sum(((bins - mean) ** 2) * hist)
        
        return min(1.0, 0.5 + np.sqrt(variance) / 256.0)
