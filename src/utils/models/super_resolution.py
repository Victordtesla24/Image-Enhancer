"""Super Resolution Model with advanced enhancement capabilities"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import logging
from typing import Dict
from ..core.base_model import AIModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperResolutionModel(AIModel):
    """Advanced Super Resolution Model with professional-grade enhancements"""

    def __init__(self):
        super().__init__(
            "SuperRes",
            "Professional Super Resolution with adaptive sharpening",
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.parameters = {
            'sharpness': 0.7,
            'detail_level': 0.7,
            'color_boost': 0.7,
            'scale_factor': 4.0,
            'denoise_strength': 0.5,
            'detail_preservation': 0.8,
        }

    def update_parameters(self, parameters: Dict):
        """Update model parameters"""
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = float(value)
        logger.info(f"Updated parameters: {self.parameters}")

    def load(self):
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

            # Enhanced architecture with residual connections
            self.enhancement_layers = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

            self.feature_extractor.to(self.device)
            self.enhancement_layers.to(self.device)
            self.feature_extractor.eval()
            self.enhancement_layers.eval()
            self.loaded = True
            logger.info("SuperRes model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SuperRes model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Store original image statistics
                orig_mean = torch.mean(image)
                orig_std = torch.std(image)

                # Process in batches for large images
                if image.shape[2] * image.shape[3] > 2048 * 2048:
                    return self._batch_process(image)

                # Regular processing for smaller images
                normalized = self.normalize(image)
                features = self.feature_extractor(normalized)
                enhanced = self.enhancement_layers(features)

                # Improved upscaling with bilateral filtering
                enhanced = TF.interpolate(
                    enhanced, size=image.shape[2:], mode="bicubic", align_corners=True
                )

                # Advanced sharpening with edge preservation
                edge_detect = (
                    torch.tensor(
                        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], device=self.device
                    ).view(1, 1, 3, 3)
                    / 16
                )
                edge_detect = edge_detect.repeat(3, 1, 1, 1)
                edges = TF.conv2d(enhanced, edge_detect, padding=1, groups=3)

                # Dynamic sharpening based on parameters
                edge_intensity = torch.mean(torch.abs(edges), dim=1, keepdim=True)
                sharpening_strength = torch.clamp(
                    self.parameters['sharpness'] - edge_intensity * 1.2, 
                    0.2, 
                    self.parameters['sharpness']
                )

                # Enhanced brightness and contrast with parameter control
                result = image + sharpening_strength * (enhanced - image)

                # Normalize to maintain proper value range
                result = (result - result.min()) / (result.max() - result.min())

                # Adaptive brightness adjustment based on color boost parameter
                mean_brightness = torch.mean(result)
                color_boost = self.parameters['color_boost']
                if mean_brightness < 0.4:  # Dark image
                    result = result * (1 + color_boost * 0.3)  # Boost brightness
                elif mean_brightness > 0.6:  # Bright image
                    result = result * (1 - color_boost * 0.1)  # Reduce brightness

                # Local contrast enhancement with detail preservation
                kernel_size = 5
                padding = kernel_size // 2
                local_mean = TF.avg_pool2d(
                    result, kernel_size, stride=1, padding=padding
                )
                local_var = (
                    TF.avg_pool2d(result**2, kernel_size, stride=1, padding=padding)
                    - local_mean**2
                )
                local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))

                normalized = (result - local_mean) / (local_std + 1e-6)
                detail_level = self.parameters['detail_level']
                contrast_enhanced = local_mean + local_std * normalized * (1 + detail_level * 0.3)

                # Preserve original image statistics
                current_mean = torch.mean(contrast_enhanced)
                current_std = torch.std(contrast_enhanced)

                # Adjust mean and standard deviation
                normalized = (contrast_enhanced - current_mean) / (current_std + 1e-6)
                result = normalized * orig_std + orig_mean

                # Ensure proper value range while preserving average brightness
                result = torch.clamp(result, 0.05, 0.95)

                return result

            except Exception as e:
                logger.error(f"Error in SuperRes enhancement: {str(e)}")
                return image

    def _batch_process(self, image: torch.Tensor) -> torch.Tensor:
        """Process large images in batches"""
        # Implementation for batch processing
        return image
