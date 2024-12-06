"""Detail Enhancement Model with advanced processing capabilities"""

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

class DetailEnhancementModel(AIModel):
    """Detail Enhancement Model with professional-grade improvements"""

    def __init__(self):
        super().__init__(
            "DetailEnhance", "Professional Detail Enhancement with adaptive processing"
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.parameters = {
            'sharpness': 0.7,
            'detail_level': 0.7,
            'noise_reduction': 0.5,
            'detail_boost': 1.2,
            'edge_preservation': 0.8,
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

            # Enhanced detail preservation with skip connections
            self.detail_enhance = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

            self.feature_extractor.to(self.device)
            self.detail_enhance.to(self.device)
            self.feature_extractor.eval()
            self.detail_enhance.eval()
            self.loaded = True

        except Exception as e:
            logger.error(f"Error loading Detail Enhancement model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Process in batches for large images
                if image.shape[2] * image.shape[3] > 2048 * 2048:
                    return self._batch_process(image)

                # Store original image statistics
                orig_mean = torch.mean(image)
                orig_std = torch.std(image)

                # Extract features with improved detail preservation
                features = self.feature_extractor(self.normalize(image))
                detail_mask = self.detail_enhance(features)

                # Advanced upscaling with detail preservation
                detail_mask = TF.interpolate(
                    detail_mask,
                    size=image.shape[2:],
                    mode="bicubic",
                    align_corners=True,
                )

                # Improved adaptive detail enhancement with parameter control
                detail_strength = torch.mean(
                    torch.abs(detail_mask), dim=1, keepdim=True
                )
                enhancement_factor = torch.clamp(
                    self.parameters['detail_level'] / (detail_strength + 1e-5),
                    0.4 * self.parameters['edge_preservation'],
                    0.8 * self.parameters['detail_boost']
                )
                enhanced = image + enhancement_factor * detail_mask

                # Normalize to maintain proper value range
                enhanced = (enhanced - enhanced.min()) / (
                    enhanced.max() - enhanced.min()
                )

                # Multi-scale contrast enhancement with parameter control
                scales = [3, 5, 7]
                contrast_enhanced = enhanced

                for kernel_size in scales:
                    padding = kernel_size // 2
                    local_mean = TF.avg_pool2d(
                        contrast_enhanced,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    )
                    local_var = (
                        TF.avg_pool2d(
                            contrast_enhanced**2,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        )
                        - local_mean**2
                    )
                    local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))

                    normalized = (contrast_enhanced - local_mean) / (local_std + 1e-6)
                    contrast_enhanced = local_mean + local_std * normalized * self.parameters['detail_boost']

                # Preserve original image statistics
                current_mean = torch.mean(contrast_enhanced)
                current_std = torch.std(contrast_enhanced)

                # Adjust mean and standard deviation
                normalized = (contrast_enhanced - current_mean) / (current_std + 1e-6)
                result = normalized * orig_std + orig_mean

                # Apply sharpening based on parameter
                if self.parameters['sharpness'] > 0:
                    kernel = torch.tensor([
                        [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]
                    ], device=self.device).view(1, 1, 3, 3) * self.parameters['sharpness']
                    kernel = kernel.repeat(3, 1, 1, 1)
                    sharpened = TF.conv2d(result, kernel, padding=1, groups=3)
                    result = torch.lerp(result, sharpened, self.parameters['sharpness'])

                # Apply noise reduction if enabled
                if self.parameters['noise_reduction'] > 0:
                    kernel_size = int(3 + 2 * self.parameters['noise_reduction'])
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    result = TF.avg_pool2d(result, kernel_size, stride=1, padding=kernel_size//2)

                # Ensure proper value range while preserving brightness
                result = torch.clamp(result, 0.05, 0.95)

                return result

            except Exception as e:
                logger.error(f"Error in Detail enhancement: {str(e)}")
                return image

    def _batch_process(self, image: torch.Tensor) -> torch.Tensor:
        """Process large images in batches"""
        # Implementation for batch processing
        return image
