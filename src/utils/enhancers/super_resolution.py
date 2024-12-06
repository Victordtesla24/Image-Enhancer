"""Super resolution enhancement module using advanced techniques"""

import cv2
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.processor import CoreProcessor

logger = logging.getLogger(__name__)

class SuperResolutionNet(nn.Module):
    """Custom super resolution network"""
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class SuperResolutionEnhancer:
    """Handles super resolution enhancement"""

    def __init__(self):
        self.core = CoreProcessor()
        self.config = self.core.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SuperResolutionNet().to(self.device)
        logger.info("Super resolution enhancer initialized")

    def enhance(self, img, target_width):
        """Apply super resolution enhancement"""
        try:
            logger.info("Applying super resolution")

            # Convert to BGR for OpenCV processing
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            current_height, current_width = img.shape[:2]
            scale_factor = target_width / current_width
            target_height = int(current_height * scale_factor)

            logger.info(f"Scaling from {current_width}x{current_height} to {target_width}x{target_height}")

            # Enhanced multi-step upscaling
            steps = []
            current_scale = 1.0
            while current_scale < scale_factor:
                if scale_factor - current_scale > 2:
                    step_scale = 2.0
                else:
                    step_scale = scale_factor / current_scale
                steps.append(step_scale)
                current_scale *= step_scale

            # Process each upscaling step
            for step_scale in steps:
                # Apply bilateral filter for edge-preserving smoothing
                img = cv2.bilateralFilter(img, 5, 75, 75)

                # Calculate intermediate size
                intermediate_width = int(current_width * step_scale)
                intermediate_height = int(current_height * step_scale)

                # Convert to tensor for model processing
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
                img_tensor = img_tensor.to(self.device) / 255.0

                # Apply model enhancement
                with torch.no_grad():
                    enhanced = self.model(img_tensor)
                
                # Convert back to numpy
                enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)

                # Upscale using Lanczos
                img = cv2.resize(
                    enhanced,
                    (intermediate_width, intermediate_height),
                    interpolation=cv2.INTER_LANCZOS4
                )

                # Apply adaptive sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) / 9.0
                img = cv2.filter2D(img, -1, kernel)

                # Apply detail-preserving denoising
                img = cv2.fastNlMeansDenoisingColored(
                    img,
                    None,
                    h=3,
                    hColor=3,
                    templateWindowSize=7,
                    searchWindowSize=21
                )

                current_width = intermediate_width
                current_height = intermediate_height

            # Final refinement
            img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            
            # Ensure exact target size
            img = cv2.resize(
                img,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4
            )

            # Convert back to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("Super resolution completed")
            return img

        except Exception as e:
            logger.error(f"Error in super resolution: {str(e)}")
            raise
