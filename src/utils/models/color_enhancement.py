"""Color Enhancement Model with professional color processing capabilities"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
import logging
from ..core.base_model import AIModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorEnhancementModel(AIModel):
    """Color Enhancement Model with professional color processing"""

    def __init__(self):
        super().__init__(
            "ColorEnhance", "Professional Color Enhancement with natural preservation"
        )

    def load(self):
        try:
            # Improved color processing layers
            self.color_enhance = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

            # Initialize with improved weights
            for m in self.color_enhance.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            self.color_enhance.to(self.device)
            self.color_enhance.eval()
            self.loaded = True

        except Exception as e:
            logger.error(f"Error loading Color Enhancement model: {str(e)}")
            raise

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                # Process in batches for large images
                if image.shape[2] * image.shape[3] > 2048 * 2048:
                    return self._batch_process(image)

                # Store original image statistics per channel
                orig_mean_channels = torch.mean(image, dim=(2, 3), keepdim=True)
                orig_std_channels = torch.std(image, dim=(2, 3), keepdim=True)

                # Initial enhancement
                enhanced = self.color_enhance(image)

                # Strong blend with original to maintain natural look
                enhanced = enhanced * 0.3 + image * 0.7

                # Calculate luminance
                luminance = (
                    0.299 * image[:, 0:1]
                    + 0.587 * image[:, 1:2]
                    + 0.114 * image[:, 2:3]
                )
                luminance = luminance.repeat(1, 3, 1, 1)

                # Very gentle brightness adjustment
                mean_luminance = torch.mean(luminance)
                if mean_luminance < 0.4:  # Dark image
                    enhanced = enhanced * torch.clamp(1.1 - luminance * 0.1, 0.95, 1.05)
                elif mean_luminance > 0.6:  # Bright image
                    enhanced = enhanced * torch.clamp(1.0 + luminance * 0.1, 0.95, 1.05)

                # Local contrast enhancement
                kernel_size = 5
                padding = kernel_size // 2

                local_mean = TF.avg_pool2d(
                    enhanced, kernel_size, stride=1, padding=padding
                )
                local_var = (
                    TF.avg_pool2d(enhanced**2, kernel_size, stride=1, padding=padding)
                    - local_mean**2
                )
                local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))

                normalized = (enhanced - local_mean) / (local_std + 1e-6)
                enhanced = (
                    local_mean + local_std * normalized * 1.05
                )  # Very gentle contrast

                # Process each channel separately to maintain color accuracy
                for c in range(3):
                    channel = enhanced[:, c : c + 1]
                    orig_mean = orig_mean_channels[:, c : c + 1]
                    orig_std = orig_std_channels[:, c : c + 1]

                    # Normalize and restore original statistics
                    channel_mean = torch.mean(channel, dim=(2, 3), keepdim=True)
                    channel_std = torch.std(channel, dim=(2, 3), keepdim=True)

                    channel = (channel - channel_mean) / (channel_std + 1e-6)
                    channel = channel * orig_std + orig_mean

                    enhanced[:, c : c + 1] = channel

                # Final blend with original for maximum preservation
                result = enhanced * 0.6 + image * 0.4

                # Ensure values are in valid range
                result = torch.clamp(result, 0.0, 1.0)

                return result

            except Exception as e:
                logger.error(f"Error in Color enhancement: {str(e)}")
                return image
