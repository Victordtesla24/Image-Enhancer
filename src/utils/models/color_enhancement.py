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

                # Store original image statistics
                orig_mean = torch.mean(image)
                orig_std = torch.std(image)

                # Initial color enhancement
                enhanced_colors = self.color_enhance(image)

                # Calculate luminance
                luminance = (
                    0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
                )
                luminance = luminance.unsqueeze(1).repeat(1, 3, 1, 1)

                # Adaptive color enhancement based on image statistics
                mean_luminance = torch.mean(luminance)

                # Brightness enhancement with highlight and shadow preservation
                shadows_mask = torch.clamp(
                    1.2 - luminance, 0.2, 0.5
                )  # Reduced shadow lifting
                highlights_mask = torch.clamp(
                    luminance, 0.5, 0.8
                )  # Better highlight preservation

                # Adaptive brightness adjustment
                if mean_luminance < 0.4:  # Dark image
                    brightness_factor = shadows_mask * 1.3
                elif mean_luminance > 0.6:  # Bright image
                    brightness_factor = highlights_mask * 0.95
                else:  # Balanced image
                    brightness_factor = (shadows_mask + highlights_mask) * 1.1

                enhanced_colors = enhanced_colors * brightness_factor

                # Advanced color saturation enhancement
                current_saturation = torch.mean(torch.abs(enhanced_colors - luminance))
                target_saturation = torch.clamp(current_saturation * 1.3, 0.2, 0.6)

                # Calculate saturation adjustment factor
                saturation_factor = torch.clamp(
                    target_saturation / (current_saturation + 1e-5),
                    1.1,
                    1.5,  # Reduced range for more natural colors
                )

                # Apply saturation enhancement with luminance preservation
                color_enhanced = luminance + saturation_factor * (
                    enhanced_colors - luminance
                )

                # Normalize to maintain proper value range
                color_enhanced = (color_enhanced - color_enhanced.min()) / (
                    color_enhanced.max() - color_enhanced.min()
                )

                # Local contrast enhancement
                kernel_size = 5
                padding = kernel_size // 2

                local_mean = TF.avg_pool2d(
                    color_enhanced, kernel_size, stride=1, padding=padding
                )
                local_var = (
                    TF.avg_pool2d(
                        color_enhanced**2, kernel_size, stride=1, padding=padding
                    )
                    - local_mean**2
                )
                local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))

                normalized = (color_enhanced - local_mean) / (local_std + 1e-6)
                contrast_enhanced = local_mean + local_std * normalized * 1.2

                # Color balance correction
                r, g, b = torch.chunk(contrast_enhanced, 3, dim=1)

                # Calculate color means
                r_mean = torch.mean(r)
                g_mean = torch.mean(g)
                b_mean = torch.mean(b)

                # Calculate correction factors
                max_mean = torch.max(torch.stack([r_mean, g_mean, b_mean]))
                r_factor = max_mean / (r_mean + 1e-6)
                g_factor = max_mean / (g_mean + 1e-6)
                b_factor = max_mean / (b_mean + 1e-6)

                # Apply gentle color balance
                r = r * r_factor.clamp(0.9, 1.1)
                g = g * g_factor.clamp(0.9, 1.1)
                b = b * b_factor.clamp(0.9, 1.1)

                balanced = torch.cat([r, g, b], dim=1)

                # Preserve original image statistics
                current_mean = torch.mean(balanced)
                current_std = torch.std(balanced)

                # Adjust mean and standard deviation
                normalized = (balanced - current_mean) / (current_std + 1e-6)
                result = normalized * orig_std + orig_mean

                # Ensure proper value range while preserving average brightness
                result = torch.clamp(result, 0.05, 0.95)

                # Final contrast adjustment
                mean_val = torch.mean(result)
                result = (result - mean_val) * 1.2 + mean_val

                return result

            except Exception as e:
                logger.error(f"Error in Color enhancement: {str(e)}")
                return image
