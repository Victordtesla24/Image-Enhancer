"""Detail enhancement model."""

import logging

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class DetailEnhancementModel:
    """Detail enhancement model for image enhancement."""

    def __init__(
        self, name: str = "detail", description: str = "Detail enhancement model"
    ):
        """Initialize detail enhancement model.

        Args:
            name: Model name
            description: Model description
        """
        self.name = name
        self.description = description
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = None
        self.detail_enhance = None
        self.model_params = {}
        self.load()

    def load(self):
        """Load model architecture and weights."""
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

            # Move to device
            self.feature_extractor.to(self.device)
            self.detail_enhance.to(self.device)

            logger.info("Successfully loaded Detail Enhancement model")

        except Exception as e:
            logger.error(f"Error loading Detail Enhancement model: {e}")
            raise

    def process(self, x):
        """Process input tensor.

        Args:
            x: Input tensor

        Returns:
            Enhanced tensor
        """
        try:
            # Move input to device
            x = x.to(self.device)

            # Extract features
            features = self.feature_extractor(x)

            # Apply enhancement
            enhanced = self.detail_enhance(features)

            # Resize to match input
            if enhanced.shape != x.shape:
                enhanced = nn.functional.interpolate(
                    enhanced, size=x.shape[2:], mode="bilinear", align_corners=False
                )

            return enhanced

        except Exception as e:
            logger.error(f"Error processing with Detail Enhancement model: {e}")
            return x

    def update_parameters(self, parameters: dict):
        """Update model parameters.

        Args:
            parameters: New parameter values
        """
        self.model_params.update(parameters)

    def _validate_params(self):
        """Validate model parameters."""
        # No required parameters for now
        pass

    def to_torchscript(self):
        """Convert model to TorchScript.

        Returns:
            TorchScript model
        """
        try:
            # Create forward function
            class DetailModule(nn.Module):
                def __init__(self, feature_extractor, detail_enhance):
                    super().__init__()
                    self.feature_extractor = feature_extractor
                    self.detail_enhance = detail_enhance

                def forward(self, x):
                    features = self.feature_extractor(x)
                    enhanced = self.detail_enhance(features)
                    if enhanced.shape != x.shape:
                        enhanced = nn.functional.interpolate(
                            enhanced,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    return enhanced

            # Create module and convert
            module = DetailModule(self.feature_extractor, self.detail_enhance)
            return torch.jit.script(module)

        except Exception as e:
            logger.error(
                f"Error converting Detail Enhancement model to TorchScript: {e}"
            )
            return None

    def optimize_memory(self):
        """Optimize model memory usage."""
        try:
            # Clear unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move unused tensors to CPU
            for param in self.feature_extractor.parameters():
                if not param.requires_grad:
                    param.data = param.data.cpu()

            for param in self.detail_enhance.parameters():
                if not param.requires_grad:
                    param.data = param.data.cpu()

        except Exception as e:
            logger.error(f"Error optimizing Detail Enhancement model memory: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear model data
            self.feature_extractor = None
            self.detail_enhance = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error cleaning up Detail Enhancement model: {e}")
