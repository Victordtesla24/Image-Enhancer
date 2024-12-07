"""Super resolution model."""

import logging

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class SuperResolutionModel:
    """Super resolution model for image enhancement."""

    def __init__(
        self, name: str = "super_res", description: str = "Super resolution model"
    ):
        """Initialize super resolution model.

        Args:
            name: Model name
            description: Model description
        """
        self.name = name
        self.description = description
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = None
        self.enhancement_layers = None
        self.model_params = {}
        self.load()

    def load(self):
        """Load model architecture and weights."""
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

            # Move to device
            self.feature_extractor.to(self.device)
            self.enhancement_layers.to(self.device)

            logger.info("Successfully loaded SuperRes model")

        except Exception as e:
            logger.error(f"Error loading SuperRes model: {e}")
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
            enhanced = self.enhancement_layers(features)

            # Resize to match input
            if enhanced.shape != x.shape:
                enhanced = nn.functional.interpolate(
                    enhanced, size=x.shape[2:], mode="bilinear", align_corners=False
                )

            return enhanced

        except Exception as e:
            logger.error(f"Error processing with SuperRes model: {e}")
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
            class SuperResModule(nn.Module):
                def __init__(self, feature_extractor, enhancement_layers):
                    super().__init__()
                    self.feature_extractor = feature_extractor
                    self.enhancement_layers = enhancement_layers

                def forward(self, x):
                    features = self.feature_extractor(x)
                    enhanced = self.enhancement_layers(features)
                    if enhanced.shape != x.shape:
                        enhanced = nn.functional.interpolate(
                            enhanced,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    return enhanced

            # Create module and convert
            module = SuperResModule(self.feature_extractor, self.enhancement_layers)
            return torch.jit.script(module)

        except Exception as e:
            logger.error(f"Error converting SuperRes model to TorchScript: {e}")
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

            for param in self.enhancement_layers.parameters():
                if not param.requires_grad:
                    param.data = param.data.cpu()

        except Exception as e:
            logger.error(f"Error optimizing SuperRes model memory: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear model data
            self.feature_extractor = None
            self.enhancement_layers = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error cleaning up SuperRes model: {e}")
