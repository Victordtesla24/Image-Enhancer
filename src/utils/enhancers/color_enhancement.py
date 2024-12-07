"""Color enhancement module."""

import logging
from typing import Dict

import torch
import torch.nn as nn

from ..models.color_enhancement import ColorEnhancementModel

logger = logging.getLogger(__name__)


class ColorEnhancer:
    """Color enhancement processor."""

    def __init__(self):
        """Initialize color enhancer."""
        self.model = ColorEnhancementModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = {
            "saturation": 1.2,
            "contrast": 1.1,
            "brightness": 1.0,
            "color_boost": 1.3,
            "color_balance": 0.5,
            "color_preservation": 0.8,
        }

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        """Enhance image colors.

        Args:
            image: Input image tensor

        Returns:
            Enhanced image tensor
        """
        try:
            # Move to device
            image = image.to(self.device)

            # Process image
            enhanced = self.model.process(image)

            return enhanced

        except Exception as e:
            logger.error(f"Error in color enhancement: {e}")
            return image

    def update_parameters(self, parameters: Dict):
        """Update enhancement parameters.

        Args:
            parameters: Parameter dictionary
        """
        self.parameters.update(parameters)
        self.model.update_parameters(parameters)
