"""Base AI model class definition"""

import gc
import torch
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModel:
    """Base class for AI models"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        """Enhance image - to be implemented by subclasses"""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup model resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "feature_extractor"):
            del self.feature_extractor
        torch.cuda.empty_cache()
        gc.collect()

    def _batch_process(
        self, image: torch.Tensor, section_height: int = 1024
    ) -> torch.Tensor:
        """Process large images in batches"""
        _, _, H, W = image.shape
        sections = []

        for i in range(0, H, section_height):
            section = image[:, :, i : min(i + section_height, H), :]
            enhanced_section = self.enhance(section)
            sections.append(enhanced_section)

        return torch.cat(sections, dim=2)
