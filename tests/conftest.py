"""Shared fixtures for tests."""

import os
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.quality_management.quality_manager import QualityManager

@pytest.fixture
def mock_model() -> nn.Module:
    """Create mock PyTorch model."""
    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            self.model_params = {}
            self.version = "1.0.0"
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)
            
        def process(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {"image": self.forward(x)}
            
        def update_parameters(self, params: Dict[str, Any]) -> None:
            self.model_params.update(params)
            
        def _validate_params(self, params: Dict[str, Any]) -> bool:
            return True
    
    model = MockModel()
    # Initialize parameters to ensure they exist
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    return model

@pytest.fixture
def test_image() -> np.ndarray:
    """Create test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # Add white square
    return img

@pytest.fixture
def processed_image() -> np.ndarray:
    """Create processed test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 200  # Add lighter square
    return img

@pytest.fixture
def quality_manager() -> QualityManager:
    """Create quality manager instance."""
    manager = QualityManager()
    manager.initialize()
    return manager
