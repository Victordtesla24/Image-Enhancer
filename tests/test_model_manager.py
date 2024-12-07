"""Test suite for model manager."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.model_management.model_manager import ModelManager


@pytest.fixture
def model_manager(tmp_path):
    """Create model manager with temporary directory."""
    return ModelManager(models_dir=str(tmp_path))


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = MagicMock()
    model.process = MagicMock(return_value={"image": torch.zeros(1, 3, 64, 64)})
    model.update_parameters = MagicMock()
    model._validate_params = MagicMock()
    model.model_params = {}
    model.version = "1.0.0"
    return model


def test_model_loading(model_manager, tmp_path):
    """Test model loading."""
    # Create test model
    model = torch.nn.Linear(10, 10)
    model_path = tmp_path / "test_model.pt"
    torch.save(model, model_path)

    # Test loading
    loaded = model_manager.load_model(model_path)
    assert isinstance(loaded, torch.nn.Module)


def test_model_saving(model_manager, tmp_path):
    """Test model saving."""
    # Create test model
    model = torch.nn.Linear(10, 10)
    save_path = tmp_path / "test_model.pt"

    # Test saving
    model_manager.save_model(model, save_path)
    assert save_path.exists()


def test_model_validation(model_manager, mock_model):
    """Test model validation."""
    # Test valid model
    assert model_manager.validate_model(mock_model)

    # Test None model
    assert not model_manager.validate_model(None)

    # Test invalid model (no required methods)
    invalid_model = MagicMock()
    invalid_model._mock_return_value = None
    invalid_model.forward = None
    assert not model_manager.validate_model(invalid_model)

    # Test model with required methods but no parameters
    mock_model_no_params = MagicMock()
    mock_model_no_params._mock_return_value = None
    mock_model_no_params.forward = MagicMock()
    mock_model_no_params.parameters = lambda: iter([])
    assert not model_manager.validate_model(mock_model_no_params)


def test_model_optimization(model_manager, mock_model):
    """Test model optimization."""
    # Test optimization
    optimized = model_manager.optimize_model(mock_model)
    assert optimized is not None


def test_model_versioning(model_manager, mock_model):
    """Test model versioning."""
    version = model_manager.get_model_version(mock_model)
    assert isinstance(version, str)
    assert version == "1.0.0"


def test_model_metadata(model_manager, mock_model):
    """Test model metadata."""
    metadata = model_manager.get_model_metadata(mock_model)
    assert isinstance(metadata, dict)
    assert "version" in metadata
    assert "parameters" in metadata


def test_model_cleanup(model_manager, mock_model):
    """Test model cleanup."""
    # Add mock model
    model_manager.models["test"] = mock_model

    # Test cleanup
    model_manager.cleanup()
    assert len(model_manager.loaded_models) == 0
