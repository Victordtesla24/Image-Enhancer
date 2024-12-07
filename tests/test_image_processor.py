"""Test suite for image processor."""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.image_processor import ImageProcessor


@pytest.fixture
def test_image():
    """Create test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # Add white square
    return img


@pytest.fixture
def test_output_dir(tmp_path):
    """Create test output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def processor():
    """Create image processor."""
    return ImageProcessor()


def test_load_image(processor, tmp_path):
    """Test image loading."""
    # Create test image
    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

    # Test loading
    loaded = processor.load_image(image_path)
    assert loaded is not None
    assert loaded.shape == (100, 100, 3)


def test_save_image(processor, test_image, test_output_dir):
    """Test image saving."""
    output_path = test_output_dir / "test_output.jpg"

    # Test saving
    processor.save_image(test_image, output_path)
    assert output_path.exists()

    # Test loading saved image
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape == test_image.shape


def test_enhance_image(processor, test_image):
    """Test image enhancement."""
    # Test enhancement
    enhanced, metrics = processor.enhance_image(test_image)
    assert enhanced is not None
    assert enhanced.shape == test_image.shape
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_process_batch(processor, test_output_dir):
    """Test batch processing."""
    # Create test images
    image_paths = []
    for i in range(3):
        path = test_output_dir / f"test_{i}.jpg"
        cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
        image_paths.append(path)

    # Test batch processing
    results = processor.process_batch(image_paths, test_output_dir)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)


def test_quality_metrics(processor, test_image):
    """Test quality metrics calculation."""
    metrics = processor.calculate_quality_metrics(test_image)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_model_parameters(processor):
    """Test model parameter setting."""
    model_name = processor.get_supported_models()[0]
    params = {"param1": 1.0, "param2": 2.0}

    processor.set_model_parameters(model_name, params)

    # Verify parameters were set
    assert processor.model_manager.models[model_name].model_params == params


def test_error_handling(processor, test_output_dir):
    """Test error handling."""
    # Test invalid image path
    with pytest.raises(FileNotFoundError):
        processor.load_image("invalid.jpg")

    # Test invalid output path
    with pytest.raises(Exception):
        processor.save_image(np.zeros((100, 100, 3)), "/invalid/path/test.jpg")

    # Test invalid model name
    with pytest.raises(ValueError):
        processor.set_model_parameters("invalid_model", {})
