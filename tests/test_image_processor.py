"""Test suite for image processor."""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.utils.image_processor import ImageProcessor

@pytest.fixture
def processor():
    """Create processor instance."""
    proc = ImageProcessor()
    proc.initialize()
    return proc

@pytest.fixture
def test_image():
    """Create test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_load_image(processor):
    """Test loading images."""
    # Test non-existent file
    assert processor.load_image("nonexistent.jpg") is None
    
    # Test invalid file
    with open("invalid.jpg", "w") as f:
        f.write("invalid")
    assert processor.load_image("invalid.jpg") is None
    os.remove("invalid.jpg")

def test_save_image(processor, test_image, tmp_path):
    """Test saving images."""
    # Test saving to read-only directory
    read_only_path = tmp_path / "read_only"
    read_only_path.mkdir()
    read_only_path.chmod(0o444)  # Read-only
    
    with pytest.raises(OSError):
        processor.save_image(test_image, str(read_only_path / "test.jpg"))
        
    # Test saving to writable directory
    output_path = tmp_path / "test.jpg"
    assert processor.save_image(test_image, str(output_path))
    assert output_path.exists()

def test_enhance_image(processor, test_image):
    """Test image enhancement."""
    enhanced = processor.enhance_image(test_image)
    assert enhanced is not None
    assert enhanced.shape == test_image.shape
    assert enhanced.dtype == test_image.dtype
    assert not np.array_equal(enhanced, test_image)

def test_process_batch(processor, test_image):
    """Test batch processing."""
    batch = [test_image] * 3
    results = processor._process_batch(batch)
    assert len(results) == len(batch)
    for result in results:
        assert result is not None
        assert result.shape == test_image.shape

def test_quality_metrics(processor, test_image):
    """Test quality metrics calculation."""
    metrics = processor.calculate_quality_metrics(test_image)
    assert isinstance(metrics, dict)
    assert "sharpness" in metrics
    assert "contrast" in metrics
    assert "noise" in metrics

def test_model_parameters(processor):
    """Test model parameter access."""
    params = processor.get_model_parameters()
    assert isinstance(params, dict)
    assert "batch_size" in params
    assert "quality_threshold" in params
    assert "enhancement_level" in params
