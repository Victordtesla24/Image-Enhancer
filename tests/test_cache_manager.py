"""Test suite for cache manager."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.core.cache_manager import CacheManager


@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary cache directory."""
    return tmp_path / "test_cache"


@pytest.fixture
def cache_manager(cache_dir):
    """Create cache manager with test directory."""
    return CacheManager(
        cache_dir=str(cache_dir),
        max_memory_usage=0.001,  # Small memory limit for testing
        cleanup_interval=1,
    )


@pytest.fixture
def test_data():
    """Create test data."""
    return {
        "dict": {"key": "value", "number": 42},
        "array": np.random.rand(64, 64, 3).astype(np.float32),
        "tensor": torch.rand(64, 64, 3),
        "image": Image.new("RGB", (64, 64)),
    }


def test_initialization(cache_manager, cache_dir):
    """Test cache manager initialization."""
    assert cache_manager.cache_dir == Path(cache_dir)
    assert cache_manager.cleanup_interval == 1
    assert cache_manager.max_memory_usage > 0

    # Check directory creation
    assert cache_manager.models_dir.exists()
    assert cache_manager.results_dir.exists()
    assert cache_manager.temp_dir.exists()


def test_model_cache_operations(cache_manager, test_data):
    """Test model cache save and load."""
    model_id = "test_model"
    data = test_data["dict"]

    # Test saving
    assert cache_manager.save_model_cache(model_id, data)

    # Test loading
    loaded = cache_manager.load_model_cache(model_id)
    assert loaded == data

    # Test non-existent model
    assert cache_manager.load_model_cache("nonexistent") is None


def test_result_operations(cache_manager, test_data):
    """Test result save and load operations."""
    # Test dictionary
    assert cache_manager.save_result("dict_result", test_data["dict"])
    loaded_dict = cache_manager.load_result("dict_result")
    assert loaded_dict == test_data["dict"]

    # Test numpy array
    assert cache_manager.save_result("array_result", test_data["array"])
    loaded_array = cache_manager.load_result("array_result")
    np.testing.assert_array_almost_equal(loaded_array, test_data["array"])

    # Test tensor
    assert cache_manager.save_result("tensor_result", test_data["tensor"])
    loaded_tensor = cache_manager.load_result("tensor_result")
    np.testing.assert_array_almost_equal(loaded_tensor, test_data["tensor"].numpy())

    # Test image
    assert cache_manager.save_result("image_result", test_data["image"])
    loaded_image = cache_manager.load_result("image_result")
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.size == test_data["image"].size


def test_memory_tracking(cache_manager, test_data):
    """Test memory usage tracking."""
    # Save multiple items
    for i in range(5):
        cache_manager.save_model_cache(f"model_{i}", test_data["dict"])

    # Check memory tracking
    assert len(cache_manager.memory_usage) > 0
    assert len(cache_manager.access_history) > 0

    # Check total memory usage
    total_usage = cache_manager._get_total_memory_usage()
    assert total_usage > 0


def test_cleanup_trigger(cache_manager, test_data):
    """Test cleanup triggering."""
    # Fill cache with large data to trigger cleanup
    large_data = {"large": "x" * 1000000}  # 1MB of data

    # Save until cleanup is triggered
    saved_count = 0
    for i in range(10):
        if cache_manager.save_model_cache(f"model_{i}", large_data):
            saved_count += 1

    # Verify some files were not saved due to memory limits
    assert saved_count < 10


def test_memory_limits(cache_manager):
    """Test memory limits."""
    # Try to save data larger than memory limit
    large_data = {"large": "x" * 10000000}  # 10MB of data
    assert not cache_manager.save_model_cache("large_model", large_data)


def test_error_handling(cache_manager):
    """Test error handling."""
    # Test invalid model save
    assert not cache_manager.save_model_cache("invalid", object())

    # Test invalid result save
    assert not cache_manager.save_result("invalid", object())

    # Test loading non-existent result
    assert cache_manager.load_result("nonexistent") is None


def test_access_tracking(cache_manager, test_data):
    """Test access history tracking."""
    model_id = "test_model"

    # Save model
    cache_manager.save_model_cache(model_id, test_data["dict"])
    cache_path = str(cache_manager.models_dir / f"{model_id}.json")

    # Get initial access time
    initial_access = cache_manager.access_history.get(cache_path, 0)

    # Wait briefly
    time.sleep(0.1)

    # Load model
    cache_manager.load_model_cache(model_id)
    new_access = cache_manager.access_history.get(cache_path, 0)

    assert new_access > initial_access


def test_cleanup_policy(cache_manager, test_data):
    """Test cleanup policy."""
    # Fill cache with data
    for i in range(5):
        cache_manager.save_model_cache(f"recent_{i}", test_data["dict"])

    time.sleep(0.1)

    # Force cleanup by saving multiple large files
    for i in range(5):
        large_data = {"large": "x" * (1000000 + i * 100000)}  # Increasing sizes
        cache_manager.save_model_cache(f"large_data_{i}", large_data)

    # Check that some files were cleaned up
    remaining_files = list(cache_manager.models_dir.glob("*.json"))
    assert len(remaining_files) < 10  # Less than total attempted saves
