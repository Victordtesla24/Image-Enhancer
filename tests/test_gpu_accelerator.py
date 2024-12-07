"""Test suite for GPU accelerator."""

import numpy as np
import pytest
import torch

from src.utils.core.gpu_accelerator import DeviceManager, GPUAccelerator


@pytest.fixture
def device_manager():
    """Create device manager."""
    return DeviceManager()


@pytest.fixture
def gpu_accelerator():
    """Create GPU accelerator."""
    return GPUAccelerator()


@pytest.fixture
def test_array():
    """Create test array."""
    return np.random.rand(64, 64, 3).astype(np.float32)


def test_device_initialization(device_manager):
    """Test device initialization."""
    assert len(device_manager.devices) > 0
    assert all(isinstance(d, dict) for d in device_manager.devices)
    for device in device_manager.devices:
        assert "index" in device
        assert "name" in device
        assert "memory_total" in device
        assert "memory_used" in device


def test_device_selection(device_manager):
    """Test device selection."""
    device = device_manager.get_next_device()
    assert isinstance(device, torch.device)

    optimal_device = device_manager.get_optimal_device()
    assert isinstance(optimal_device, torch.device)


def test_memory_stats(device_manager):
    """Test memory statistics."""
    stats = device_manager.memory_stats()
    assert isinstance(stats, dict)
    if torch.cuda.is_available():
        assert len(stats) > 0
        for device_stats in stats.values():
            assert "allocated" in device_stats
            assert "cached" in device_stats
            assert "total" in device_stats


def test_batch_size_calculation(gpu_accelerator):
    """Test batch size calculation."""
    batch_size = gpu_accelerator._calculate_optimal_batch_size()
    assert isinstance(batch_size, int)
    assert batch_size > 0
    assert batch_size <= 32


def test_array_to_gpu(gpu_accelerator, test_array):
    """Test array to GPU conversion."""
    tensor = gpu_accelerator.to_gpu(test_array)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[-3:] == (3, 64, 64)
    assert tensor.device.type in ["cuda", "cpu"]


def test_tensor_to_cpu(gpu_accelerator, test_array):
    """Test tensor to CPU conversion."""
    tensor = gpu_accelerator.to_gpu(test_array)
    array = gpu_accelerator.to_cpu(tensor)
    assert isinstance(array, np.ndarray)
    assert array.shape == test_array.shape
    np.testing.assert_array_almost_equal(array, test_array)


def test_single_batch_processing(gpu_accelerator):
    """Test single batch processing."""
    arrays = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
    results = gpu_accelerator._process_batch_single(arrays)
    assert len(results) == len(arrays)
    assert all(isinstance(r, torch.Tensor) for r in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_parallel_batch_processing(gpu_accelerator):
    """Test parallel batch processing."""
    arrays = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(10)]
    results = gpu_accelerator._process_batch_parallel(arrays)
    assert len(results) == len(arrays)
    assert all(isinstance(r, torch.Tensor) for r in results)


def test_memory_cleanup(gpu_accelerator):
    """Test memory cleanup."""
    # Get initial memory stats
    initial_stats = gpu_accelerator.get_memory_stats()

    # Create some tensors to allocate memory
    _ = [torch.rand(1000, 1000) for _ in range(5)]

    # Clear memory
    gpu_accelerator.clear_memory()

    # Get final memory stats
    final_stats = gpu_accelerator.get_memory_stats()

    if torch.cuda.is_available():
        for device in final_stats:
            assert (
                final_stats[device]["allocated"] <= initial_stats[device]["allocated"]
            )


def test_invalid_input_handling(gpu_accelerator):
    """Test invalid input handling."""
    # Test invalid array type
    with pytest.raises(TypeError):
        gpu_accelerator.to_gpu([1, 2, 3])

    # Test invalid tensor type
    with pytest.raises(TypeError):
        gpu_accelerator.to_cpu([1, 2, 3])


def test_batch_processing_empty_input(gpu_accelerator):
    """Test batch processing with empty input."""
    results = gpu_accelerator.process_batch([])
    assert len(results) == 0


def test_batch_processing_single_item(gpu_accelerator, test_array):
    """Test batch processing with single item."""
    results = gpu_accelerator.process_batch([test_array])
    assert len(results) == 1
    assert isinstance(results[0], torch.Tensor)


def test_device_manager_clear_memory(device_manager):
    """Test device manager memory clearing."""
    # Create some tensors
    if torch.cuda.is_available():
        tensors = [torch.rand(1000, 1000).cuda() for _ in range(5)]

    # Clear memory
    device_manager.clear_memory()

    # Check memory stats
    stats = device_manager.memory_stats()
    if torch.cuda.is_available():
        for device_stats in stats.values():
            assert device_stats["allocated"] >= 0


def test_batch_processing(gpu_accelerator):
    """Test batch processing of tensors."""
    input_data = [torch.randn(3, 224, 224) for _ in range(10)]
    processed = gpu_accelerator.process_batch(input_data)
    assert processed is not None
    assert len(processed) == len(input_data)
    assert all(isinstance(p, torch.Tensor) for p in processed)
    assert all(p.shape == input_data[0].shape for p in processed)
