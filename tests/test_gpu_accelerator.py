"""Test suite for GPU accelerator."""

import pytest
import torch
from unittest.mock import MagicMock

from src.utils.core.gpu_accelerator import GPUAccelerator

@pytest.fixture
def accelerator():
    """Create GPU accelerator instance."""
    return GPUAccelerator()

def test_initialization(accelerator):
    """Test accelerator initialization."""
    assert accelerator.initialized
    assert accelerator.device in ['cpu', 'cuda']

def test_device_selection(accelerator):
    """Test device selection."""
    device = accelerator.get_device()
    assert device in ['cpu', 'cuda']
    
    # Test CPU fallback
    accelerator.set_device('cpu')
    assert accelerator.get_device() == 'cpu'

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_operations():
    """Test CUDA-specific operations."""
    accelerator = GPUAccelerator()
    accelerator.set_device('cuda')
    assert accelerator.get_device() == 'cuda'
    
    # Test memory info
    memory_info = accelerator.get_memory_info()
    assert all(k in memory_info for k in ['total', 'allocated', 'cached', 'free'])
    assert all(isinstance(v, int) for v in memory_info.values())

def test_error_handling(accelerator):
    """Test error handling."""
    # Test invalid device
    with pytest.raises(ValueError):
        accelerator.set_device('invalid')
        
    # Test CUDA unavailable
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            accelerator.set_device('cuda')

def test_cleanup(accelerator):
    """Test cleanup."""
    accelerator.cleanup()  # Should not raise errors
