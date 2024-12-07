"""Test suite for core processor."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.utils.core.processor import DatasetHandler, Processor, SessionManager


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_files(temp_data_dir):
    """Create sample data files."""
    files = []
    for i in range(10):
        data = np.random.rand(64, 64, 3).astype(np.float32)
        file_path = temp_data_dir / f"data_{i}.npy"
        np.save(file_path, data)
        files.append(str(file_path))
    return files


@pytest.fixture
def dataset_handler(sample_data_files):
    """Create dataset handler."""
    return DatasetHandler(sample_data_files, chunk_size=4)


@pytest.fixture
def session_manager():
    """Create session manager."""
    return SessionManager(cleanup_interval=1)


@pytest.fixture
def processor():
    """Create core processor."""

    class TestProcessor(Processor):
        def __init__(self):
            # Create mock for ResourceManager
            self.mock_resource_manager = MagicMock()
            self.mock_resource_manager.initialize = MagicMock()
            self.mock_resource_manager.check_resource_availability = MagicMock(return_value=True)
            self.mock_resource_manager.allocate_resources = MagicMock(return_value=True)
            self.mock_resource_manager.release_resources = MagicMock()
            
            # Create mock for SystemMonitor
            self.mock_system_monitor = MagicMock()
            self.mock_system_monitor.initialize = MagicMock()
            self.mock_system_monitor.get_system_metrics = MagicMock(return_value={})
            
            # Patch the imports
            with (
                patch("src.utils.core.processor.ResourceManager", return_value=self.mock_resource_manager),
                patch("src.utils.core.processor.SystemMonitor", return_value=self.mock_system_monitor),
                patch("src.utils.core.processor.QualityManager"),
                patch("src.utils.core.processor.GPUAccelerator")
            ):
                super().__init__()

        def _apply_processing(self, tensor):
            # Split batch into individual tensors
            return [t.unsqueeze(0) for t in tensor]

    return TestProcessor()


def test_dataset_handler_initialization(dataset_handler, sample_data_files):
    """Test dataset handler initialization."""
    assert len(dataset_handler) == len(sample_data_files)
    assert dataset_handler.chunk_size == 4
    assert len(dataset_handler.current_chunk) == 0


def test_dataset_item_loading(dataset_handler):
    """Test dataset item loading."""
    # Get first item
    item = dataset_handler[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (64, 64, 3)

    # Get item from next chunk
    item = dataset_handler[5]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (64, 64, 3)


def test_dataset_chunk_loading(dataset_handler):
    """Test chunk loading mechanism."""
    # Load first chunk
    dataset_handler._load_chunk(0)
    assert len(dataset_handler.current_chunk) == 4

    # Load partial chunk
    dataset_handler._load_chunk(8)
    assert len(dataset_handler.current_chunk) == 2


def test_session_manager_initialization(session_manager):
    """Test session manager initialization."""
    assert session_manager.cleanup_interval == 1
    assert isinstance(session_manager.sessions, dict)


def test_session_creation(session_manager):
    """Test session creation and management."""
    # Create session
    config = {"batch_size": 4}
    assert session_manager.create_session("test_session", config)

    # Try to create duplicate session
    assert not session_manager.create_session("test_session", config)

    # Get session info
    info = session_manager.get_session_info("test_session")
    assert info["config"] == config
    assert info["status"] == "active"
    assert info["processed_items"] == 0


def test_session_updates(session_manager):
    """Test session updates."""
    session_manager.create_session("test_session", {})

    # Update session
    session_manager.update_session("test_session", 5)
    info = session_manager.get_session_info("test_session")
    assert info["processed_items"] == 5

    # Update again
    session_manager.update_session("test_session", 3)
    info = session_manager.get_session_info("test_session")
    assert info["processed_items"] == 8


def test_session_cleanup(session_manager):
    """Test session cleanup functionality."""
    # Create active session
    session_manager.create_session("active", {})
    assert session_manager.get_session_info("active") is not None
    
    # Create inactive session
    session_manager.create_session("inactive", {})
    session_manager.end_session("inactive")
    
    # Force cleanup
    session_manager.cleanup_interval = 0  # Immediate cleanup
    session_manager.cleanup_sessions()
    
    # Verify active session remains
    assert session_manager.get_session_info("active") is not None
    
    # Verify inactive session is removed
    assert "inactive" not in session_manager.sessions
    assert session_manager.get_session_info("inactive") is None


def test_processor_initialization(processor):
    """Test processor initialization."""
    assert isinstance(processor.resource_manager, MagicMock)
    assert isinstance(processor.system_monitor, MagicMock)
    assert isinstance(processor.session_manager, SessionManager)


def test_batch_size_calculation(processor):
    """Test batch size calculation."""
    batch_size = processor._calculate_optimal_batch_size()
    assert isinstance(batch_size, int)
    assert 1 <= batch_size <= 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_gpu_batch_size_calculation(processor):
    """Test GPU-aware batch size calculation."""
    batch_size = processor._calculate_optimal_batch_size()
    assert isinstance(batch_size, int)
    assert batch_size > 0

    # Should be limited by GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    max_possible = gpu_memory // (
        1024 * 1024 * 3 * 4 * 2
    )  # Based on 1024x1024 RGB float32
    assert batch_size <= max_possible


def test_dataset_processing(processor, sample_data_files):
    """Test large dataset processing."""
    # Process dataset
    results, info = processor.process_dataset(sample_data_files, "test_session")

    # Check results
    assert len(results) == len(sample_data_files)
    assert all(isinstance(r, torch.Tensor) for r in results)

    # Check processing info
    assert info["total_items"] == len(sample_data_files)
    assert info["processed_items"] == len(sample_data_files)
    assert info["errors"] == 0
    assert "duration" in info
    assert len(info["system_metrics"]) > 0


def test_error_handling(processor):
    """Test error handling in processor."""
    # Create a test error
    class TestError(Exception):
        pass
            
    def raise_error(*args, **kwargs):
        raise TestError("Test error")
            
    # Replace _process_batch with error-raising function
    processor._process_batch = raise_error
        
    # Process dataset
    results, info = processor.process_dataset(["test.jpg"], "test_session")
        
    # Verify error handling
    assert len(results) == 0
    assert info["processed_items"] == 0
    assert info["errors"] == 1
    assert info["error"] == "Test error"
    assert isinstance(info["system_metrics"], dict)
    assert len(processor.error_handler.error_history) == 1
    assert "Test error" in str(processor.error_handler.error_history[0])


def test_resource_management(processor, sample_data_files):
    """Test resource management during processing."""
    # Mock resource manager to simulate resource limits
    processor.resource_manager.check_resource_availability = MagicMock(
        side_effect=[True, False, True]
    )

    # Process dataset
    results, info = processor.process_dataset(sample_data_files[:3], "test_session")

    # Verify resource checks
    assert processor.resource_manager.check_resource_availability.call_count > 0
    assert info["processed_items"] > 0
