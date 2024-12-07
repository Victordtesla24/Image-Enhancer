"""Test suite for system integrator and monitoring."""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.core.system_integrator import (
    ResourceManager,
    SystemIntegrator,
    SystemMonitor,
)


@pytest.fixture
def system_monitor():
    """Create system monitor."""
    return SystemMonitor()


@pytest.fixture
def resource_manager():
    """Create resource manager."""
    return ResourceManager()


@pytest.fixture
def system_integrator():
    """Create system integrator."""
    return SystemIntegrator()


def test_system_monitor_initialization(system_monitor):
    """Test system monitor initialization."""
    assert system_monitor.sampling_interval == 1.0
    assert isinstance(system_monitor.metrics_history, list)
    assert system_monitor.start_time > 0


def test_system_metrics(system_monitor):
    """Test system metrics collection."""
    metrics = system_monitor.get_system_metrics()

    # Check CPU metrics
    assert "cpu" in metrics
    assert "usage_percent" in metrics["cpu"]
    assert "count" in metrics["cpu"]
    assert isinstance(metrics["cpu"]["usage_percent"], (int, float))

    # Check memory metrics
    assert "memory" in metrics
    assert "total" in metrics["memory"]
    assert "available" in metrics["memory"]
    assert "used" in metrics["memory"]
    assert "percent" in metrics["memory"]

    # Check disk metrics
    assert "disk" in metrics
    assert "total" in metrics["disk"]
    assert "used" in metrics["disk"]
    assert "free" in metrics["disk"]

    # Check network metrics
    assert "network" in metrics
    assert "bytes_sent" in metrics["network"]
    assert "bytes_recv" in metrics["network"]


def test_metrics_history(system_monitor):
    """Test metrics history tracking."""
    # Get metrics multiple times
    for _ in range(3):
        system_monitor.get_system_metrics()
        time.sleep(0.1)

    history = system_monitor.get_metrics_history()
    assert len(history) == 3

    # Test with limit
    limited_history = system_monitor.get_metrics_history(limit=2)
    assert len(limited_history) == 2


def test_resource_utilization(system_monitor):
    """Test resource utilization summary."""
    utilization = system_monitor.get_resource_utilization()

    assert "cpu_usage" in utilization
    assert "memory_usage" in utilization
    assert "disk_usage" in utilization
    assert "uptime" in utilization

    if torch.cuda.is_available():
        assert "gpu_memory_usage" in utilization


def test_resource_manager_initialization(resource_manager):
    """Test resource manager initialization."""
    assert resource_manager.memory_limit == 0.8
    assert resource_manager.cpu_limit == 0.9
    assert isinstance(resource_manager.monitor, SystemMonitor)
    assert isinstance(resource_manager.allocated_resources, dict)


def test_resource_availability(resource_manager):
    """Test resource availability checking."""
    # Test with no memory requirement
    assert resource_manager.check_resource_availability()

    # Test with reasonable memory requirement
    assert resource_manager.check_resource_availability(1024 * 1024)  # 1MB

    # Test with excessive memory requirement
    assert not resource_manager.check_resource_availability(
        1024 * 1024 * 1024 * 1024
    )  # 1TB


def test_resource_allocation(resource_manager):
    """Test resource allocation."""
    # Allocate resources
    assert resource_manager.allocate_resources("task1", 1024 * 1024)

    # Check allocation status
    status = resource_manager.get_allocation_status()
    assert "task1" in status["allocated_resources"]

    # Release resources
    resource_manager.release_resources("task1")
    status = resource_manager.get_allocation_status()
    assert "task1" not in status["allocated_resources"]


def test_system_integrator_initialization(system_integrator):
    """Test system integrator initialization."""
    assert isinstance(system_integrator.monitor, SystemMonitor)
    assert isinstance(system_integrator.resource_manager, ResourceManager)
    assert isinstance(system_integrator.components, dict)


def test_component_management(system_integrator):
    """Test component management."""
    # Create mock component
    component = MagicMock()

    # Register component
    system_integrator.register_component("test", component)
    assert "test" in system_integrator.components

    # Get component
    retrieved = system_integrator.get_component("test")
    assert retrieved == component

    # Remove component
    system_integrator.remove_component("test")
    assert "test" not in system_integrator.components


def test_system_info(system_integrator):
    """Test system information retrieval."""
    info = system_integrator.get_system_info()

    assert "device" in info
    assert "components" in info
    assert "system_metrics" in info
    assert "resource_utilization" in info
    assert "allocation_status" in info


def test_system_health(system_integrator):
    """Test system health checking."""
    health = system_integrator.check_system_health()

    assert "status" in health
    assert "warnings" in health
    assert "metrics" in health
    assert health["status"] in ["healthy", "warning", "critical"]


@patch("torch.cuda.empty_cache")
def test_cleanup(mock_empty_cache, system_integrator):
    """Test system cleanup."""
    # Add mock component with cleanup method
    component = MagicMock()
    component.cleanup = MagicMock()
    system_integrator.register_component("test", component)

    # Perform cleanup
    system_integrator.cleanup()

    # Verify component cleanup was called
    component.cleanup.assert_called_once()

    if torch.cuda.is_available():
        mock_empty_cache.assert_called_once()
