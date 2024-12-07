"""Test suite for real-time updates."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.user_interface import ComparisonUI, ProgressUI
from src.utils.core.processor import Processor


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with patch("src.components.user_interface.st") as mock_st:
        mock_st.empty = MagicMock(return_value=MagicMock())
        mock_st.progress = MagicMock(return_value=MagicMock())
        mock_st.columns = MagicMock(
            return_value=[MagicMock(), MagicMock(), MagicMock()]
        )
        mock_st.metric = MagicMock()
        mock_st.line_chart = MagicMock()
        mock_st.bar_chart = MagicMock()
        yield mock_st


def test_real_time_progress_updates(mock_streamlit):
    """Test real-time progress updates."""
    ui = ProgressUI()

    # Test initial state
    assert len(ui.metrics_history) == 0
    assert len(ui.quality_history) == 0
    assert len(ui.performance_history) == 0

    # Simulate progress updates
    for i in range(5):
        metrics = {
            "sharpness": 0.5 + i * 0.1,
            "contrast": 0.6 + i * 0.1,
            "detail": 0.7 + i * 0.1,
        }
        quality_score = 0.6 + i * 0.1
        performance_data = {
            "cpu_usage": 40 + i * 10,
            "memory_usage": 50 + i * 5,
            "gpu_usage": 60 + i * 5,
        }

        ui.update_progress(metrics, quality_score, performance_data, f"Step {i+1}")
        time.sleep(0.6)  # Wait for throttle interval

    # Verify history
    assert len(ui.metrics_history) == 5
    assert len(ui.quality_history) == 5
    assert len(ui.performance_history) == 5

    # Verify metric trends
    assert ui.metrics_history[-1]["sharpness"] > ui.metrics_history[0]["sharpness"]
    assert ui.quality_history[-1] > ui.quality_history[0]


def test_real_time_performance_monitoring(mock_streamlit):
    """Test real-time performance monitoring."""
    ui = ProgressUI()

    # Test performance warnings
    critical_performance = {
        "cpu_usage": 95.0,
        "memory_usage": 90.0,
        "gpu_usage": 98.0,
        "warnings": ["Critical resource usage"],
        "recommendations": ["Reduce batch size"],
    }

    ui.update_progress({"test": 0.5}, 0.8, critical_performance, "Processing")

    # Verify performance history
    assert len(ui.performance_history) == 1
    assert "warnings" in ui.performance_history[0]
    assert "recommendations" in ui.performance_history[0]


def test_real_time_metric_calculation(mock_streamlit):
    """Test real-time metric calculation."""
    ui = ProgressUI()

    # Add test metrics
    metrics = [{"test": 0.5}, {"test": 0.7}, {"test": 0.9}]

    for m in metrics:
        ui.update_progress(m, sum(m.values()), {"cpu_usage": 50}, "Processing")
        time.sleep(0.6)  # Wait for throttle interval

    # Calculate metric change
    change = ui._calculate_metric_change("test")
    assert change > 0  # Metrics should show improvement


def test_real_time_eta_calculation(mock_streamlit):
    """Test real-time ETA calculation."""
    ui = ProgressUI()

    # Add quality scores
    for i in range(3):
        ui.quality_history.append(0.5 + i * 0.2)
        time.sleep(0.1)

    # Calculate ETA
    eta = ui._estimate_remaining_time(0.3)  # 0.3 seconds elapsed
    assert isinstance(eta, float)
    assert eta > 0


def test_real_time_visualization_updates(mock_streamlit):
    """Test real-time visualization updates."""
    ui = ProgressUI()

    # Add test data
    metrics = {"sharpness": 0.8, "contrast": 0.7}
    quality_score = 0.75
    performance_data = {
        "cpu_usage": 60,
        "memory_usage": 70,
        "warnings": ["Test warning"],
        "recommendations": ["Test recommendation"],
    }

    # Update multiple times
    for _ in range(3):
        ui.update_progress(metrics, quality_score, performance_data, "Processing")
        time.sleep(0.6)  # Wait for throttle interval

    # Verify visualization calls
    assert mock_streamlit.metric.call_count > 0
    assert mock_streamlit.line_chart.call_count > 0


def test_real_time_comparison_updates(mock_streamlit):
    """Test real-time comparison updates."""
    ui = ComparisonUI()

    # Create test data
    original = np.zeros((64, 64, 3))
    enhanced = np.ones((64, 64, 3)) * 0.5
    metrics = {"sharpness": 0.8, "contrast": 0.7, "detail": 0.9}

    # Show comparison
    ui.show_comparison(original, enhanced, metrics, metrics)

    # Verify comparison visualization
    assert mock_streamlit.columns.call_count > 0
    assert mock_streamlit.image.call_count > 0
    assert mock_streamlit.metric.call_count > 0


def test_real_time_throttling(mock_streamlit):
    """Test update throttling mechanism."""
    ui = ProgressUI()

    metrics = {"test": 0.5}
    performance = {"cpu_usage": 50}

    # First update
    ui.update_progress(metrics, 0.5, performance, "Processing")
    assert len(ui.metrics_history) == 1

    # Immediate second update (should be throttled)
    ui.update_progress(metrics, 0.5, performance, "Processing")
    assert len(ui.metrics_history) == 1

    # Wait for throttle interval
    time.sleep(0.6)

    # Third update (should go through)
    ui.update_progress(metrics, 0.5, performance, "Processing")
    assert len(ui.metrics_history) == 2


def test_real_time_error_handling(mock_streamlit):
    """Test error handling in real-time updates."""
    ui = ProgressUI()

    # Initial update
    ui.update_progress(
        metrics={'quality': 0.8},
        quality_score=0.8,
        performance_data={'memory': 1000},
        status="Processing"
    )

    # Error update
    ui.update_progress(
        metrics=None,
        quality_score=None,
        performance_data=None,
        status="Error: Failed to process image"
    )

    # Verify metrics history
    assert len(ui.metrics_history) == 3  # Initial + Error + Empty
    assert ui.metrics_history[0] == {'quality': 0.8}  # Initial metrics
    assert ui.metrics_history[1] == {}  # Error metrics
    assert ui.metrics_history[2] == {}  # Empty metrics after error

    # Verify quality history
    assert len(ui.quality_history) == 3
    assert ui.quality_history[0] == 0.8  # Initial quality
    assert ui.quality_history[1] == 0.0  # Error quality
    assert ui.quality_history[2] == 0.0  # Empty quality after error

    # Verify performance history
    assert len(ui.performance_history) == 3
    assert ui.performance_history[0] == {'memory': 1000}  # Initial performance
    assert ui.performance_history[1] == {}  # Error performance
    assert ui.performance_history[2] == {}  # Empty performance after error

    # Verify feedback history
    assert len(ui.feedback_history) == 1
    assert ui.feedback_history[0]['type'] == 'error'
    assert 'Failed to process image' in ui.feedback_history[0]['message']
