"""Test suite for user interface components."""

import time
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

from src.components.user_interface import (
    ComparisonUI,
    FeedbackUI,
    ProgressUI,
    QualityAdjustmentUI,
    SuggestionsUI,
)


@pytest.fixture
def test_image():
    """Create test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # Add white square
    return img


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with patch("src.components.user_interface.st") as mock_st:
        # Mock slider returns
        mock_st.sidebar.slider.return_value = 1.0
        mock_st.slider.return_value = 3

        # Mock progress bar
        mock_st.progress = MagicMock()

        # Mock columns
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

        # Mock metrics
        mock_st.metric = MagicMock()

        # Mock charts
        mock_st.line_chart = MagicMock()
        mock_st.bar_chart = MagicMock()

        # Mock text elements
        mock_st.text = MagicMock()
        mock_st.write = MagicMock()
        mock_st.info = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.success = MagicMock()
        mock_st.error = MagicMock()

        # Mock headers
        mock_st.header = MagicMock()
        mock_st.subheader = MagicMock()

        # Mock image display
        mock_st.image = MagicMock()

        # Mock divider
        mock_st.divider = MagicMock()

        # Mock multiselect
        mock_st.multiselect = MagicMock(return_value=[])

        # Mock text area
        mock_st.text_area = MagicMock(return_value="")

        yield mock_st


def test_quality_adjustment(mock_streamlit, test_image):
    """Test quality adjustment interface."""
    ui = QualityAdjustmentUI()
    params = ui.show_quality_controls(test_image)

    # Verify parameters
    assert isinstance(params, dict)
    assert "sharpness" in params
    assert "contrast" in params
    assert "brightness" in params
    assert "detail" in params
    assert "noise_reduction" in params

    # Verify streamlit calls
    mock_streamlit.sidebar.header.assert_called_with("Quality Adjustments")
    assert mock_streamlit.sidebar.slider.call_count >= 5


def test_comparison_view(mock_streamlit, test_image):
    """Test comparison interface."""
    ui = ComparisonUI()
    metrics = {"sharpness": 0.8, "contrast": 0.7}

    ui.show_comparison(test_image, test_image, metrics, metrics)

    # Verify streamlit calls
    mock_streamlit.subheader.assert_called()
    mock_streamlit.columns.assert_called()
    mock_streamlit.image.assert_called()
    mock_streamlit.metric.assert_called()


def test_feedback_collection(mock_streamlit):
    """Test feedback collection interface."""
    ui = FeedbackUI()
    params = {"sharpness": 1.0, "contrast": 1.0}

    feedback = ui.collect_feedback(None, params)

    # Verify feedback structure
    assert isinstance(feedback, dict)
    assert "quality_rating" in feedback
    assert "aspect_ratings" in feedback
    assert "issues" in feedback
    assert "comments" in feedback
    assert "enhancement_params" in feedback
    assert "timestamp" in feedback

    # Verify streamlit calls
    mock_streamlit.subheader.assert_called()
    mock_streamlit.slider.assert_called()
    mock_streamlit.multiselect.assert_called()
    mock_streamlit.text_area.assert_called()


def test_progress_initialization():
    """Test progress UI initialization."""
    ui = ProgressUI()
    assert isinstance(ui.metrics_history, list)
    assert isinstance(ui.quality_history, list)
    assert isinstance(ui.performance_history, list)
    assert isinstance(ui.feedback_history, list)
    assert ui.start_time > 0
    assert ui.last_update > 0
    assert ui.update_interval == 0.5


def test_progress_update(mock_streamlit):
    """Test progress update functionality."""
    ui = ProgressUI()

    metrics = {"sharpness": 0.8, "contrast": 0.7, "detail": 0.9}

    quality_score = 0.85

    performance_data = {
        "cpu_usage": 45.5,
        "memory_usage": 60.2,
        "gpu_usage": 80.0,
        "warnings": ["High GPU usage detected"],
        "recommendations": ["Consider reducing batch size"],
    }

    # First update
    ui.update_progress(metrics, quality_score, performance_data, "Processing")

    assert len(ui.metrics_history) == 1
    assert len(ui.quality_history) == 1
    assert len(ui.performance_history) == 1

    # Test update throttling
    ui.update_progress(metrics, quality_score, performance_data, "Processing")
    assert len(ui.metrics_history) == 1  # Should not update due to throttling

    # Wait for throttle interval
    time.sleep(0.6)

    # Update again
    ui.update_progress(metrics, quality_score, performance_data, "Processing")
    assert len(ui.metrics_history) == 2


def test_progress_rendering(mock_streamlit):
    """Test progress rendering."""
    ui = ProgressUI()

    # Add some test data
    metrics = {"sharpness": 0.8, "contrast": 0.7}
    quality_score = 0.85
    performance_data = {
        "cpu_usage": 45.5,
        "memory_usage": 60.2,
        "warnings": ["Test warning"],
        "recommendations": ["Test recommendation"],
    }

    ui.update_progress(metrics, quality_score, performance_data, "Processing")

    # Verify streamlit calls
    mock_streamlit.subheader.assert_called()
    mock_streamlit.columns.assert_called()
    mock_streamlit.metric.assert_called()


def test_quality_metrics_rendering(mock_streamlit):
    """Test quality metrics rendering."""
    ui = ProgressUI()

    # Add test metrics
    metrics1 = {"sharpness": 0.7, "contrast": 0.6}
    metrics2 = {"sharpness": 0.8, "contrast": 0.7}

    ui.metrics_history.extend([metrics1, metrics2])
    ui._render_quality_metrics()

    # Verify metric displays
    assert mock_streamlit.metric.call_count >= 2

    # Verify trend chart
    mock_streamlit.line_chart.assert_called_once()


def test_performance_metrics_rendering(mock_streamlit):
    """Test performance metrics rendering."""
    ui = ProgressUI()

    performance_data = {
        "cpu_usage": 45.5,
        "memory_usage": 60.2,
        "gpu_usage": 80.0,
        "warnings": ["High GPU usage"],
        "recommendations": ["Reduce batch size"],
    }

    ui.performance_history.append(performance_data)
    ui._render_performance_metrics()

    # Verify resource metrics
    assert mock_streamlit.metric.call_count >= 3

    # Verify warnings and recommendations
    mock_streamlit.warning.assert_called_once()
    mock_streamlit.info.assert_called_once()


def test_processing_stats_rendering(mock_streamlit):
    """Test processing statistics rendering."""
    ui = ProgressUI()

    # Add test data
    for _ in range(3):
        ui.metrics_history.append({"test": 0.5})
        ui.quality_history.append(0.8)
        time.sleep(0.1)

    ui._render_processing_stats()

    # Verify statistics display
    assert mock_streamlit.metric.call_count >= 3


def test_metric_change_calculation():
    """Test metric change calculation."""
    ui = ProgressUI()

    # Add test metrics
    ui.metrics_history.extend([{"test": 0.5}, {"test": 0.75}])

    change = ui._calculate_metric_change("test")
    assert change == 0.5  # 50% increase

    # Test with zero previous value
    ui.metrics_history = [{"test": 0.0}, {"test": 0.5}]
    change = ui._calculate_metric_change("test")
    assert change == 0.0


def test_remaining_time_estimation():
    """Test remaining time estimation."""
    ui = ProgressUI()
    
    # Test with no history
    eta = ui._estimate_remaining_time(0.0)
    assert isinstance(eta, float)
    assert eta == 1.0  # Default value when no history
    
    # Test with insufficient history
    ui.quality_history = [0.5]
    eta = ui._estimate_remaining_time(1.0)
    assert isinstance(eta, float)
    assert eta == 1.0  # Default value with insufficient data
    
    # Test with valid history
    ui.quality_history = [0.5, 0.6, 0.7, 0.8]
    ui.current_step = 4
    eta = ui._estimate_remaining_time(10.0)
    assert isinstance(eta, float)
    assert eta > 0.0  # Should return positive value
    
    # Test with target quality reached
    ui.quality_history = [0.9, 0.92, 0.95, 0.96]
    eta = ui._estimate_remaining_time(10.0)
    assert isinstance(eta, float)
    assert eta == 1.0  # Default value when target reached
    
    # Test with no improvement
    ui.quality_history = [0.8, 0.8, 0.8, 0.8]
    eta = ui._estimate_remaining_time(10.0)
    assert isinstance(eta, float)
    assert eta == 1.0  # Default value when no improvement


def test_user_feedback_integration(mock_streamlit):
    """Test user feedback integration."""
    ui = ProgressUI()

    feedback = {
        "quality_rating": 4,
        "issues": ["Noise", "Blur"],
        "suggestion": "Increase sharpness",
    }

    ui.add_user_feedback(feedback)
    assert len(ui.feedback_history) == 1
    assert ui.feedback_history[0]["feedback"] == feedback


def test_feedback_summary():
    """Test feedback summary generation."""
    ui = ProgressUI()

    # Add test feedback
    feedbacks = [
        {
            "quality_rating": 4,
            "issues": ["Noise", "Blur"],
            "suggestion": "Increase sharpness",
        },
        {
            "quality_rating": 3,
            "issues": ["Noise", "Color"],
            "suggestion": "Adjust color balance",
        },
    ]

    for feedback in feedbacks:
        ui.add_user_feedback(feedback)

    summary = ui.get_feedback_summary()

    assert summary["total_feedback"] == 2
    assert summary["average_rating"] == 3.5
    assert summary["common_issues"]["Noise"] == 2
    assert len(summary["improvement_suggestions"]) == 2


def test_edge_cases(mock_streamlit):
    """Test edge cases in progress UI."""
    ui = ProgressUI()

    # Test with empty history
    ui._render_progress("Processing")

    # Test with single data point
    ui.update_progress({"test": 0.5}, 0.8, {"cpu_usage": 50}, "Processing")
    ui._render_progress("Processing")

    # Test with invalid metrics
    ui.update_progress({}, 0.0, {}, "Processing")
    ui._render_progress("Processing")

    # Verify streamlit calls
    assert mock_streamlit.subheader.call_count >= 1
    assert mock_streamlit.metric.call_count >= 1


def test_performance_warnings(mock_streamlit):
    """Test performance warning generation."""
    ui = ProgressUI()

    # Test with high resource usage
    performance_data = {
        "cpu_usage": 95.0,
        "memory_usage": 90.0,
        "gpu_usage": 98.0,
        "warnings": ["Critical CPU usage", "High memory usage", "Critical GPU usage"],
        "recommendations": [
            "Reduce batch size",
            "Enable memory optimization",
            "Consider using CPU fallback",
        ],
    }

    ui.update_progress({"test": 0.5}, 0.8, performance_data, "Processing")

    # Verify warning displays
    assert mock_streamlit.warning.call_count >= 1
    assert mock_streamlit.info.call_count >= 1


def test_real_time_updates(mock_streamlit):
    """Test real-time update functionality."""
    ui = ProgressUI()

    # Simulate real-time updates
    for i in range(3):
        metrics = {"quality": 0.5 + i * 0.1}
        performance = {"cpu_usage": 50 + i * 10}

        if i > 0:
            time.sleep(0.6)  # Wait for throttle interval

        ui.update_progress(metrics, 0.8, performance, f"Step {i+1}")

    assert len(ui.metrics_history) == 3
    assert len(ui.performance_history) == 3

    # Verify progress rendering
    assert mock_streamlit.metric.call_count > 0
    assert mock_streamlit.line_chart.call_count > 0


def test_suggestions_display(mock_streamlit):
    """Test suggestions display interface."""
    ui = SuggestionsUI()
    metrics = {
        "sharpness": 0.15,  # Will be processed last due to alphabetical sorting and will trigger warning
        "contrast": 0.3,
    }
    thresholds = {"min_sharpness": 0.5, "min_contrast": 0.4}

    suggestions = ui.show_suggestions(metrics, thresholds)

    # Verify subheader
    mock_streamlit.subheader.assert_called_once_with("Enhancement Suggestions")

    # Verify warning and suggestion messages were displayed
    mock_streamlit.warning.assert_called_once_with("Very low sharpness detected")
    mock_streamlit.info.assert_has_calls(
        [
            call("Consider increasing sharpness (current: 0.15, target: 0.50)"),
            call("Consider increasing contrast (current: 0.30, target: 0.40)"),
        ]
    )

    # Verify success message not shown
    mock_streamlit.success.assert_not_called()

    # Verify suggestion history
    assert len(ui.suggestion_history) == 1
    assert len(ui.suggestion_history[0]["suggestions"]) == 2
    assert ui.suggestion_history[0]["suggestions"] == [
        "Consider increasing sharpness (current: 0.15, target: 0.50)",
        "Consider increasing contrast (current: 0.30, target: 0.40)",
    ]


def test_quality_adjustment_edge_cases(mock_streamlit):
    """Test quality adjustment with edge cases."""
    ui = QualityAdjustmentUI()

    # Test with empty image
    empty_image = np.zeros((1, 1, 3), dtype=np.uint8)
    params = ui.show_quality_controls(empty_image)
    assert all(0 <= v <= 2.0 for v in params.values())

    # Test with large image
    large_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    params = ui.show_quality_controls(large_image)
    assert all(0 <= v <= 2.0 for v in params.values())


def test_feedback_validation(mock_streamlit):
    """Test feedback validation."""
    ui = FeedbackUI()

    # Test with empty parameters
    feedback = ui.collect_feedback(None, {})
    assert isinstance(feedback["aspect_ratings"], dict)
    assert isinstance(feedback["quality_rating"], (int, float))

    # Test with invalid parameters
    feedback = ui.collect_feedback(None, {"invalid": -1})
    assert isinstance(feedback["aspect_ratings"], dict)
    assert isinstance(feedback["quality_rating"], (int, float))
