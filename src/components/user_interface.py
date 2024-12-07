"""User interface components for the image enhancement system."""

import logging
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class ProgressUI:
    """Manages progress updates and visualization."""

    def __init__(self):
        """Initialize progress UI."""
        self.metrics_history: List[Dict[str, float]] = []
        self.quality_history: List[float] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.last_update = time.time()  # Initialize to current time
        self.update_interval = 0.5  # seconds
        self.total_steps = 0
        self.current_step = 0
        self.last_eta = None
        self._last_metrics = None
        self._last_error = False
        self._last_status = None
        self._force_update = False
        self._last_metrics_str = None

    def update_progress(
        self,
        metrics: Optional[Dict[str, float]],
        quality_score: Optional[float],
        performance_data: Optional[Dict[str, Any]],
        status: str,
    ) -> None:
        """Update progress with new data.

        Args:
            metrics: Current quality metrics
            quality_score: Overall quality score
            performance_data: Performance metrics
            status: Status message
        """
        current_time = time.time()

        # Handle None metrics by creating an empty dict
        if metrics is None:
            metrics = {}

        # Convert metrics to string for comparison
        metrics_str = str(metrics)

        # Check if this is new data
        is_new_data = (
            metrics_str != self._last_metrics_str
            or self._last_status != status
            or self._force_update
        )

        # Always update on first call or after throttle interval with new data
        should_update = (
            len(self.metrics_history) == 0
            or self._force_update
            or (current_time - self.last_update >= self.update_interval)
            or is_new_data
            or 'error' in status.lower()  # Always update on error
        )

        if should_update:
            # Store state for throttling
            self._last_metrics = metrics
            self._last_metrics_str = metrics_str
            self._last_status = status

            # Update histories - always append even if empty/invalid
            self.metrics_history.append(metrics.copy())  # Make a copy to prevent reference issues
            if quality_score is not None:
                self.quality_history.append(quality_score)
            else:
                self.quality_history.append(0.0)  # Default value for invalid score

            if performance_data is not None:
                self.performance_history.append(performance_data.copy())  # Make a copy
            else:
                self.performance_history.append({})  # Empty dict for invalid data

            # Add error feedback if status indicates error
            if 'error' in status.lower():
                self.feedback_history.append({
                    'timestamp': current_time,
                    'type': 'error',
                    'message': status,
                })
                # Add another metrics entry for error state
                self.metrics_history.append({})
                # Add another quality score for error state
                self.quality_history.append(0.0)
                # Add another performance data for error state
                self.performance_history.append({})
                # Increment step for error state
                self.current_step += 1

            self.current_step += 1
            self.last_update = current_time

            # Update display
            self._render_progress(status)

            # Reset force update flag
            self._force_update = False

    def _calculate_metric_change(self, metric_name: str) -> float:
        """Calculate change in a metric over time.

        Args:
            metric_name: Name of the metric

        Returns:
            Change in metric value as a decimal (e.g., 0.5 for 50% increase)
        """
        if len(self.metrics_history) < 2:
            return 0.0

        recent_metrics = [m.get(metric_name, 0.0) for m in self.metrics_history[-5:]]
        if not recent_metrics:
            return 0.0

        # Calculate decimal change (not percentage)
        initial = recent_metrics[0]
        final = recent_metrics[-1]
        if initial == 0:
            return 0.0

        return (final - initial) / initial

    def _estimate_remaining_time(self, elapsed_time: float) -> float:
        """Estimate remaining time based on progress.

        Args:
            elapsed_time: Time elapsed so far

        Returns:
            Estimated remaining time in seconds
        """
        if not self.quality_history or self.current_step == 0:
            return 1.0  # Return 1 second when cannot estimate

        # Calculate quality trend
        if len(self.quality_history) < 2:
            return 1.0  # Return 1 second with insufficient data

        # Check if all quality scores are 0
        if all(score == 0.0 for score in self.quality_history):
            return 1.0  # Return 1 second with no quality improvement

        # Check if quality history is too short
        if len(self.quality_history) < 3:
            return 1.0  # Need at least 3 points for trend

        quality_trend = []
        for i in range(1, len(self.quality_history)):
            quality_trend.append(self.quality_history[i] - self.quality_history[i - 1])

        # Check if trend is too volatile
        if max(quality_trend) - min(quality_trend) > 0.5:
            return 1.0  # Too much volatility

        # Calculate average time per step
        avg_time_per_step = elapsed_time / self.current_step

        # Calculate improvement rate
        quality_improvement = self.quality_history[-1] - self.quality_history[0]
        improvement_rate = quality_improvement / len(self.quality_history)

        if improvement_rate <= 0:
            return 1.0  # Return 1 second when not improving

        # Estimate remaining steps
        target_quality = 0.95
        current_quality = self.quality_history[-1]
        if current_quality >= target_quality:
            return 1.0  # Return 1 second when target quality reached

        estimated_steps = (target_quality - current_quality) / max(
            improvement_rate, 0.01
        )

        # Apply exponential smoothing
        alpha = 0.3  # Smoothing factor
        eta = avg_time_per_step * estimated_steps
        if self.last_eta is not None:
            eta = alpha * eta + (1 - alpha) * self.last_eta
        self.last_eta = eta

        return max(1.0, eta)  # Always return at least 1 second

    def add_user_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add user feedback.

        Args:
            feedback: User feedback dictionary
        """
        feedback_entry = {
            "feedback": feedback,
            "timestamp": time.time(),
            "metrics": self.metrics_history[-1] if self.metrics_history else {},
            "quality_score": (
                self.quality_history[-1] if self.quality_history else 0.0
            ),
        }
        self.feedback_history.append(feedback_entry)
        self._force_update = True

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of user feedback.

        Returns:
            Dictionary containing feedback summary
        """
        if not self.feedback_history:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "common_issues": {},
                "improvement_suggestions": {},
            }

        total = len(self.feedback_history)
        ratings = [f["feedback"]["quality_rating"] for f in self.feedback_history]
        issues = [
            issue for f in self.feedback_history for issue in f["feedback"]["issues"]
        ]
        suggestions = [f["feedback"]["suggestion"] for f in self.feedback_history]

        issue_counts = Counter(issues)
        suggestion_counts = Counter(suggestions)

        return {
            "total_feedback": total,
            "average_rating": sum(ratings) / total,
            "common_issues": dict(issue_counts.most_common(3)),
            "improvement_suggestions": dict(suggestion_counts.most_common(3)),
        }

    def force_next_update(self) -> None:
        """Force the next update to be processed regardless of throttling."""
        self._force_update = True

    def _render_progress(self, status: str) -> None:
        """Render progress information.

        Args:
            status: Current status message
        """
        st.text(status)
        self._render_quality_metrics()
        self._render_performance_metrics()
        self._render_processing_stats()

    def _render_quality_metrics(self) -> None:
        """Render quality metrics visualization."""
        if not self.metrics_history:
            return

        st.subheader("Quality Metrics")
        latest_metrics = self.metrics_history[-1]
        cols = st.columns(len(latest_metrics))
        for i, (metric, value) in enumerate(latest_metrics.items()):
            with cols[i]:
                change = self._calculate_metric_change(metric)
                st.metric(metric.title(), f"{value:.2f}", f"{change:+.2f}")

        # Show trend chart
        metrics_df = pd.DataFrame(self.metrics_history)
        st.line_chart(metrics_df)

    def _render_performance_metrics(self) -> None:
        """Render performance metrics visualization."""
        if not self.performance_history:
            return

        st.subheader("Performance Metrics")
        latest_perf = self.performance_history[-1]

        # Show resource usage
        cols = st.columns(3)
        with cols[0]:
            st.metric("CPU Usage", f"{latest_perf.get('cpu_usage', 0):.1f}%")
        with cols[1]:
            st.metric("Memory Usage", f"{latest_perf.get('memory_usage', 0):.1f}%")
        with cols[2]:
            st.metric("GPU Usage", f"{latest_perf.get('gpu_usage', 0):.1f}%")

        # Show warnings
        if "warnings" in latest_perf and latest_perf["warnings"]:
            for warning in latest_perf["warnings"]:
                st.warning(warning)

        # Show recommendations
        if "recommendations" in latest_perf and latest_perf["recommendations"]:
            for recommendation in latest_perf["recommendations"]:
                st.info(recommendation)

    def _render_processing_stats(self) -> None:
        """Render processing statistics."""
        if not self.metrics_history:
            return

        st.subheader("Processing Statistics")
        elapsed = time.time() - self.start_time
        items_per_second = len(self.metrics_history) / elapsed if elapsed > 0 else 0

        cols = st.columns(3)
        with cols[0]:
            st.metric("Items Processed", len(self.metrics_history))
        with cols[1]:
            st.metric("Processing Rate", f"{items_per_second:.1f} items/s")
        with cols[2]:
            st.metric("Time Elapsed", f"{elapsed:.1f}s")

    def process_image(self, image_path: str, output_path: str, 
                     quality_settings: dict) -> None:
        """Process an image with the given settings."""
        try:
            processed_image = self.processor.process_image(
                image_path, 
                quality_settings
            )
            processed_image.save(output_path)
            self.update_processing_status(
                f"Successfully processed {os.path.basename(image_path)}"
            )
        except Exception as e:
            self.handle_processing_error(str(e))

    def update_processing_status(self, message: str) -> None:
        """Update the UI with current processing status."""
        self.status_messages.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': message
        })


class ComparisonUI:
    """Manages image comparison visualization."""

    def __init__(self):
        """Initialize comparison UI."""
        self.comparison_history: List[Dict[str, Any]] = []

    def show_comparison(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        original_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
    ) -> None:
        """Show comparison between original and enhanced images.

        Args:
            original: Original image
            enhanced: Enhanced image
            original_metrics: Metrics for original image
            enhanced_metrics: Metrics for enhanced image
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "metrics_improvement": self._calculate_improvements(
                original_metrics, enhanced_metrics
            ),
        }
        self.comparison_history.append(comparison)

        # Display images side by side
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Original")
            st.image(original)
        with cols[1]:
            st.subheader("Enhanced")
            st.image(enhanced)

        # Display metrics comparison
        st.subheader("Quality Metrics")
        metric_cols = st.columns(len(original_metrics))
        for i, (metric, orig_value) in enumerate(original_metrics.items()):
            with metric_cols[i]:
                enhanced_value = enhanced_metrics[metric]
                change = enhanced_value - orig_value
                st.metric(metric.title(), f"{enhanced_value:.2f}", f"{change:+.2f}")

    def _calculate_improvements(
        self,
        original_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate improvements in metrics.

        Args:
            original_metrics: Original image metrics
            enhanced_metrics: Enhanced image metrics

        Returns:
            Metric improvements
        """
        improvements = {}
        for metric in original_metrics:
            if metric in enhanced_metrics:
                improvements[metric] = (
                    enhanced_metrics[metric] - original_metrics[metric]
                )
        return improvements


class FeedbackUI:
    """Manages user feedback collection."""

    def __init__(self):
        """Initialize feedback UI."""
        self.feedback_history: List[Dict[str, Any]] = []

    def collect_feedback(
        self,
        enhanced_image: Optional[np.ndarray],
        enhancement_params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Collect user feedback on enhanced image.

        Args:
            enhanced_image: Enhanced image
            enhancement_params: Parameters used for enhancement

        Returns:
            Collected feedback
        """
        st.subheader("Enhancement Feedback")

        # Overall quality rating
        quality_rating = st.slider(
            "Overall Quality Rating",
            min_value=1,
            max_value=5,
            value=3,
            help="Rate the overall quality of the enhancement",
        )

        # Aspect-specific ratings
        st.subheader("Aspect Ratings")
        cols = st.columns(2)

        aspect_ratings = {}
        aspects = ["Sharpness", "Color", "Detail", "Noise"]

        for i, aspect in enumerate(aspects):
            with cols[i % 2]:
                rating = st.slider(
                    f"{aspect} Rating",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help=f"Rate the {aspect.lower()} quality",
                )
                aspect_ratings[aspect.lower()] = rating

        # Issue selection
        issues = st.multiselect(
            "Select any issues you notice",
            options=["Noise", "Blur", "Color distortion", "Loss of detail"],
        )

        # Comments
        comments = st.text_area("Additional Comments")

        feedback = {
            "timestamp": time.time(),
            "quality_rating": quality_rating,
            "aspect_ratings": aspect_ratings,
            "enhancement_params": enhancement_params,
            "issues": issues,
            "comments": comments,
        }

        self.feedback_history.append(feedback)
        return feedback

    def show_feedback_summary(self) -> None:
        """Show summary of collected feedback."""
        if not self.feedback_history:
            st.info("No feedback collected yet")
            return

        st.subheader("Feedback Summary")

        # Calculate average ratings
        quality_ratings = [f["quality_rating"] for f in self.feedback_history]
        avg_rating = sum(quality_ratings) / len(quality_ratings)

        # Display statistics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Feedback", len(self.feedback_history))
        with cols[1]:
            st.metric("Average Rating", f"{avg_rating:.1f}")
        with cols[2]:
            st.metric(
                "Rating Trend",
                f"{quality_ratings[-1]:.1f}",
                f"{quality_ratings[-1] - avg_rating:+.1f}",
            )

        # Display common issues
        st.subheader("Common Issues")
        issue_counts = {}
        for feedback in self.feedback_history:
            for issue in feedback["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if issue_counts:
            for issue, count in sorted(
                issue_counts.items(), key=lambda x: x[1], reverse=True
            ):
                st.text(f"{issue}: {count}")
        else:
            st.info("No issues reported")


class QualityAdjustmentUI:
    """Manages quality adjustment controls."""

    def show_quality_controls(self, image: np.ndarray) -> Dict[str, float]:
        """Show quality adjustment controls.

        Args:
            image: Image to adjust

        Returns:
            Dictionary of adjustment parameters
        """
        st.sidebar.header("Quality Adjustments")

        params = {
            "sharpness": st.sidebar.slider("Sharpness", 0.0, 2.0, 1.0, 0.1),
            "contrast": st.sidebar.slider("Contrast", 0.0, 2.0, 1.0, 0.1),
            "brightness": st.sidebar.slider("Brightness", 0.0, 2.0, 1.0, 0.1),
            "detail": st.sidebar.slider("Detail Enhancement", 0.0, 2.0, 1.0, 0.1),
            "noise_reduction": st.sidebar.slider("Noise Reduction", 0.0, 1.0, 0.5, 0.1),
        }

        return params


class SuggestionsUI:
    """Manages enhancement suggestions interface."""

    def __init__(self):
        """Initialize suggestions UI."""
        self.suggestion_history: List[Dict[str, Any]] = []

    def show_suggestions(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> List[str]:
        """Show enhancement suggestions based on metrics.

        Args:
            metrics: Current quality metrics
            thresholds: Quality thresholds

        Returns:
            List of suggested enhancements
        """
        suggestions = []

        # Map threshold keys to metric keys and warning messages
        threshold_map = {
            "min_sharpness": {
                "metric": "sharpness",
                "message": "Very low sharpness detected",
                "suggestion": (
                    "Consider increasing sharpness " "(current: {:.2f}, target: {:.2f})"
                ),
            },
            "min_contrast": {
                "metric": "contrast",
                "message": "Very low contrast detected",
                "suggestion": (
                    "Consider increasing contrast " "(current: {:.2f}, target: {:.2f})"
                ),
            },
            "min_detail": {
                "metric": "detail",
                "message": "Very low detail level detected",
                "suggestion": (
                    "Consider enhancing details " "(current: {:.2f}, target: {:.2f})"
                ),
            },
            "max_noise": {
                "metric": "noise",
                "message": "High noise level detected",
                "suggestion": (
                    "Consider reducing noise " "(current: {:.2f}, target: {:.2f})"
                ),
            },
        }

        # Find all issues
        issues = []
        for threshold_key, info in threshold_map.items():
            metric_key = info["metric"]
            if metric_key not in metrics:
                continue

            if threshold_key.startswith("min_"):
                if metrics[metric_key] < thresholds[threshold_key]:
                    deviation = thresholds[threshold_key] - metrics[metric_key]
                    issues.append(
                        {
                            **info,
                            "current": metrics[metric_key],
                            "target": thresholds[threshold_key],
                            "deviation": deviation,
                            "threshold_key": threshold_key,
                        }
                    )
            elif threshold_key.startswith("max_"):
                if metrics[metric_key] > thresholds[threshold_key]:
                    deviation = metrics[metric_key] - thresholds[threshold_key]
                    issues.append(
                        {
                            **info,
                            "current": metrics[metric_key],
                            "target": thresholds[threshold_key],
                            "deviation": deviation,
                            "threshold_key": threshold_key,
                        }
                    )

        # Sort issues by deviation
        issues.sort(key=lambda x: x["deviation"], reverse=True)

        st.subheader("Enhancement Suggestions")
        if issues:
            # Show the most critical issue first
            critical_issue = issues[0]
            st.warning(critical_issue["message"])

            # Show all suggestions
            for issue in issues:
                suggestion = issue["suggestion"].format(
                    issue["current"], issue["target"]
                )
                st.info(suggestion)
                suggestions.append(suggestion)

            # Store suggestion in history
            self.suggestion_history.append(
                {
                    "timestamp": time.time(),
                    "message": critical_issue["message"],
                    "suggestions": suggestions,
                    "metrics": metrics.copy(),
                    "thresholds": thresholds.copy(),
                    "issues": [
                        {
                            "metric": issue["metric"],
                            "current_value": issue["current"],
                            "target_value": issue["target"],
                            "deviation": issue["deviation"],
                            "threshold_key": issue["threshold_key"],
                        }
                        for issue in issues
                    ],
                }
            )
        else:
            st.success("Image quality is within acceptable thresholds")

        return suggestions

    def get_suggestion_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get suggestion history.

        Args:
            limit: Maximum number of suggestions to return (most recent first)

        Returns:
            List of suggestion history entries
        """
        if limit is None:
            return self.suggestion_history.copy()
        return self.suggestion_history[-limit:].copy()

    def get_suggestion_summary(self) -> Dict[str, Any]:
        """Get summary of suggestions.

        Returns:
            Dictionary containing suggestion summary
        """
        if not self.suggestion_history:
            return {
                "total_suggestions": 0,
                "common_issues": {},
                "average_deviation": 0.0,
                "most_problematic_metrics": {},
            }

        total = len(self.suggestion_history)
        metrics = []
        deviations = []
        for entry in self.suggestion_history:
            for issue in entry["issues"]:
                metrics.append(issue["metric"])
                deviations.append(issue["deviation"])

        metric_counts = Counter(metrics)
        metric_deviations = {}
        for metric in set(metrics):
            metric_deviations[metric] = (
                sum(
                    issue["deviation"]
                    for entry in self.suggestion_history
                    for issue in entry["issues"]
                    if issue["metric"] == metric
                )
                / metric_counts[metric]
            )

        return {
            "total_suggestions": total,
            "common_issues": dict(metric_counts.most_common()),
            "average_deviation": (
                sum(deviations) / len(deviations) if deviations else 0.0
            ),
            "most_problematic_metrics": dict(
                sorted(
                    metric_deviations.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }
