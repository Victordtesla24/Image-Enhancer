"""Quality improvement analysis module."""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class QualityImprovementAnalyzer:
    """Analyzer for quality improvements and feedback generation."""

    def analyze_quality_improvement(
        self, metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze quality improvement from processing.

        Args:
            metrics_comparison: Metrics comparison dictionary

        Returns:
            Quality improvement analysis
        """
        analysis = {
            "overall_improvement": 0.0,
            "significant_improvements": [],
            "degradations": [],
            "stable_metrics": [],
        }

        improvements = []

        for metric, comparison in metrics_comparison.items():
            improvement = comparison["improvement"]
            improvement_percentage = comparison["improvement_percentage"]

            if abs(improvement_percentage) < 1.0:
                analysis["stable_metrics"].append(metric)
            elif improvement_percentage > 5.0:
                analysis["significant_improvements"].append(
                    {"metric": metric, "improvement": improvement_percentage}
                )
                improvements.append(improvement_percentage)
            elif improvement_percentage < -5.0:
                analysis["degradations"].append(
                    {"metric": metric, "degradation": abs(improvement_percentage)}
                )

        if improvements:
            analysis["overall_improvement"] = np.mean(improvements)

        return analysis

    def generate_analysis_feedback(self, analysis: Dict[str, Any]) -> None:
        """Generate warnings and recommendations based on analysis.

        Args:
            analysis: Processing analysis dictionary
        """
        self._check_degradations(analysis)
        self._check_overall_improvement(analysis)
        self._check_accuracy_scores(analysis)

    def _check_degradations(self, analysis: Dict[str, Any]) -> None:
        """Check for quality degradations and generate warnings."""
        for degradation in analysis["quality_improvement"]["degradations"]:
            if "original" in degradation and "processed" in degradation:
                analysis["warnings"].append(
                    f"Quality degradation in {degradation['metric']}: {degradation['degradation']:.1f}% "
                    f"(from {degradation['original']:.2f} to {degradation['processed']:.2f})"
                )
            else:
                analysis["warnings"].append(
                    f"Quality degradation in {degradation['metric']}: {degradation['degradation']:.1f}%"
                )

        if analysis["quality_improvement"]["degradations"]:
            analysis["recommendations"].append(
                "Consider adjusting enhancement parameters to address quality degradations"
            )

    def _check_overall_improvement(self, analysis: Dict[str, Any]) -> None:
        """Check overall improvement and generate warnings."""
        if "overall_improvement" not in analysis["quality_improvement"]:
            # Calculate overall improvement if not present
            improvements = analysis["quality_improvement"].get("improvements", [])
            degradations = analysis["quality_improvement"].get("degradations", [])
            
            if improvements or degradations:
                improvement_values = []
                if improvements:
                    improvement_values.extend([1.0] * len(improvements))
                if degradations:
                    improvement_values.extend([-1.0] * len(degradations))
                overall_improvement = np.mean(improvement_values) if improvement_values else 0.0
            else:
                overall_improvement = 0.0
                
            analysis["quality_improvement"]["overall_improvement"] = overall_improvement

        overall_improvement = analysis["quality_improvement"]["overall_improvement"]
        if overall_improvement < 0:
            analysis["warnings"].append(
                f"Overall quality decreased by {abs(overall_improvement):.1f}%"
            )
        elif overall_improvement < 1.0:
            analysis["warnings"].append("Minimal quality improvement detected")

    def _check_accuracy_scores(self, analysis: Dict[str, Any]) -> None:
        """Check accuracy scores and generate recommendations."""
        if analysis["accuracy_scores"]["overall_accuracy"] < 0.8:
            analysis["warnings"].append(
                f"Low overall accuracy: {analysis['accuracy_scores']['overall_accuracy']:.2f}"
            )
            analysis["recommendations"].append(
                "Consider using more aggressive enhancement settings"
            )

    def compare_metrics(
        self, original_metrics: Dict[str, float], processed_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare original and processed metrics.

        Args:
            original_metrics: Original image metrics
            processed_metrics: Processed image metrics

        Returns:
            Metrics comparison dictionary
        """
        comparison = {}
        for metric in original_metrics:
            original_value = original_metrics[metric]
            processed_value = processed_metrics[metric]
            improvement = processed_value - original_value
            improvement_percentage = (
                (improvement / original_value * 100) if original_value != 0 else 0
            )

            comparison[metric] = {
                "original": original_value,
                "processed": processed_value,
                "improvement": improvement,
                "improvement_percentage": improvement_percentage,
            }

        return comparison
