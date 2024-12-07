"""Tests for quality improvement analysis."""


def test_quality_improvement_analysis(quality_manager):
    """Test quality improvement analysis."""
    metrics_comparison = {
        "sharpness": {
            "original": 0.5,
            "processed": 0.6,
            "improvement": 0.1,
            "improvement_percentage": 20.0,
        },
        "contrast": {
            "original": 0.7,
            "processed": 0.65,
            "improvement": -0.05,
            "improvement_percentage": -7.14,
        },
        "detail": {
            "original": 0.8,
            "processed": 0.805,
            "improvement": 0.005,
            "improvement_percentage": 0.625,
        },
    }

    analysis = quality_manager._analyze_quality_improvement(metrics_comparison)

    # Check analysis structure
    assert isinstance(analysis["overall_improvement"], float)
    assert isinstance(analysis["significant_improvements"], list)
    assert isinstance(analysis["degradations"], list)
    assert isinstance(analysis["stable_metrics"], list)

    # Check categorization
    assert len(analysis["significant_improvements"]) == 1  # sharpness
    assert len(analysis["degradations"]) == 1  # contrast
    assert len(analysis["stable_metrics"]) == 1  # detail


def test_analysis_feedback_generation(quality_manager):
    """Test analysis feedback generation."""
    analysis = {
        "accuracy_scores": {
            "structural_similarity": 0.8,
            "color_accuracy": 0.85,
            "overall_accuracy": 0.75,
        },
        "quality_improvement": {
            "degradations": [
                {"metric": "sharpness", "degradation": 10.5},
                {"metric": "contrast", "degradation": 7.2},
            ]
        },
        "warnings": [],
        "recommendations": [],
    }

    quality_manager._generate_analysis_feedback(analysis)

    # Check warnings
    assert len(analysis["warnings"]) >= 3  # 2 degradations + low overall accuracy

    # Check recommendations
    assert len(analysis["recommendations"]) >= 2  # General adjustment + specific parameters
