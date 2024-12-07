"""Tests for processing accuracy analysis."""


def test_processing_accuracy_analysis(quality_manager, test_image, processed_image):
    """Test processing accuracy analysis."""
    analysis = quality_manager.analyze_processing_accuracy(test_image, processed_image)

    # Check analysis structure
    assert "metrics_comparison" in analysis
    assert "accuracy_scores" in analysis
    assert "quality_improvement" in analysis
    assert "warnings" in analysis
    assert "recommendations" in analysis

    # Check metrics comparison
    for metric, comparison in analysis["metrics_comparison"].items():
        assert "original" in comparison
        assert "processed" in comparison
        assert "improvement" in comparison
        assert "improvement_percentage" in comparison

    # Check accuracy scores
    scores = analysis["accuracy_scores"]
    assert "structural_similarity" in scores
    assert "psnr" in scores
    assert "feature_preservation" in scores
    assert "color_accuracy" in scores
    assert "overall_accuracy" in scores

    # Check quality improvement
    improvement = analysis["quality_improvement"]
    assert "overall_improvement" in improvement
    assert "significant_improvements" in improvement
    assert "degradations" in improvement
    assert "stable_metrics" in improvement


def test_accuracy_scores_calculation(quality_manager, test_image, processed_image):
    """Test accuracy scores calculation."""
    scores = quality_manager._calculate_accuracy_scores(
        test_image,
        processed_image,
        quality_manager.calculate_quality_metrics(test_image),
        quality_manager.calculate_quality_metrics(processed_image),
    )

    # Check all scores
    assert 0 <= scores["structural_similarity"] <= 1
    assert scores["psnr"] >= 0
    assert 0 <= scores["feature_preservation"] <= 1
    assert 0 <= scores["color_accuracy"] <= 1
    assert 0 <= scores["overall_accuracy"] <= 1
