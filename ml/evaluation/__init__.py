"""
Evaluation Package
"""
from ml.evaluation.evaluator import (
    ModelEvaluator,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_det_curve,
    plot_threshold_analysis,
    compare_models,
    generate_evaluation_report,
    evaluate_all_models
)

__all__ = [
    "ModelEvaluator",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_det_curve",
    "plot_threshold_analysis",
    "compare_models",
    "generate_evaluation_report",
    "evaluate_all_models",
]