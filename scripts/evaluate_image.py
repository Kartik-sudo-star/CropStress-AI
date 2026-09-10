#!/usr/bin/env python
"""Quick evaluation of trained image detector on test split."""
import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.image_detector import load_image_detector
from ml.preprocessing.image_preprocessing import ImagePreprocessor, create_preprocessor_from_config
from ml.evaluation.evaluator import ModelEvaluator, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from ml.training.train_image import create_data_loaders


def main():
    config_path = "config.local.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["general"]["device"])
    
    # Create preprocessor and test loader
    preprocessor = create_preprocessor_from_config(config)
    train_loader, val_loader, test_loader = create_data_loaders(
        config, preprocessor,
        config["image_detector"]["training"]["batch_size"],
        config["general"]["num_workers"]
    )
    
    # Load best model
    model_path = Path(config["paths"]["models_dir"]) / "image_detector" / "best_model.pth"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    model = load_image_detector(str(model_path), config, device)
    evaluator = ModelEvaluator(model, device)
    
    print("Evaluating on test set...")
    result = evaluator.evaluate_dataset(test_loader, return_details=True)
    metrics = result["metrics"]
    
    print("\n=== TEST METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Threshold analysis
    print("\nThreshold analysis...")
    thresh = evaluator.threshold_analysis(test_loader)
    print(f"  Best threshold (F1): {thresh['best_f1_threshold']['threshold']:.3f}")
    print(f"  Best Youden: {thresh['best_youden_threshold']['threshold']:.3f}")
    
    # Save plots
    out_dir = Path("docs/evaluation_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=["Real", "Fake"],
                          save_path=out_dir / "confusion_matrix.png")
    
    # Compute and save ROC/PR curves using evaluator plotting functions
    plot_roc_curve(result["labels"], result["probabilities"], save_path=out_dir / "roc_curve.png")
    plot_pr_curve(result["labels"], result["probabilities"], save_path=out_dir / "pr_curve.png")
    
    print(f"\nPlots saved to {out_dir}")
    
    # Also save metrics JSON
    import json
    with open(out_dir / "test_metrics.json", "w") as f:
        # Convert numpy arrays to lists
        def convert(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(metrics), f, indent=2)
    print("Metrics saved to docs/evaluation_plots/test_metrics.json")


if __name__ == "__main__":
    main()