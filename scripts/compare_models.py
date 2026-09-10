#!/usr/bin/env python
"""
Model Comparison Script

Generates detailed comparison report between Model A (Image Only), 
Model B (Sensor Only), and Model C (Multimodal).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger
import numpy as np


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path) as f:
        return json.load(f)


def compare_models(metrics_dict: Dict) -> Dict:
    """
    Compare models and generate insights.
    
    Returns:
        Comparison analysis dictionary
    """
    analysis = {
        "best_overall": None,
        "best_per_class": {},
        "improvements": {},
        "recommendations": [],
    }
    
    model_names = list(metrics_dict.keys())
    
    # Find best overall model (by F1 macro)
    best_f1 = -1
    best_model = None
    for name, metrics in metrics_dict.items():
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model = name
    
    analysis["best_overall"] = {
        "model": best_model,
        "f1_macro": best_f1,
    }
    
    # Per-class best
    class_names = metrics_dict[model_names[0]].get("class_labels", [])
    for cls in class_names:
        best_f1_cls = -1
        best_model_cls = None
        for name, metrics in metrics_dict.items():
            f1 = metrics["per_class_f1"].get(cls, 0)
            if f1 > best_f1_cls:
                best_f1_cls = f1
                best_model_cls = name
        analysis["best_per_class"][cls] = {
            "model": best_model_cls,
            "f1": best_f1_cls,
        }
    
    # Calculate improvements of multimodal over unimodal
    if "Multimodal" in metrics_dict and "Image Only" in metrics_dict and "Sensor Only" in metrics_dict:
        multi = metrics_dict["Multimodal"]
        image = metrics_dict["Image Only"]
        sensor = metrics_dict["Sensor Only"]
        
        analysis["improvements"]["multimodal_vs_image"] = {
            "f1_macro_diff": multi["f1_macro"] - image["f1_macro"],
            "accuracy_diff": multi["accuracy"] - image["accuracy"],
            "relative_improvement_pct": ((multi["f1_macro"] - image["f1_macro"]) / image["f1_macro"]) * 100,
        }
        
        analysis["improvements"]["multimodal_vs_sensor"] = {
            "f1_macro_diff": multi["f1_macro"] - sensor["f1_macro"],
            "accuracy_diff": multi["accuracy"] - sensor["accuracy"],
            "relative_improvement_pct": ((multi["f1_macro"] - sensor["f1_macro"]) / sensor["f1_macro"]) * 100,
        }
        
        analysis["improvements"]["image_vs_sensor"] = {
            "f1_macro_diff": image["f1_macro"] - sensor["f1_macro"],
            "accuracy_diff": image["accuracy"] - sensor["accuracy"],
        }
    
    # Generate recommendations
    if "Multimodal" in metrics_dict:
        multi_f1 = metrics_dict["Multimodal"]["f1_macro"]
        image_f1 = metrics_dict.get("Image Only", {}).get("f1_macro", 0)
        sensor_f1 = metrics_dict.get("Sensor Only", {}).get("f1_macro", 0)
        
        if multi_f1 > max(image_f1, sensor_f1) + 0.02:
            analysis["recommendations"].append(
                "Multimodal fusion significantly outperforms unimodal models. "
                "Use multimodal approach for production deployment."
            )
        elif multi_f1 > max(image_f1, sensor_f1):
            analysis["recommendations"].append(
                "Multimodal shows marginal improvement. "
                "Consider if added complexity is justified for your use case."
            )
        else:
            analysis["recommendations"].append(
                "Multimodal does not outperform best unimodal model. "
                "Investigate fusion strategy or use best unimodal model."
            )
        
        # Check if one modality dominates
        if image_f1 > sensor_f1 + 0.05:
            analysis["recommendations"].append(
                "Image-only model significantly outperforms sensor-only. "
                "Visual features are more discriminative for this dataset."
            )
        elif sensor_f1 > image_f1 + 0.05:
            analysis["recommendations"].append(
                "Sensor-only model significantly outperforms image-only. "
                "Environmental features are more predictive for this dataset."
            )
        else:
            analysis["recommendations"].append(
                "Both modalities contribute similarly. "
                "Multimodal fusion is well-justified."
            )
    
    return analysis


def generate_comparison_report(
    metrics_dict: Dict,
    analysis: Dict,
    output_path: Path,
) -> str:
    """Generate detailed comparison report in Markdown."""
    
    lines = [
        "# Model Comparison Report\n",
        "## Executive Summary\n",
        f"Best overall model: **{analysis['best_overall']['model']}** (F1-macro: {analysis['best_overall']['f1_macro']:.4f})\n\n",
        
        "## Overall Metrics Comparison\n",
        "| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | Precision (W) | Recall (W) | F1 (W) |\n",
        "|-------|----------|-------------------|----------------|------------|---------------|------------|--------|\n",
    ]
    
    for name, metrics in metrics_dict.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision_macro']:.4f} | {metrics['recall_macro']:.4f} | "
            f"{metrics['f1_macro']:.4f} | {metrics['precision_weighted']:.4f} | "
            f"{metrics['recall_weighted']:.4f} | {metrics['f1_weighted']:.4f} |\n"
        )
    
    lines.append("\n## Per-Class Best Model\n")
    lines.append("| Class | Best Model | F1 Score |\n")
    lines.append("|-------|------------|----------|\n")
    
    for cls, info in analysis["best_per_class"].items():
        lines.append(f"| {cls} | {info['model']} | {info['f1']:.4f} |\n")
    
    lines.append("\n## Improvement Analysis\n")
    
    if "multimodal_vs_image" in analysis["improvements"]:
        imp = analysis["improvements"]["multimodal_vs_image"]
        lines.append(f"### Multimodal vs Image Only\n")
        lines.append(f"- F1-macro improvement: **{imp['f1_macro_diff']:+.4f}** ({imp['relative_improvement_pct']:+.1f}%)\n")
        lines.append(f"- Accuracy improvement: **{imp['accuracy_diff']:+.4f}**\n\n")
    
    if "multimodal_vs_sensor" in analysis["improvements"]:
        imp = analysis["improvements"]["multimodal_vs_sensor"]
        lines.append(f"### Multimodal vs Sensor Only\n")
        lines.append(f"- F1-macro improvement: **{imp['f1_macro_diff']:+.4f}** ({imp['relative_improvement_pct']:+.1f}%)\n")
        lines.append(f"- Accuracy improvement: **{imp['accuracy_diff']:+.4f}**\n\n")
    
    if "image_vs_sensor" in analysis["improvements"]:
        imp = analysis["improvements"]["image_vs_sensor"]
        lines.append(f"### Image Only vs Sensor Only\n")
        lines.append(f"- F1-macro difference: **{imp['f1_macro_diff']:+.4f}**\n")
        lines.append(f"- Accuracy difference: **{imp['accuracy_diff']:+.4f}**\n\n")
    
    lines.append("## Recommendations\n")
    for i, rec in enumerate(analysis["recommendations"], 1):
        lines.append(f"{i}. {rec}\n")
    
    lines.append("\n## Statistical Significance Notes\n")
    lines.append("- All metrics computed on held-out test set\n")
    lines.append("- Per-class metrics use macro averaging\n")
    lines.append("- Confidence intervals not computed (requires multiple runs)\n")
    lines.append("- Statistical significance testing requires repeated experiments\n")
    
    report = "".join(lines)
    
    with open(output_path, "w") as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--metrics", default="docs/evaluation/all_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--output", default="docs/model_comparison_report.md", help="Output report path")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    metrics = load_metrics(metrics_path)
    logger.info(f"Loaded metrics for {len(metrics)} models")
    
    analysis = compare_models(metrics)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = generate_comparison_report(metrics, analysis, output_path)
    
    # Also save analysis as JSON
    analysis_path = output_path.with_suffix(".json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Comparison report saved to {output_path}")
    logger.info(f"Analysis saved to {analysis_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Best Model: {analysis['best_overall']['model']} (F1: {analysis['best_overall']['f1_macro']:.4f})")
    
    if "multimodal_vs_image" in analysis["improvements"]:
        imp = analysis["improvements"]["multimodal_vs_image"]
        print(f"Multimodal vs Image: F1 diff = {imp['f1_macro_diff']:+.4f} ({imp['relative_improvement_pct']:+.1f}%)")
    
    if "multimodal_vs_sensor" in analysis["improvements"]:
        imp = analysis["improvements"]["multimodal_vs_sensor"]
        print(f"Multimodal vs Sensor: F1 diff = {imp['f1_macro_diff']:+.4f} ({imp['relative_improvement_pct']:+.1f}%)")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis["recommendations"], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()