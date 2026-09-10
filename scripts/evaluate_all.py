#!/usr/bin/env python
"""
Evaluate All Models Script

Evaluates all trained models on test set and generates comparison report.
"""

import argparse
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models import (
    load_image_detector, load_video_detector,
    load_audio_detector, load_multimodal_detector
)
from ml.preprocessing import (
    InferenceImagePreprocessor, VideoPreprocessor, AudioPreprocessor
)
from ml.evaluation import ModelEvaluator, generate_evaluation_report, compare_models
from ml.evidence import create_evidence_manager


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation."""
    
    def __init__(self, manifest_path: str, preprocessor, file_type: str):
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        self.preprocessor = preprocessor
        self.file_type = file_type
        self.label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, "0": 0, "1": 1}
        self.df["label_idx"] = self.df["label"].map(self.label_map)
        self.df = self.df.dropna(subset=["label_idx"])
        self.df["label_idx"] = self.df["label_idx"].astype(int)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.file_type == "image":
            tensor = self.preprocessor.preprocess_file(row["file_path"])
            return tensor, row["label_idx"]
        elif self.file_type == "video":
            video_tensor, _, _ = self.preprocessor.preprocess(row["video_path"])
            return video_tensor, row["label_idx"]
        elif self.file_type == "audio":
            if hasattr(self.preprocessor, 'preprocess'):
                if self.preprocessor.__class__.__name__ == "AudioPreprocessor":
                    waveform, mel = self.preprocessor.preprocess(row["audio_path"], return_waveform=True)
                    return waveform, mel, row["label_idx"]
            mel = self.preprocessor.preprocess(row["audio_path"])
            return mel, row["label_idx"]
        
        return None, row["label_idx"]


def collate_video(batch):
    """Collate function for video batch."""
    videos = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Pad to same length
    max_t = max(v.shape[0] for v in videos)
    padded = []
    for v in videos:
        if v.shape[0] < max_t:
            pad = torch.zeros(max_t - v.shape[0], *v.shape[1:])
            v = torch.cat([v, pad], dim=0)
        padded.append(v)
    return torch.stack(padded), torch.tensor(labels)


def collate_audio(batch):
    """Collate function for audio batch."""
    if len(batch[0]) == 3:
        # (waveform, mel, label)
        waveforms = [item[0] for item in batch]
        mels = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        return torch.stack(waveforms), torch.stack(mels), torch.tensor(labels)
    else:
        # (mel, label)
        mels = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(mels), torch.tensor(labels)


def evaluate_model(model_name: str, config: dict, device: torch.device) -> dict:
    """Evaluate a single model."""
    paths = config["paths"]
    model_dir = Path(paths["models_dir"])
    
    if model_name == "image":
        model_path = model_dir / "image_detector" / "best_model.pth"
        preproc_path = model_dir / "image_detector" / "preprocessor.pkl"
        
        if not model_path.exists():
            return {"error": "Model not found"}
        
        model = load_image_detector(str(model_path), config, device)
        preprocessor = InferenceImagePreprocessor(str(preproc_path))
        
        # Create dataset
        from ml.evaluation.evaluator import ModelEvaluator
        import pandas as pd
        
        test_df = pd.read_csv(Path(paths["data_test"]) / "manifest.csv")
        # Filter for image type
        test_df = test_df[test_df.get("media_type", "image") == "image"]
        
        # Create dataloader
        dataset = SimpleDataset(
            str(Path(paths["data_test"]) / "manifest.csv"),
            preprocessor, "image"
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        
        evaluator = ModelEvaluator(model, device)
        metrics = evaluator.evaluate_dataset(loader)
        
        return metrics
    
    elif model_name == "video":
        model_path = model_dir / "video_detector" / "best_model.pth"
        preproc_path = model_dir / "video_detector" / "preprocessor.pkl"
        
        if not model_path.exists():
            return {"error": "Model not found"}
        
        model = load_video_detector(str(model_path), config, device)
        preprocessor = VideoPreprocessor.load(str(preproc_path))
        
        dataset = SimpleDataset(
            str(Path(paths["data_test"]) / "manifest.csv"),
            preprocessor, "video"
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_video)
        
        evaluator = ModelEvaluator(model, device)
        metrics = evaluator.evaluate_dataset(loader)
        
        return metrics
    
    elif model_name == "audio":
        model_path = model_dir / "audio_detector" / "best_model.pth"
        preproc_path = model_dir / "audio_detector" / "preprocessor.pkl"
        
        if not model_path.exists():
            return {"error": "Model not found"}
        
        model = load_audio_detector(str(model_path), config, device)
        preprocessor = AudioPreprocessor.load(str(preproc_path))
        
        dataset = SimpleDataset(
            str(Path(paths["data_test"]) / "manifest.csv"),
            preprocessor, "audio"
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_audio)
        
        evaluator = ModelEvaluator(model, device)
        metrics = evaluator.evaluate_dataset(loader)
        
        return metrics
    
    elif model_name == "multimodal":
        model_path = model_dir / "multimodal_detector" / "best_model.pth"
        
        if not model_path.exists():
            return {"error": "Model not found"}
        
        model = load_multimodal_detector(str(model_path), config, device)
        # Would need both video and audio preprocessors
        # Simplified for now
        return {"error": "Multimodal evaluation not fully implemented"}
    
    return {"error": f"Unknown model: {model_name}"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", help="Output report path")
    args = parser.parse_args()
    
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["general"]["device"])
    
    models_to_eval = ["image", "video", "audio", "multimodal"]
    results = {}
    
    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name}...")
        try:
            metrics = evaluate_model(model_name, config, device)
            results[model_name] = metrics
            if "error" not in metrics:
                print(f"  AUC: {metrics.get('roc_auc', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")
            else:
                print(f"  {metrics['error']}")
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {"error": str(e)}
    
    # Generate comparison report
    if args.output:
        # Add metrics key for compare_models
        comparison_data = {}
        for name, metrics in results.items():
            if "error" not in metrics:
                comparison_data[name] = {"metrics": metrics}
        
        if comparison_data:
            report = generate_evaluation_report(
                "Model Comparison",
                {},  # Will be filled by compare_models
                save_path=args.output
            )
            print(f"\nComparison report saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        if "error" in metrics:
            print(f"  {name}: FAILED - {metrics['error']}")
        else:
            print(f"  {name}: AUC={metrics.get('roc_auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}, Acc={metrics.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main()