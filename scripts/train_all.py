#!/usr/bin/env python
"""
Train All Models Script

Trains image, video, audio, and multimodal deepfake detectors sequentially.
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_training(script_name: str, config: str, resume: str = None, epochs: int = None) -> bool:
    """Run a training script."""
    cmd = [sys.executable, f"ml/training/{script_name}.py", "--config", config]
    
    if resume:
        cmd.extend(["--resume", resume])
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train all deepfake detection models")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--skip", nargs="+", choices=["image", "video", "audio", "multimodal"], 
                        default=[], help="Models to skip")
    parser.add_argument("--resume-image", help="Resume image training from checkpoint")
    parser.add_argument("--resume-video", help="Resume video training from checkpoint")
    parser.add_argument("--resume-audio", help="Resume audio training from checkpoint")
    parser.add_argument("--resume-multimodal", help="Resume multimodal training from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs for all models")
    parser.add_argument("--image-epochs", type=int, help="Override image epochs")
    parser.add_argument("--video-epochs", type=int, help="Override video epochs")
    parser.add_argument("--audio-epochs", type=int, help="Override audio epochs")
    parser.add_argument("--multimodal-epochs", type=int, help="Override multimodal epochs")
    args = parser.parse_args()
    
    models_to_train = [
        ("image", "train_image", args.resume_image, args.image_epochs or args.epochs),
        ("video", "train_video", args.resume_video, args.video_epochs or args.epochs),
        ("audio", "train_audio", args.resume_audio, args.audio_epochs or args.epochs),
        ("multimodal", "train_multimodal", args.resume_multimodal, args.multimodal_epochs or args.epochs),
    ]
    
    print("Deepfake Detection - Train All Models")
    print(f"Config: {args.config}")
    print(f"Skipping: {args.skip if args.skip else 'None'}")
    
    results = {}
    
    for model_name, script, resume, epochs in models_to_train:
        if model_name in args.skip:
            print(f"\nSkipping {model_name} (requested)")
            results[model_name] = "skipped"
            continue
        
        print(f"\n{'#'*60}")
        print(f"# Training {model_name.upper()} Detector")
        print(f"{'#'*60}")
        
        success = run_training(script, args.config, resume, epochs)
        results[model_name] = "success" if success else "failed"
        
        if not success:
            print(f"\n{model_name} training failed!")
            # Continue with other models
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for model, status in results.items():
        status_icon = "✓" if status == "success" else ("○" if status == "skipped" else "✗")
        print(f"  {status_icon} {model}: {status}")
    
    all_success = all(s in ["success", "skipped"] for s in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()