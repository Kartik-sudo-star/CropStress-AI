#!/usr/bin/env python
"""
Dataset Preparation Script for Deepfake Detection

Creates manifest files and splits for deepfake detection datasets.
Supports FaceForensics++, DFDC, ASVspoof, and custom datasets.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import hashlib
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_path: str):
    """Configure logging."""
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(log_path, level="DEBUG", rotation="10 MB", retention=5)


def scan_dataset_directory(
    data_root: Path,
    dataset_name: str,
    media_type: str
) -> List[Dict]:
    """
    Scan dataset directory and create manifest entries.
    
    Expected structure:
    data_root/
        real/
            *.jpg, *.mp4, *.wav
        fake/
            *.jpg, *.mp4, *.wav
        OR
        <manipulation_type>/
            real/
            fake/
    """
    entries = []
    
    # Supported extensions
    if media_type == "image":
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    elif media_type == "video":
        extensions = {".mp4", ".mov", ".avi", ".mkv"}
    elif media_type == "audio":
        extensions = {".wav", ".mp3", ".m4a", ".flac"}
    else:
        extensions = set()
    
    # Check for real/fake structure
    real_dir = data_root / "real"
    fake_dir = data_root / "fake"
    
    if real_dir.exists() and fake_dir.exists():
        # Simple real/fake structure
        for label, label_dir in [("real", real_dir), ("fake", fake_dir)]:
            for ext in extensions:
                for file_path in label_dir.rglob(f"*{ext}"):
                    rel_path = file_path.relative_to(data_root)
                    entries.append({
                        "file_path": str(rel_path),
                        "label": label,
                        "dataset": dataset_name,
                        "manipulation": "original" if label == "real" else "unknown",
                        "split": "unknown"  # Will be set during split
                    })
    else:
        # Check for manipulation-type structure
        for manip_dir in data_root.iterdir():
            if not manip_dir.is_dir():
                continue
            
            manip_name = manip_dir.name
            for label, label_dir in [("real", manip_dir / "real"), ("fake", manip_dir / "fake")]:
                if label_dir.exists():
                    for ext in extensions:
                        for file_path in label_dir.rglob(f"*{ext}"):
                            rel_path = file_path.relative_to(data_root)
                            entries.append({
                                "file_path": str(rel_path),
                                "label": label,
                                "dataset": dataset_name,
                                "manipulation": manip_name,
                                "split": "unknown"
                            })
    
    logger.info(f"Scanned {dataset_name}: found {len(entries)} files")
    return entries


def compute_file_hashes(entries: List[Dict], data_root: Path) -> List[Dict]:
    """Compute SHA-256 hashes for all files."""
    for entry in entries:
        file_path = data_root / entry["file_path"]
        if file_path.exists():
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            entry["sha256"] = sha256.hexdigest()
            
            # Get file size
            entry["file_size"] = file_path.stat().st_size
        else:
            entry["sha256"] = ""
            entry["file_size"] = 0
    return entries


def split_dataset(
    entries: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    group_aware: bool = False,
    group_column: str = "manipulation",
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test."""
    df = pd.DataFrame(entries)
    
    if stratify:
        # Stratify by label
        stratify_col = df["label"]
    else:
        stratify_col = None
    
    if group_aware and group_column in df.columns:
        # Group-aware split: keep same group in same split
        groups = df[group_column].unique()
        np.random.seed(random_seed)
        np.random.shuffle(groups)
        
        n_groups = len(groups)
        train_groups = groups[:int(n_groups * train_ratio)]
        val_groups = groups[int(n_groups * train_ratio):int(n_groups * (train_ratio + val_ratio))]
        test_groups = groups[int(n_groups * (train_ratio + val_ratio)):]
        
        train_df = df[df[group_column].isin(train_groups)]
        val_df = df[df[group_column].isin(val_groups)]
        test_df = df[df[group_column].isin(test_groups)]
    else:
        # Standard split
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio),
            stratify=stratify_col, random_state=random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=val_size,
            stratify=temp_df["label"] if stratify else None,
            random_state=random_seed
        )
    
    # Add split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train label dist: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Val label dist: {val_df['label'].value_counts().to_dict()}")
    logger.info(f"Test label dist: {test_df['label'].value_counts().to_dict()}")
    
    return (
        train_df.to_dict("records"),
        val_df.to_dict("records"),
        test_df.to_dict("records")
    )


def save_manifests(
    train_entries: List[Dict],
    val_entries: List[Dict],
    test_entries: List[Dict],
    output_dir: Path
):
    """Save manifest CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, entries in [("train", train_entries), ("validation", val_entries), ("test", test_entries)]:
        df = pd.DataFrame(entries)
        df.to_csv(output_dir / f"{name}_manifest.csv", index=False)
        logger.info(f"Saved {name} manifest: {len(df)} entries")


def save_metadata(
    train_entries: List[Dict],
    val_entries: List[Dict],
    test_entries: List[Dict],
    config: Dict,
    output_dir: Path
):
    """Save dataset metadata."""
    metadata = {
        "train_size": len(train_entries),
        "val_size": len(val_entries),
        "test_size": len(test_entries),
        "train_ratio": config["dataset"]["split"]["train_ratio"],
        "val_ratio": config["dataset"]["split"]["val_ratio"],
        "test_ratio": config["dataset"]["split"]["test_ratio"],
        "stratify": config["dataset"]["split"]["stratify"],
        "group_aware": config["dataset"]["split"]["group_aware"],
        "split_by": config["dataset"]["split"].get("split_by", "source_video"),
        "label_classes": ["real", "fake"],
        "label_mapping": {"real": 0, "fake": 1},
        "datasets": list(set(e["dataset"] for e in train_entries + val_entries + test_entries)),
        "manipulations": list(set(e.get("manipulation", "unknown") for e in train_entries + val_entries + test_entries)),
    }
    
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Saved dataset metadata")


def main():
    parser = argparse.ArgumentParser(description="Prepare deepfake detection dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data-root", help="Root directory of dataset")
    parser.add_argument("--dataset-name", default="custom", help="Dataset name")
    parser.add_argument("--media-type", default="image", choices=["image", "video", "audio"])
    parser.add_argument("--output-dir", help="Output directory for manifests")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"] + "/prepare_dataset.log")
    
    # Determine paths
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(config["paths"]["data_raw"]) / args.media_type
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["paths"]["data_processed"])
    
    logger.info(f"Preparing dataset: {args.dataset_name}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")
    
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        sys.exit(1)
    
    # Scan dataset
    entries = scan_dataset_directory(data_root, args.dataset_name, args.media_type)
    
    if not entries:
        logger.error("No files found in dataset")
        sys.exit(1)
    
    # Compute hashes
    entries = compute_file_hashes(entries, data_root)
    
    # Split
    train_entries, val_entries, test_entries = split_dataset(
        entries,
        train_ratio=config["dataset"]["split"]["train_ratio"],
        val_ratio=config["dataset"]["split"]["val_ratio"],
        test_ratio=config["dataset"]["split"]["test_ratio"],
        stratify=config["dataset"]["split"]["stratify"],
        group_aware=config["dataset"]["split"]["group_aware"],
        group_column=config["dataset"]["split"].get("split_by", "manipulation"),
        random_seed=config["general"]["random_seed"]
    )
    
    # Save manifests
    save_manifests(train_entries, val_entries, test_entries, output_dir)
    
    # Save metadata
    save_metadata(train_entries, val_entries, test_entries, config, output_dir)
    
    # Copy files to split directories (optional - for easier access)
    for split_name, split_entries in [("train", train_entries), ("validation", val_entries), ("test", test_entries)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for entry in split_entries:
            src = data_root / entry["file_path"]
            dst = split_dir / entry["file_path"]
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import shutil
                    shutil.copy2(src, dst)
                except Exception as e:
                    logger.warning(f"Failed to copy {src}: {e}")
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()