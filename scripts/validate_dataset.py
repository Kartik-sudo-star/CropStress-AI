#!/usr/bin/env python
"""
Dataset Validation Script for Deepfake Detection

Validates dataset integrity, checks for data leakage, verifies splits.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import hashlib
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_path: str):
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(log_path, level="DEBUG", rotation="10 MB", retention=5)


def validate_manifest(manifest_path: Path) -> Tuple[bool, List[str]]:
    """Validate manifest file structure and content."""
    issues = []
    
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        return False, [f"Failed to read manifest: {e}"]
    
    # Required columns
    required_cols = ["file_path", "label", "split"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    if issues:
        return False, issues
    
    # Check labels
    valid_labels = {"real", "fake", "Real", "Fake", "0", "1"}
    invalid_labels = set(df["label"].unique()) - valid_labels
    if invalid_labels:
        issues.append(f"Invalid labels: {invalid_labels}")
    
    # Check splits
    valid_splits = {"train", "validation", "test"}
    invalid_splits = set(df["split"].unique()) - valid_splits
    if invalid_splits:
        issues.append(f"Invalid splits: {invalid_splits}")
    
    # Check for missing files
    missing_files = []
    for _, row in df.iterrows():
        # Would check actual file existence here
        pass
    
    # Check duplicates
    dup_paths = df[df.duplicated(subset=["file_path"], keep=False)]
    if not dup_paths.empty:
        issues.append(f"Duplicate file paths found: {len(dup_paths)} entries")
    
    # Check hash duplicates (same file, different paths)
    if "sha256" in df.columns:
        dup_hashes = df[df.duplicated(subset=["sha256"], keep=False)]
        if not dup_hashes.empty:
            issues.append(f"Duplicate file hashes (same content): {len(dup_hashes)} entries")
    
    return len(issues) == 0, issues


def check_data_leakage(
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    group_column: str = "manipulation"
) -> Tuple[bool, List[str]]:
    """Check for data leakage between splits."""
    issues = []
    
    try:
        train_df = pd.read_csv(train_manifest)
        val_df = pd.read_csv(val_manifest)
        test_df = pd.read_csv(test_manifest)
    except Exception as e:
        return False, [f"Failed to read manifests: {e}"]
    
    # Check file path overlap
    train_paths = set(train_df["file_path"])
    val_paths = set(val_df["file_path"])
    test_paths = set(test_df["file_path"])
    
    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths
    
    if train_val_overlap:
        issues.append(f"File path overlap between train/val: {len(train_val_overlap)} files")
    if train_test_overlap:
        issues.append(f"File path overlap between train/test: {len(train_test_overlap)} files")
    if val_test_overlap:
        issues.append(f"File path overlap between val/test: {len(val_test_overlap)} files")
    
    # Check hash overlap (same content)
    if "sha256" in train_df.columns:
        train_hashes = set(train_df["sha256"])
        val_hashes = set(val_df["sha256"])
        test_hashes = set(test_df["sha256"])
        
        hash_train_val = train_hashes & val_hashes
        hash_train_test = train_hashes & test_hashes
        hash_val_test = val_hashes & test_hashes
        
        if hash_train_val:
            issues.append(f"Hash overlap train/val: {len(hash_train_val)} files (same content)")
        if hash_train_test:
            issues.append(f"Hash overlap train/test: {len(hash_train_test)} files (same content)")
        if hash_val_test:
            issues.append(f"Hash overlap val/test: {len(hash_val_test)} files (same content)")
    
    # Check group leakage (e.g., same manipulation type across splits)
    if group_column in train_df.columns:
        train_groups = set(train_df[group_column].unique())
        val_groups = set(val_df[group_column].unique())
        test_groups = set(test_df[group_column].unique())
        
        group_overlap = train_groups & val_groups & test_groups
        if group_overlap:
            issues.append(f"Group ({group_column}) present in all splits: {group_overlap}")
    
    return len(issues) == 0, issues


def analyze_class_distribution(
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path
) -> Dict:
    """Analyze class distribution across splits."""
    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    test_df = pd.read_csv(test_manifest)
    
    def normalize_label(label):
        if str(label).lower() in ["real", "0"]:
            return "real"
        return "fake"
    
    for df in [train_df, val_df, test_df]:
        df["label_norm"] = df["label"].apply(normalize_label)
    
    return {
        "train": train_df["label_norm"].value_counts().to_dict(),
        "validation": val_df["label_norm"].value_counts().to_dict(),
        "test": test_df["label_norm"].value_counts().to_dict(),
        "train_total": len(train_df),
        "val_total": len(val_df),
        "test_total": len(test_df),
    }


def check_file_existence(manifest_path: Path, data_root: Path) -> Tuple[bool, List[str]]:
    """Check if all files in manifest exist."""
    issues = []
    df = pd.read_csv(manifest_path)
    
    missing = []
    for _, row in df.iterrows():
        file_path = data_root / row["file_path"]
        if not file_path.exists():
            missing.append(str(row["file_path"]))
    
    if missing:
        issues.append(f"Missing files: {len(missing)} (first 10: {missing[:10]})")
    
    return len(issues) == 0, issues


def verify_hashes(manifest_path: Path, data_root: Path) -> Tuple[bool, List[str]]:
    """Verify SHA-256 hashes match file content."""
    issues = []
    df = pd.read_csv(manifest_path)
    
    if "sha256" not in df.columns:
        return True, ["No SHA-256 column to verify"]
    
    mismatches = []
    for _, row in df.iterrows():
        file_path = data_root / row["file_path"]
        if file_path.exists() and row["sha256"]:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            computed = sha256.hexdigest()
            if computed != row["sha256"]:
                mismatches.append(str(row["file_path"]))
    
    if mismatches:
        issues.append(f"Hash mismatches: {len(mismatches)} files (first 10: {mismatches[:10]})")
    
    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Validate deepfake detection dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--manifest", help="Single manifest to validate")
    parser.add_argument("--train-manifest", help="Train manifest path")
    parser.add_argument("--val-manifest", help="Validation manifest path")
    parser.add_argument("--test-manifest", help="Test manifest path")
    parser.add_argument("--data-root", help="Root directory for file existence check")
    parser.add_argument("--output", help="Output validation report JSON")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config["paths"]["logs_dir"] + "/validate_dataset.log")
    
    all_issues = []
    results = {}
    
    # Single manifest validation
    if args.manifest:
        manifest_path = Path(args.manifest)
        logger.info(f"Validating manifest: {manifest_path}")
        valid, issues = validate_manifest(manifest_path)
        results["manifest_validation"] = {"valid": valid, "issues": issues}
        all_issues.extend(issues)
    
    # Split validation
    if args.train_manifest and args.val_manifest and args.test_manifest:
        logger.info("Checking data leakage...")
        valid, issues = check_data_leakage(
            Path(args.train_manifest),
            Path(args.val_manifest),
            Path(args.test_manifest),
            group_column=config["dataset"]["split"].get("split_by", "manipulation")
        )
        results["leakage_check"] = {"valid": valid, "issues": issues}
        all_issues.extend(issues)
        
        logger.info("Analyzing class distribution...")
        dist = analyze_class_distribution(
            Path(args.train_manifest),
            Path(args.val_manifest),
            Path(args.test_manifest)
        )
        results["class_distribution"] = dist
        logger.info(f"Class distribution: {dist}")
    
    # File existence check
    if args.data_root and args.manifest:
        data_root = Path(args.data_root)
        logger.info("Checking file existence...")
        valid, issues = check_file_existence(Path(args.manifest), data_root)
        results["file_existence"] = {"valid": valid, "issues": issues}
        all_issues.extend(issues)
        
        logger.info("Verifying hashes...")
        valid, issues = verify_hashes(Path(args.manifest), data_root)
        results["hash_verification"] = {"valid": valid, "issues": issues}
        all_issues.extend(issues)
    
    # Summary
    results["summary"] = {
        "total_issues": len(all_issues),
        "passed": len(all_issues) == 0,
        "issues": all_issues
    }
    
    logger.info(f"Validation complete. Total issues: {len(all_issues)}")
    if all_issues:
        for issue in all_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.success("All validations passed!")
    
    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation report saved to {args.output}")
    
    # Exit code
    sys.exit(0 if len(all_issues) == 0 else 1)


if __name__ == "__main__":
    main()