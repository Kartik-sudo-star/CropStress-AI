"""
Scripts Package
"""
from scripts.prepare_dataset import main as prepare_dataset_main
from scripts.validate_dataset import main as validate_dataset_main

__all__ = [
    "prepare_dataset_main",
    "validate_dataset_main",
]