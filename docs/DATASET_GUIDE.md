# Dataset Guide for Deepfake Detection

This guide explains how to obtain and prepare datasets for training the deepfake detection models.

## Supported Datasets

### 1. FaceForensics++ (Image/Video)
- **Source**: https://github.com/ondyari/FaceForensics
- **Size**: ~1000 original videos, 4 manipulation methods
- **Manipulations**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Splits**: Pre-defined train/val/test splits
- **License**: Research only

**Download**:
```bash
# Run the official download script
python download_faceforensics.py --data_dir data/raw --methods Deepfakes Face2Face FaceSwap NeuralTextures
```

**Structure after download**:
```
data/raw/faceforensics/
├── original_sequences/youtube/c23/videos/
├── manipulated_sequences/Deepfakes/c23/videos/
├── manipulated_sequences/Face2Face/c23/videos/
├── manipulated_sequences/FaceSwap/c23/videos/
└── manipulated_sequences/NeuralTextures/c23/videos/
```

### 2. DFDC (DeepFake Detection Challenge) - Video
- **Source**: https://ai.googleblog.com/2019/09/deepfake-detection-challenge.html
- **Kaggle**: https://www.kaggle.com/c/deepfake-detection-challenge
- **Size**: ~100k videos
- **License**: DFDC Dataset License

**Download**:
```bash
# Using Kaggle API
kaggle competitions download -c deepfake-detection-challenge
unzip deepfake-detection-challenge.zip -d data/raw/dfdc
```

### 3. ASVspoof 2019/2021 - Audio
- **Source**: https://www.asvspoof.org/
- **Size**: Large audio dataset for spoofing detection
- **License**: Research only

**Download**:
```bash
# Register at asvspoof.org and download
# Place in data/raw/asvspoof2019/
```

### 4. Celeb-DF (Video)
- **Source**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Size**: 590 original, 5639 deepfake videos
- **License**: Research only

### 5. Custom Datasets
Place your data in the following structure:
```
data/raw/
├── images/
│   ├── real/
│   │   ├── img1.jpg
│   │   └── ...
│   └── fake/
│       ├── img1.jpg
│       └── ...
├── videos/
│   ├── real/
│   └── fake/
└── audio/
    ├── real/
    └── fake/
```

## Dataset Preparation

### 1. Configure config.yaml
Edit the dataset section in `config.yaml`:
```yaml
dataset:
  split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    stratify: true
    group_aware: true
    split_by: "source_video"
```

### 2. Run Preparation Script
```bash
# For custom dataset
python scripts/prepare_dataset.py --config config.yaml \
  --data-root data/raw/images \
  --dataset-name my_dataset \
  --media-type image

# For video
python scripts/prepare_dataset.py --config config.yaml \
  --data-root data/raw/videos \
  --dataset-name my_dataset \
  --media-type video
```

### 3. Validate Dataset
```bash
python scripts/validate_dataset.py --config config.yaml \
  --train-manifest data/processed/train_manifest.csv \
  --val-manifest data/processed/validation_manifest.csv \
  --test-manifest data/processed/test_manifest.csv \
  --data-root data/raw
```

## Manifest Format

The preparation script creates CSV manifests with these columns:

| Column | Description |
|--------|-------------|
| `file_path` | Relative path from data root |
| `label` | `real` or `fake` |
| `dataset` | Dataset name |
| `manipulation` | Manipulation type (Deepfakes, Face2Face, etc.) |
| `split` | `train`, `validation`, or `test` |
| `sha256` | SHA-256 hash of file |
| `file_size` | File size in bytes |

Example:
```csv
file_path,label,dataset,manipulation,split,sha256,file_size
real/img001.jpg,real,faceforensics,original,train,a1b2c3...,102400
fake/img002.jpg,fake,faceforensics,Deepfakes,train,d4e5f6...,204800
```

## Data Leakage Prevention

**Critical**: The system uses group-aware splitting to prevent data leakage:
- Videos from the same source are kept in the same split
- Frames from the same video never appear in different splits
- Same identity/person doesn't appear in train and test

Configure in `config.yaml`:
```yaml
dataset:
  split:
    group_aware: true
    split_by: "source_video"  # or "identity", "session"
```

## Class Imbalance Handling

The system automatically handles class imbalance:
```yaml
dataset:
  class_balance:
    enabled: true
    method: "weighted_sampler"  # or "oversample", "class_weights"
```

## Robustness Testing

The evaluation includes testing against common transformations:
- JPEG compression (quality 75, 50, 30, 10)
- Resizing (0.5x, 0.25x)
- Gaussian noise
- Blur
- Frame dropping (video)
- MP3 compression (audio)

## Dataset Statistics

After preparation, check `data/processed/dataset_metadata.json`:
```json
{
  "train_size": 10000,
  "val_size": 2000,
  "test_size": 2000,
  "label_classes": ["real", "fake"],
  "label_mapping": {"real": 0, "fake": 1},
  "datasets": ["faceforensics", "dfdc"],
  "manipulations": ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
}
```

## Troubleshooting

### Common Issues

1. **Missing files**: Check that downloaded files match manifest paths
2. **Hash mismatches**: Re-download corrupted files
3. **Class imbalance**: Enable `class_balance` in config
4. **Memory issues**: Reduce batch size in config
5. **Split leakage**: Ensure `group_aware: true` and check `validate_dataset.py` output

### Validation Report

Run validation to get detailed report:
```bash
python scripts/validate_dataset.py --output validation_report.json
```

Check for:
- File path overlaps between splits
- Hash overlaps (same content in different splits)
- Group leakage (same manipulation/source across splits)
- Missing files
- Label distribution balance

## Citation

If using these datasets in research, please cite the original papers:

- FaceForensics++: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
- DFDC: Dolhansky et al., "The Deepfake Detection Challenge (DFDC) Preview Dataset", arXiv 2020
- ASVspoof: Yamagishi et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection", Proc. Interspeech 2019