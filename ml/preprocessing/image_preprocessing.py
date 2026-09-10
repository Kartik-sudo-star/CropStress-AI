"""
Image preprocessing for deepfake detection
"""
import os
import hashlib
import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as T
from loguru import logger


@dataclass
class ImageMetadata:
    """Metadata extracted from image"""
    filename: str
    file_size: int
    sha256_hash: str
    width: int
    height: int
    channels: int
    format: str
    mode: str
    exif_data: Dict[str, Any]
    has_face: bool = False
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_confidence: float = 0.0


class ImagePreprocessor:
    """Preprocessor for image-based deepfake detection"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        face_detection_enabled: bool = True,
        min_face_size: int = 64,
        face_confidence_threshold: float = 0.9,
        device: str = "cpu"
    ):
        self.target_size = target_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.face_detection_enabled = face_detection_enabled
        self.min_face_size = min_face_size
        self.face_confidence_threshold = face_confidence_threshold
        self.device = device
        
        # Initialize face detector
        self.face_detector = None
        if face_detection_enabled:
            self._init_face_detector()
        
        # Build transforms
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        self.inference_transform = self._build_inference_transform()
    
    def _init_face_detector(self):
        """Initialize face detector (using OpenCV DNN)"""
        try:
            # Use OpenCV's DNN face detector
            model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(model_path):
                self.face_detector = cv2.CascadeClassifier(model_path)
            else:
                logger.warning("Haar cascade not found, face detection disabled")
                self.face_detection_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize face detector: {e}")
            self.face_detection_enabled = False
    
    def _build_train_transform(self) -> A.Compose:
        """Build training augmentation pipeline"""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.1),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
            A.Normalize(mean=self.normalization_mean, std=self.normalization_std),
            ToTensorV2(),
        ])
    
    def _build_val_transform(self) -> A.Compose:
        """Build validation transform (no augmentation)"""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=self.normalization_mean, std=self.normalization_std),
            ToTensorV2(),
        ])
    
    def _build_inference_transform(self) -> A.Compose:
        """Build inference transform (same as validation)"""
        return self._build_val_transform()
    
    def compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def extract_metadata(self, file_path: str) -> ImageMetadata:
        """Extract comprehensive metadata from image"""
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        sha256_hash = self.compute_hash(file_path)
        
        # Open with PIL for metadata
        with Image.open(file_path) as img:
            width, height = img.size
            channels = len(img.getbands())
            format_ = img.format
            mode = img.mode
            
            # Extract EXIF data
            exif_data = {}
            try:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        # Convert non-serializable values
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        exif_data[tag] = value
            except Exception:
                pass
        
        # Face detection
        has_face = False
        face_bbox = None
        face_confidence = 0.0
        
        if self.face_detection_enabled and self.face_detector is not None:
            img_cv = cv2.imread(file_path)
            if img_cv is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size)
                )
                if len(faces) > 0:
                    # Take the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face
                    has_face = True
                    face_bbox = (int(x), int(y), int(w), int(h))
                    face_confidence = 0.9  # Haar cascade doesn't give confidence
        
        return ImageMetadata(
            filename=filename,
            file_size=file_size,
            sha256_hash=sha256_hash,
            width=width,
            height=height,
            channels=channels,
            format=format_ or "unknown",
            mode=mode,
            exif_data=exif_data,
            has_face=has_face,
            face_bbox=face_bbox,
            face_confidence=face_confidence
        )
    
    def load_and_preprocess(self, file_path: str, mode: str = "inference") -> Tuple[torch.Tensor, ImageMetadata]:
        """Load image and apply preprocessing"""
        metadata = self.extract_metadata(file_path)
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop to face if detected
        if metadata.has_face and metadata.face_bbox:
            x, y, w, h = metadata.face_bbox
            # Add padding
            pad = int(0.2 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)
            image = image[y1:y2, x1:x2]
        
        # Apply transform
        if mode == "train":
            transform = self.train_transform
        elif mode == "val":
            transform = self.val_transform
        else:
            transform = self.inference_transform
        
        augmented = transform(image=image)
        tensor = augmented["image"]
        
        return tensor, metadata
    
    def preprocess_batch(self, file_paths: List[str], mode: str = "inference") -> Tuple[torch.Tensor, List[ImageMetadata]]:
        """Preprocess a batch of images"""
        tensors = []
        metadata_list = []
        
        for path in file_paths:
            tensor, meta = self.load_and_preprocess(path, mode)
            tensors.append(tensor)
            metadata_list.append(meta)
        
        return torch.stack(tensors), metadata_list
    
    def validate_image(self, file_path: str, max_size_mb: int = 10) -> Tuple[bool, str]:
        """Validate image file"""
        # Check file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
        
        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            return False, f"Unsupported format: {ext}"
        
        # Try to open
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            return False, f"Invalid image file: {e}"
        
        return True, "Valid"


class ImageForensicAnalyzer:
    """Forensic analysis of images for manipulation detection"""
    
    def __init__(self):
        pass
    
    def error_level_analysis(self, image_path: str, quality: int = 90) -> np.ndarray:
        """Perform Error Level Analysis (ELA)"""
        # Load original
        original = Image.open(image_path).convert('RGB')
        
        # Save as JPEG with specified quality
        import io
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        
        # Compute difference
        original_arr = np.array(original, dtype=np.float32)
        compressed_arr = np.array(compressed, dtype=np.float32)
        
        diff = np.abs(original_arr - compressed_arr)
        diff = diff.max(axis=2)  # Max across channels
        
        # Normalize
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff = np.zeros_like(diff, dtype=np.uint8)
        
        return diff
    
    def noise_analysis(self, image_path: str) -> Dict[str, float]:
        """Analyze noise patterns for manipulation detection"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {}
        
        # Estimate noise using Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = laplacian.var()
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # High frequency energy
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        high_freq = magnitude[center_h-30:center_h+30, center_w-30:center_w+30]
        high_freq_energy = np.mean(high_freq)
        
        return {
            "noise_variance": float(noise_level),
            "high_freq_energy": float(high_freq_energy),
            "noise_consistency": float(noise_level / (high_freq_energy + 1e-6))
        }
    
    def compression_analysis(self, image_path: str) -> Dict[str, Any]:
        """Analyze JPEG compression artifacts"""
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # DCT block analysis (8x8 blocks)
        h, w = y_channel.shape
        block_size = 8
        dct_coeffs = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = y_channel[i:i+block_size, j:j+block_size].astype(np.float32)
                dct = cv2.dct(block)
                dct_coeffs.append(dct.flatten())
        
        if dct_coeffs:
            dct_coeffs = np.array(dct_coeffs)
            # Analyze quantization patterns
            ac_coeffs = dct_coeffs[:, 1:]  # Skip DC coefficient
            zero_ratio = np.mean(ac_coeffs == 0)
            mean_ac = np.mean(np.abs(ac_coeffs))
            
            return {
                "zero_ac_ratio": float(zero_ratio),
                "mean_ac_magnitude": float(mean_ac),
                "estimated_quality": self._estimate_jpeg_quality(zero_ratio, mean_ac)
            }
        return {}
    
    def _estimate_jpeg_quality(self, zero_ratio: float, mean_ac: float) -> int:
        """Rough JPEG quality estimation"""
        # Heuristic based on AC coefficient statistics
        if zero_ratio > 0.95:
            return 95
        elif zero_ratio > 0.9:
            return 85
        elif zero_ratio > 0.8:
            return 75
        elif zero_ratio > 0.7:
            return 60
        elif zero_ratio > 0.5:
            return 40
        else:
            return 20
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Run all forensic analyses"""
        results = {}
        
        try:
            results["ela"] = self.error_level_analysis(image_path)
            results["noise"] = self.noise_analysis(image_path)
            results["compression"] = self.compression_analysis(image_path)
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            results["error"] = str(e)
        
        return results


# ==================== LEGACY-COMPATIBLE API ====================
# Interface expected by ml/training/train_image.py and backend/app/main.py:
#   preprocess_file(path, is_training, use_face_detection) -> Tensor[C, H, W]
#   save(path) / load(path) via pickle
#   create_preprocessor_from_config(config_dict)
#   InferenceImagePreprocessor(pkl_path)

def _image_preprocess_file(self, image_path: str, is_training: bool = False,
                           use_face_detection: bool = True) -> "torch.Tensor":
    """Load and preprocess a single image file (legacy signature)."""
    prev = self.face_detection_enabled
    self.face_detection_enabled = bool(use_face_detection) and prev
    try:
        mode = "train" if is_training else "inference"
        tensor, _ = self.load_and_preprocess(image_path, mode=mode)
        return tensor
    finally:
        self.face_detection_enabled = prev


def _image_get_params(self) -> Dict[str, Any]:
    return {
        "target_size": tuple(self.target_size),
        "normalization_mean": tuple(self.normalization_mean),
        "normalization_std": tuple(self.normalization_std),
        "face_detection_enabled": self.face_detection_enabled,
        "min_face_size": self.min_face_size,
        "face_confidence_threshold": self.face_confidence_threshold,
        "device": self.device,
    }


def _image_save(self, path: str):
    """Persist preprocessor params to pickle file."""
    import pickle
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(self._get_params(), f)


@classmethod
def _image_load(cls, path: str) -> "ImagePreprocessor":
    """Load preprocessor params from pickle file."""
    import pickle
    with open(path, "rb") as f:
        params = pickle.load(f)
    return cls(**params)


ImagePreprocessor.preprocess_file = _image_preprocess_file
ImagePreprocessor._get_params = _image_get_params
ImagePreprocessor.save = _image_save
ImagePreprocessor.load = _image_load


class InferenceImagePreprocessor(ImagePreprocessor):
    """Inference-time preprocessor restored from a saved pickle file."""

    def __init__(self, preprocessor_path: str):
        loaded = ImagePreprocessor.load(preprocessor_path)
        self.__dict__.update(loaded.__dict__)


def create_preprocessor_from_config(config: Dict[str, Any]) -> ImagePreprocessor:
    """Build ImagePreprocessor from the global config.yaml dict."""
    dataset_cfg = (config.get("dataset", {}) or {}) if isinstance(config, dict) else {}
    img_cfg = dataset_cfg.get("image", {}) or {}
    norm = img_cfg.get("normalization", {}) or {}
    face_cfg = img_cfg.get("face_detection", {}) or {}
    general = (config.get("general", {}) or {}) if isinstance(config, dict) else {}
    return ImagePreprocessor(
        target_size=tuple(img_cfg.get("target_size", [224, 224])),
        normalization_mean=tuple(norm.get("mean", [0.485, 0.456, 0.406])),
        normalization_std=tuple(norm.get("std", [0.229, 0.224, 0.225])),
        face_detection_enabled=face_cfg.get("enabled", True),
        min_face_size=face_cfg.get("min_face_size", 64),
        face_confidence_threshold=face_cfg.get("confidence_threshold", 0.9),
        device=general.get("device", "cpu"),
    )