"""
Video preprocessing for deepfake detection
"""
import os
import hashlib
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as T
from loguru import logger


@dataclass
class VideoMetadata:
    """Metadata extracted from video"""
    filename: str
    file_size: int
    sha256_hash: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    format: str
    has_audio: bool
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]
    face_detection_results: List[Dict[str, Any]] = None


@dataclass
class FrameData:
    """Data for a single frame"""
    frame_index: int
    timestamp: float
    image: np.ndarray
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_confidence: float = 0.0


class VideoPreprocessor:
    """Preprocessor for video-based deepfake detection"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        frame_sampling_method: str = "uniform",
        num_frames: int = 16,
        min_frames: int = 8,
        face_detection_enabled: bool = True,
        min_face_size: int = 64,
        face_confidence_threshold: float = 0.9,
        max_duration_seconds: int = 60,
        device: str = "cpu"
    ):
        self.target_size = target_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.frame_sampling_method = frame_sampling_method
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.face_detection_enabled = face_detection_enabled
        self.min_face_size = min_face_size
        self.face_confidence_threshold = face_confidence_threshold
        self.max_duration_seconds = max_duration_seconds
        self.device = device
        
        # Initialize face detector
        self.face_detector = None
        if face_detection_enabled:
            self._init_face_detector()
        
        # Build transforms
        self.frame_transform = self._build_frame_transform()
    
    def _init_face_detector(self):
        """Initialize face detector"""
        try:
            model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(model_path):
                self.face_detector = cv2.CascadeClassifier(model_path)
            else:
                logger.warning("Haar cascade not found, face detection disabled")
                self.face_detection_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize face detector: {e}")
            self.face_detection_enabled = False
    
    def _build_frame_transform(self) -> A.Compose:
        """Build frame transform"""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=self.normalization_mean, std=self.normalization_std),
            ToTensorV2(),
        ])
    
    def compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def extract_metadata(self, file_path: str) -> VideoMetadata:
        """Extract comprehensive metadata from video"""
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        sha256_hash = self.compute_hash(file_path)
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec info
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        # Try to get more info with ffprobe if available
        format_ = os.path.splitext(file_path)[1].lstrip('.')
        has_audio = False
        audio_codec = None
        audio_sample_rate = None
        audio_channels = None
        
        return VideoMetadata(
            filename=filename,
            file_size=file_size,
            sha256_hash=sha256_hash,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            format=format_,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
            face_detection_results=[]
        )
    
    def sample_frames(self, file_path: str) -> List[FrameData]:
        """Sample frames from video according to strategy"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        # Check duration limit
        if duration > self.max_duration_seconds:
            logger.warning(f"Video duration {duration:.1f}s exceeds limit {self.max_duration_seconds}s, truncating")
            frame_count = int(self.max_duration_seconds * fps)
        
        # Determine frame indices to sample
        if self.frame_sampling_method == "uniform":
            indices = np.linspace(0, frame_count - 1, min(self.num_frames, frame_count), dtype=int)
        elif self.frame_sampling_method == "random":
            indices = np.sort(np.random.choice(frame_count, min(self.num_frames, frame_count), replace=False))
        elif self.frame_sampling_method == "keyframe":
            # Sample at regular intervals
            interval = max(1, frame_count // self.num_frames)
            indices = np.arange(0, frame_count, interval)[:self.num_frames]
        else:
            indices = np.linspace(0, frame_count - 1, min(self.num_frames, frame_count), dtype=int)
        
        # Ensure minimum frames
        if len(indices) < self.min_frames:
            # Pad by duplicating last frame
            pad_count = self.min_frames - len(indices)
            indices = np.concatenate([indices, np.full(pad_count, indices[-1])])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = idx / fps if fps > 0 else 0
            
            # Face detection
            face_bbox = None
            face_confidence = 0.0
            
            if self.face_detection_enabled and self.face_detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size)
                )
                if len(faces) > 0:
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face
                    face_bbox = (int(x), int(y), int(w), int(h))
                    face_confidence = 0.9
            
            frames.append(FrameData(
                frame_index=int(idx),
                timestamp=timestamp,
                image=frame_rgb,
                face_bbox=face_bbox,
                face_confidence=face_confidence
            ))
        
        cap.release()
        return frames
    
    def preprocess_frames(self, frames: List[FrameData]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Preprocess sampled frames"""
        tensors = []
        frame_info = []
        
        for frame_data in frames:
            image = frame_data.image
            
            # Crop to face if detected
            if frame_data.face_bbox:
                x, y, w, h = frame_data.face_bbox
                pad = int(0.2 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                image = image[y1:y2, x1:x2]
            
            # Apply transform
            augmented = self.frame_transform(image=image)
            tensor = augmented["image"]
            
            tensors.append(tensor)
            frame_info.append({
                "frame_index": frame_data.frame_index,
                "timestamp": frame_data.timestamp,
                "face_bbox": frame_data.face_bbox,
                "face_confidence": frame_data.face_confidence
            })
        
        return torch.stack(tensors), frame_info
    
    def load_and_preprocess(self, file_path: str) -> Tuple[torch.Tensor, VideoMetadata, List[Dict[str, Any]]]:
        """Load video, extract metadata, sample and preprocess frames"""
        metadata = self.extract_metadata(file_path)
        frames = self.sample_frames(file_path)
        tensors, frame_info = self.preprocess_frames(frames)
        
        # Update metadata with face detection results
        metadata.face_detection_results = frame_info
        
        return tensors, metadata, frame_info
    
    def validate_video(self, file_path: str, max_size_mb: int = 100, max_duration: int = 60) -> Tuple[bool, str]:
        """Validate video file"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".mp4", ".mov", ".avi"]:
            return False, f"Unsupported format: {ext}"
        
        # Check if readable
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        if duration > max_duration:
            return False, f"Video duration {duration:.1f}s exceeds limit of {max_duration}s"
        
        if frame_count == 0:
            return False, "Video has no frames"
        
        return True, "Valid"


class VideoForensicAnalyzer:
    """Forensic analysis of videos for manipulation detection"""
    
    def __init__(self):
        pass
    
    def analyze_frame_consistency(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze consistency between consecutive frames"""
        if len(frames) < 2:
            return {"error": "Insufficient frames"}
        
        differences = []
        ssim_scores = []
        
        for i in range(1, len(frames)):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            differences.append(mean_diff)
            
            # Structural similarity (simplified)
            # Using correlation as proxy
            corr = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
            ssim_scores.append(float(corr) if not np.isnan(corr) else 1.0)
        
        return {
            "mean_frame_difference": float(np.mean(differences)),
            "std_frame_difference": float(np.std(differences)),
            "mean_ssim": float(np.mean(ssim_scores)),
            "min_ssim": float(np.min(ssim_scores)),
            "max_ssim": float(np.max(ssim_scores)),
            "frame_differences": [float(d) for d in differences],
            "ssim_scores": ssim_scores
        }
    
    def detect_frame_duplication(self, frames: List[np.ndarray], threshold: float = 0.99) -> Dict[str, Any]:
        """Detect duplicated or near-duplicate frames"""
        if len(frames) < 2:
            return {"duplicates": [], "duplicate_count": 0}
        
        duplicates = []
        hashes = []
        
        for i, frame in enumerate(frames):
            # Compute perceptual hash
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (32, 32))
            dct = cv2.dct(np.float32(small))
            hash_val = dct[:8, :8].flatten()
            hash_bits = (hash_val > np.median(hash_val)).astype(int)
            hashes.append(hash_bits)
        
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                similarity = np.mean(hashes[i] == hashes[j])
                if similarity >= threshold:
                    duplicates.append({
                        "frame_1": i,
                        "frame_2": j,
                        "similarity": float(similarity)
                    })
        
        return {
            "duplicates": duplicates,
            "duplicate_count": len(duplicates)
        }
    
    def analyze_gop_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze GOP (Group of Pictures) structure"""
        # This would require ffprobe for detailed analysis
        # Simplified version using OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        frame_types = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # OpenCV doesn't easily expose frame types (I/P/B frames)
        # This is a placeholder for full implementation
        cap.release()
        
        return {
            "note": "GOP analysis requires ffprobe for accurate I/P/B frame detection",
            "frame_count": frame_count
        }
    
    def analyze(self, file_path: str, frames: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Run all forensic analyses"""
        results = {}
        
        try:
            if frames is not None:
                results["frame_consistency"] = self.analyze_frame_consistency(frames)
                results["frame_duplication"] = self.detect_frame_duplication(frames)
            
            results["gop_structure"] = self.analyze_gop_structure(file_path)
        except Exception as e:
            logger.error(f"Video forensic analysis failed: {e}")
            results["error"] = str(e)
        
        return results


# ==================== LEGACY-COMPATIBLE API ====================
# Interface expected by ml/training/train_video.py, train_multimodal.py
# and backend/app/main.py:
#   preprocess(video_path, is_training, use_face_detection)
#       -> (Tensor[T, C, H, W], frame_indices, metadata_dict)
#   preprocessor.frame_sampler.num_frames
#   save(path) / load(path) via pickle
#   create_video_preprocessor_from_config(config_dict)

def _video_frame_sampler(self):
    from types import SimpleNamespace
    return SimpleNamespace(num_frames=self.num_frames)


def _video_preprocess(self, video_path: str, is_training: bool = False,
                      use_face_detection: bool = True):
    """Sample + preprocess frames (legacy signature)."""
    prev = self.face_detection_enabled
    self.face_detection_enabled = bool(use_face_detection) and prev
    try:
        frames = self.sample_frames(video_path)
        tensors, frame_info = self.preprocess_frames(frames)
        indices = [int(f["frame_index"]) for f in frame_info]
        faces = 0
        clean_info = []
        for f in frame_info:
            bbox = f.get("face_bbox")
            if bbox is not None:
                faces += 1
                bbox = [int(v) for v in bbox]
            clean_info.append({
                "frame_index": int(f["frame_index"]),
                "timestamp": float(f["timestamp"]),
                "face_bbox": bbox,
                "face_confidence": float(f.get("face_confidence", 0.0)),
            })
        metadata = {
            "num_frames_sampled": len(frames),
            "frame_indices": indices,
            "frames": clean_info,
            "faces_detected": faces,
        }
        return tensors, indices, metadata
    finally:
        self.face_detection_enabled = prev


def _video_get_params(self) -> Dict[str, Any]:
    return {
        "target_size": tuple(self.target_size),
        "normalization_mean": tuple(self.normalization_mean),
        "normalization_std": tuple(self.normalization_std),
        "frame_sampling_method": self.frame_sampling_method,
        "num_frames": self.num_frames,
        "min_frames": self.min_frames,
        "face_detection_enabled": self.face_detection_enabled,
        "min_face_size": self.min_face_size,
        "face_confidence_threshold": self.face_confidence_threshold,
        "max_duration_seconds": self.max_duration_seconds,
        "device": self.device,
    }


def _video_save(self, path: str):
    import pickle
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(self._get_params(), f)


@classmethod
def _video_load(cls, path: str) -> "VideoPreprocessor":
    import pickle
    with open(path, "rb") as f:
        params = pickle.load(f)
    return cls(**params)


VideoPreprocessor.frame_sampler = property(_video_frame_sampler)
VideoPreprocessor.preprocess = _video_preprocess
VideoPreprocessor._get_params = _video_get_params
VideoPreprocessor.save = _video_save
VideoPreprocessor.load = _video_load


def create_video_preprocessor_from_config(config: Dict[str, Any]) -> VideoPreprocessor:
    """Build VideoPreprocessor from the global config.yaml dict."""
    dataset_cfg = (config.get("dataset", {}) or {}) if isinstance(config, dict) else {}
    vid_cfg = dataset_cfg.get("video", {}) or {}
    img_cfg = dataset_cfg.get("image", {}) or {}
    norm = img_cfg.get("normalization", {}) or {}
    sampling = vid_cfg.get("frame_sampling", {}) or {}
    face_cfg = vid_cfg.get("face_detection", {}) or {}
    general = (config.get("general", {}) or {}) if isinstance(config, dict) else {}
    return VideoPreprocessor(
        target_size=tuple(img_cfg.get("target_size", [224, 224])),
        normalization_mean=tuple(norm.get("mean", [0.485, 0.456, 0.406])),
        normalization_std=tuple(norm.get("std", [0.229, 0.224, 0.225])),
        frame_sampling_method=sampling.get("method", "uniform"),
        num_frames=sampling.get("num_frames", 16),
        min_frames=sampling.get("min_frames", 8),
        face_detection_enabled=face_cfg.get("enabled", True),
        min_face_size=face_cfg.get("min_face_size", 64),
        face_confidence_threshold=face_cfg.get("confidence_threshold", 0.9),
        max_duration_seconds=vid_cfg.get("max_duration_seconds", 60),
        device=general.get("device", "cpu"),
    )