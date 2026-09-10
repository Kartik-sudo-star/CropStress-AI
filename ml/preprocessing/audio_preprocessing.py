"""
Audio preprocessing for deepfake detection
"""
import os
import hashlib
import librosa
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from loguru import logger


@dataclass
class AudioMetadata:
    """Metadata extracted from audio"""
    filename: str
    file_size: int
    sha256_hash: str
    duration: float
    sample_rate: int
    channels: int
    format: str
    subtype: str
    bit_depth: Optional[int]
    has_speech: bool = False


@dataclass
class AudioSegment:
    """Audio segment data"""
    start_time: float
    end_time: float
    waveform: np.ndarray
    sample_rate: int


class AudioPreprocessor:
    """Preprocessor for audio-based deepfake detection"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        segment_length: int = 4,
        spectrogram_n_fft: int = 512,
        spectrogram_hop_length: int = 160,
        spectrogram_n_mels: int = 80,
        spectrogram_target_length: int = 400,
        max_duration_seconds: int = 30,
        device: str = "cpu"
    ):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = spectrogram_n_fft
        self.hop_length = spectrogram_hop_length
        self.n_mels = spectrogram_n_mels
        self.target_length = spectrogram_target_length
        self.max_duration_seconds = max_duration_seconds
        self.device = device
    
    def compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def extract_metadata(self, file_path: str) -> AudioMetadata:
        """Extract metadata from audio file"""
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        sha256_hash = self.compute_hash(file_path)
        
        info = sf.info(file_path)
        
        return AudioMetadata(
            filename=filename,
            file_size=file_size,
            sha256_hash=sha256_hash,
            duration=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            format=info.format,
            subtype=info.subtype,
            bit_depth=info.bits_per_sample if info.bits_per_sample > 0 else None
        )
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and resample audio"""
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Trim silence
        waveform, _ = librosa.effects.trim(waveform, top_db=30)
        
        # Limit duration
        max_samples = self.max_duration_seconds * self.sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform, self.sample_rate
    
    def segment_audio(self, waveform: np.ndarray) -> List[AudioSegment]:
        """Split audio into segments"""
        segment_samples = self.segment_length * self.sample_rate
        segments = []
        
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i:i + segment_samples]
            if len(segment) < segment_samples:
                # Pad last segment
                segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
            
            segments.append(AudioSegment(
                start_time=i / self.sample_rate,
                end_time=min((i + segment_samples) / self.sample_rate, len(waveform) / self.sample_rate),
                waveform=segment,
                sample_rate=self.sample_rate
            ))
        
        return segments
    
    def waveform_to_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Convert waveform to mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Fix time dimension
        if log_mel.shape[1] < self.target_length:
            # Pad
            pad_width = self.target_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > self.target_length:
            # Crop
            log_mel = log_mel[:, :self.target_length]
        
        return log_mel.astype(np.float32)
    
    def preprocess(self, file_path: str) -> Tuple[torch.Tensor, AudioMetadata, List[AudioSegment]]:
        """Load, segment, and convert to spectrograms"""
        metadata = self.extract_metadata(file_path)
        waveform, sr = self.load_audio(file_path)
        segments = self.segment_audio(waveform)
        
        spectrograms = []
        for segment in segments:
            spec = self.waveform_to_spectrogram(segment.waveform)
            spectrograms.append(spec)
        
        # Stack into tensor: (num_segments, n_mels, target_length)
        tensor = torch.from_numpy(np.stack(spectrograms)).unsqueeze(1)  # Add channel dim
        
        return tensor, metadata, segments
    
    def validate_audio(self, file_path: str, max_size_mb: int = 20, max_duration: int = 30) -> Tuple[bool, str]:
        """Validate audio file"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".wav", ".mp3", ".m4a"]:
            return False, f"Unsupported format: {ext}"
        
        try:
            info = sf.info(file_path)
            if info.duration > max_duration:
                return False, f"Audio duration {info.duration:.1f}s exceeds limit of {max_duration}s"
            if info.frames == 0:
                return False, "Audio file is empty"
        except Exception as e:
            return False, f"Cannot read audio file: {e}"
        
        return True, "Valid"


class AudioForensicAnalyzer:
    """Forensic analysis of audio for manipulation detection"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def spectral_consistency(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral consistency across time"""
        # Compute STFT
        stft = librosa.stft(waveform, n_fft=512, hop_length=160)
        magnitude = np.abs(stft)
        
        # Spectral centroid over time
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        
        # RMSE energy
        rmse = librosa.feature.rms(S=magnitude)[0]
        
        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_flatness_mean": float(np.mean(spectral_flatness)),
            "zcr_mean": float(np.mean(zcr)),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_std": float(np.std(rmse))
        }
    
    def detect_splicing(self, waveform: np.ndarray, window_size: int = 16000) -> Dict[str, Any]:
        """Detect potential audio splicing/editing"""
        # Sliding window analysis of spectral features
        hop_size = window_size // 2
        num_windows = (len(waveform) - window_size) // hop_size + 1
        
        if num_windows < 2:
            return {"splices": [], "splice_count": 0}
        
        features = []
        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            window = waveform[start:end]
            
            # MFCCs for this window
            mfcc = librosa.feature.mfcc(y=window, sr=self.sample_rate, n_mfcc=13)
            features.append(np.mean(mfcc, axis=1))
        
        features = np.array(features)  # (num_windows, 13)
        
        # Detect abrupt changes
        diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)
        threshold = np.mean(diffs) + 3 * np.std(diffs)
        
        splices = []
        for i, diff in enumerate(diffs):
            if diff > threshold:
                time = (i * hop_size) / self.sample_rate
                splices.append({
                    "time": float(time),
                    "magnitude": float(diff),
                    "threshold": float(threshold)
                })
        
        return {
            "splices": splices,
            "splice_count": len(splices),
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs))
        }
    
    def check_sample_rate_consistency(self, file_path: str) -> Dict[str, Any]:
        """Check for sample rate inconsistencies (resampling artifacts)"""
        try:
            # Load at native sample rate
            waveform_native, sr_native = librosa.load(file_path, sr=None, mono=True)
            # Load at target sample rate
            waveform_target, sr_target = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Check if resampling occurred
            resampled = sr_native != self.sample_rate
            
            # Spectral analysis for resampling artifacts
            if resampled:
                # Look for imaging artifacts
                stft = np.abs(librosa.stft(waveform_native))
                # High frequency imaging would show as mirrored content
                # This is a simplified check
                return {
                    "native_sample_rate": sr_native,
                    "target_sample_rate": sr_target,
                    "resampled": True,
                    "note": "File was resampled, potential quality loss"
                }
            else:
                return {
                    "native_sample_rate": sr_native,
                    "target_sample_rate": sr_target,
                    "resampled": False
                }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Run all forensic analyses"""
        results = {}
        
        try:
            waveform, sr = self.load_audio(file_path)
            results["spectral_consistency"] = self.spectral_consistency(waveform)
            results["splicing_detection"] = self.detect_splicing(waveform)
            results["sample_rate_check"] = self.check_sample_rate_consistency(file_path)
        except Exception as e:
            logger.error(f"Audio forensic analysis failed: {e}")
            results["error"] = str(e)
        
        return results


# ==================== LEGACY-COMPATIBLE API ====================
# Interface expected by ml/training/train_audio.py, train_multimodal.py
# and backend/app/main.py:
#   preprocess(audio_path, is_training, return_waveform=True)
#       -> (waveform[1, S], mel[1, n_mels, T]) or mel[1, n_mels, T]
#   preprocessor.segment_samples / .spectrogram_config / .target_length
#   save(path) / load(path) via pickle
#   create_audio_preprocessor_from_config(config_dict)

def _audio_load_waveform(self, file_path: str):
    """Load + resample a waveform (also attached to the forensic analyzer)."""
    waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
    return waveform.astype(np.float32)


AudioForensicAnalyzer.load_audio = _audio_load_waveform


def _audio_segment_samples(self) -> int:
    return int(self.segment_length * self.sample_rate)


def _audio_spectrogram_config(self) -> Dict[str, Any]:
    return {
        "n_fft": self.n_fft,
        "hop_length": self.hop_length,
        "n_mels": self.n_mels,
    }


def _audio_preprocess(self, audio_path: str, is_training: bool = False,
                      return_waveform: bool = True):
    """Load + convert to fixed-size waveform/mel tensors (legacy signature)."""
    waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    S = self.segment_samples
    if len(waveform) < S:
        waveform = np.pad(waveform, (0, S - len(waveform)), mode="constant")
    else:
        waveform = waveform[:S]
    waveform = waveform.astype(np.float32)
    mel = self.waveform_to_spectrogram(waveform)  # [n_mels, T]
    wave_t = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, S]
    mel_t = torch.from_numpy(mel).float().unsqueeze(0)  # [1, n_mels, T]
    if return_waveform:
        return wave_t, mel_t
    return mel_t


def _audio_get_params(self) -> Dict[str, Any]:
    return {
        "sample_rate": self.sample_rate,
        "segment_length": self.segment_length,
        "spectrogram_n_fft": self.n_fft,
        "spectrogram_hop_length": self.hop_length,
        "spectrogram_n_mels": self.n_mels,
        "spectrogram_target_length": self.target_length,
        "max_duration_seconds": self.max_duration_seconds,
        "device": self.device,
    }


def _audio_save(self, path: str):
    import pickle
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(self._get_params(), f)


@classmethod
def _audio_load(cls, path: str) -> "AudioPreprocessor":
    import pickle
    with open(path, "rb") as f:
        params = pickle.load(f)
    return cls(**params)


AudioPreprocessor.segment_samples = property(_audio_segment_samples)
AudioPreprocessor.spectrogram_config = property(_audio_spectrogram_config)
AudioPreprocessor.preprocess = _audio_preprocess
AudioPreprocessor._get_params = _audio_get_params
AudioPreprocessor.save = _audio_save
AudioPreprocessor.load = _audio_load


def create_audio_preprocessor_from_config(config: Dict[str, Any]) -> AudioPreprocessor:
    """Build AudioPreprocessor from the global config.yaml dict."""
    dataset_cfg = (config.get("dataset", {}) or {}) if isinstance(config, dict) else {}
    aud_cfg = dataset_cfg.get("audio", {}) or {}
    spec = aud_cfg.get("spectrogram", {}) or {}
    general = (config.get("general", {}) or {}) if isinstance(config, dict) else {}
    return AudioPreprocessor(
        sample_rate=aud_cfg.get("sample_rate", 16000),
        segment_length=aud_cfg.get("segment_length", 4),
        spectrogram_n_fft=spec.get("n_fft", 512),
        spectrogram_hop_length=spec.get("hop_length", 160),
        spectrogram_n_mels=spec.get("n_mels", 80),
        spectrogram_target_length=spec.get("target_length", 400),
        max_duration_seconds=aud_cfg.get("max_duration_seconds", 30),
        device=general.get("device", "cpu"),
    )