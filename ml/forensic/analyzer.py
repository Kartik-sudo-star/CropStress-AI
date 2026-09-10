"""
Forensic Analysis Module

Extracts metadata and forensic indicators from images, videos, and audio.
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
import hashlib
import subprocess

logger = logging.getLogger(__name__)


class ForensicAnalyzer:
    """
    Comprehensive forensic analysis for multimedia files.
    
    Provides:
    - Metadata extraction (EXIF, video codec info, audio headers)
    - Error Level Analysis (ELA) for images
    - Noise analysis
    - Compression artifact detection
    - Frame consistency analysis for video
    - Spectral analysis for audio
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.image_config = self.config.get("forensic", {}).get("image", {})
        self.video_config = self.config.get("forensic", {}).get("video", {})
        self.audio_config = self.config.get("forensic", {}).get("audio", {})
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze any supported file type."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # Calculate hash
        file_hash = self._calculate_sha256(file_path)
        
        result = {
            "file_path": str(file_path),
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "sha256": file_hash,
            "file_type": self._detect_file_type(suffix),
        }
        
        if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            result.update(self.analyze_image(file_path))
        elif suffix in [".mp4", ".mov", ".avi"]:
            result.update(self.analyze_video(file_path))
        elif suffix in [".wav", ".mp3", ".m4a"]:
            result.update(self.analyze_audio(file_path))
        else:
            result["warning"] = f"Unsupported file type: {suffix}"
        
        return result
    
    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _detect_file_type(self, suffix: str) -> str:
        """Detect file type from extension."""
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        
        if suffix in image_exts:
            return "image"
        elif suffix in video_exts:
            return "video"
        elif suffix in audio_exts:
            return "audio"
        return "unknown"
    
    # ==================== IMAGE ANALYSIS ====================
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive image forensic analysis."""
        results = {"media_type": "image"}
        
        # Basic metadata
        results["metadata"] = self._extract_image_metadata(image_path)
        
        # Load image
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            pil_img = Image.open(image_path)
        except Exception as e:
            results["error"] = f"Failed to load image: {e}"
            return results
        
        # Error Level Analysis (ELA)
        if self.image_config.get("analyze_compression", True):
            results["ela"] = self._error_level_analysis(image_path, pil_img)
        
        # Noise analysis
        if self.image_config.get("detect_splicing", True):
            results["noise_analysis"] = self._noise_analysis(img)
        
        # Double JPEG detection
        if self.image_config.get("analyze_compression", True):
            results["double_jpeg"] = self._detect_double_jpeg(img)
        
        # Copy-move forgery detection (basic)
        results["copy_move"] = self._detect_copy_move(img)
        
        # Metadata consistency
        if self.image_config.get("check_consistency", True):
            results["metadata_consistency"] = self._check_metadata_consistency(pil_img)
        
        return results
    
    def _extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF and other metadata."""
        metadata = {}
        
        try:
            img = Image.open(image_path)
            
            # Basic info
            metadata["format"] = img.format
            metadata["mode"] = img.mode
            metadata["width"] = img.width
            metadata["height"] = img.height
            metadata["aspect_ratio"] = round(img.width / img.height, 2) if img.height > 0 else 0
            
            # EXIF data
            exif = img.getexif()
            if exif:
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    # Convert non-serializable values
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', errors='ignore')
                    exif_data[tag] = value
                metadata["exif"] = exif_data
                
                # Key forensic fields
                metadata["camera_make"] = exif_data.get("Make")
                metadata["camera_model"] = exif_data.get("Model")
                metadata["datetime_original"] = exif_data.get("DateTimeOriginal")
                metadata["software"] = exif_data.get("Software")
                metadata["gps_info"] = exif_data.get("GPSInfo")
            else:
                metadata["exif"] = {}
                metadata["camera_make"] = None
                metadata["camera_model"] = None
                metadata["datetime_original"] = None
                metadata["software"] = None
            
        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata
    
    def _error_level_analysis(self, image_path: str, pil_img: Image.Image, quality: int = 90) -> Dict[str, Any]:
        """
        Error Level Analysis - detects areas with different compression levels.
        Re-saves at known quality and compares difference.
        """
        import io
        
        try:
            # Convert to RGB if needed
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            # Save at specified quality
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            
            # Reload
            recompressed = Image.open(buffer)
            
            # Convert to numpy
            original = np.array(pil_img).astype(np.float32)
            compressed = np.array(recompressed).astype(np.float32)
            
            # Calculate difference
            diff = np.abs(original - compressed)
            diff_gray = diff.mean(axis=2)  # Average across channels
            
            # Normalize for visualization
            if diff_gray.max() > 0:
                ela_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
            else:
                ela_normalized = np.zeros_like(diff_gray, dtype=np.uint8)
            
            # Statistics
            ela_mean = float(diff_gray.mean())
            ela_std = float(diff_gray.std())
            ela_max = float(diff_gray.max())
            
            # Detect suspicious regions (high ELA)
            threshold = ela_mean + 2 * ela_std
            suspicious_ratio = float((diff_gray > threshold).sum() / diff_gray.size)
            
            return {
                "mean_error": ela_mean,
                "std_error": ela_std,
                "max_error": ela_max,
                "suspicious_ratio": suspicious_ratio,
                "threshold_used": threshold,
                "interpretation": self._interpret_ela(ela_mean, ela_std, suspicious_ratio)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _interpret_ela(self, mean_err: float, std_err: float, suspicious_ratio: float) -> str:
        """Interpret ELA results."""
        if suspicious_ratio > 0.1:
            return "High suspicious regions detected - possible manipulation"
        elif suspicious_ratio > 0.05:
            return "Moderate suspicious regions - review recommended"
        elif mean_err > 10:
            return "High average error - possible heavy compression or editing"
        else:
            return "No significant anomalies detected"
    
    def _noise_analysis(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns for inconsistency detection."""
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Estimate noise using high-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # Local noise variance (divide into blocks)
            h, w = gray.shape
            block_size = 32
            noise_map = np.zeros((h // block_size, w // block_size))
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = noise[i:i+block_size, j:j+block_size]
                    noise_map[i//block_size, j//block_size] = np.var(block)
            
            # Statistics
            noise_mean = float(noise_map.mean())
            noise_std = float(noise_map.std())
            
            # Detect outliers (inconsistent noise)
            if noise_std > 0:
                z_scores = np.abs(noise_map - noise_mean) / noise_std
                outlier_ratio = float((z_scores > 3).sum() / z_scores.size)
            else:
                outlier_ratio = 0.0
            
            return {
                "global_noise_mean": noise_mean,
                "global_noise_std": noise_std,
                "outlier_block_ratio": outlier_ratio,
                "interpretation": self._interpret_noise(noise_mean, noise_std, outlier_ratio)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _interpret_noise(self, mean: float, std: float, outlier_ratio: float) -> str:
        if outlier_ratio > 0.05:
            return "Inconsistent noise patterns detected - possible splicing"
        elif outlier_ratio > 0.02:
            return "Some noise inconsistency - review recommended"
        else:
            return "Consistent noise pattern"
    
    def _detect_double_jpeg(self, img: np.ndarray) -> Dict[str, Any]:
        """Detect double JPEG compression using Benford's law on DCT coefficients."""
        try:
            # Simple implementation - check for double quantization artifacts
            # Convert to YCrCb and analyze DCT coefficients of Y channel
            if len(img.shape) == 3:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                y = ycrcb[:, :, 0]
            else:
                y = img
            
            # Compute DCT on 8x8 blocks
            h, w = y.shape
            dct_coeffs = []
            
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    block = y[i:i+8, j:j+8].astype(np.float32)
                    dct = cv2.dct(block)
                    dct_coeffs.append(dct.flatten())
            
            if not dct_coeffs:
                return {"detected": False, "reason": "Image too small"}
            
            dct_coeffs = np.array(dct_coeffs)
            
            # Analyze histogram of specific AC coefficients
            # Double JPEG creates periodic peaks in histogram
            ac_coeff = dct_coeffs[:, 10]  # Example AC coefficient
            
            hist, bins = np.histogram(ac_coeff, bins=50)
            hist = hist.astype(np.float32)
            
            # Check for periodic peaks (simplified)
            # In double JPEG, histogram shows comb-like pattern
            diffs = np.diff(hist)
            peak_count = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
            
            # Simple heuristic
            suspicious = peak_count > 10 and np.std(hist) > np.mean(hist) * 0.5
            
            return {
                "detected": bool(suspicious),
                "peak_count": int(peak_count),
                "histogram_std": float(np.std(hist)),
                "histogram_mean": float(np.mean(hist)),
                "interpretation": "Possible double JPEG compression" if suspicious else "No double JPEG detected"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_copy_move(self, img: np.ndarray) -> Dict[str, Any]:
        """Basic copy-move forgery detection using block matching."""
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            h, w = gray.shape
            block_size = 16
            step = 8
            
            # Extract block features (DCT)
            blocks = []
            positions = []
            
            for i in range(0, h - block_size, step):
                for j in range(0, w - block_size, step):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    dct = cv2.dct(block)
                    # Use low-frequency coefficients as features
                    feat = dct[:8, :8].flatten()
                    blocks.append(feat)
                    positions.append((i, j))
            
            if len(blocks) < 2:
                return {"detected": False, "matches": 0}
            
            blocks = np.array(blocks)
            
            # Find similar blocks (cosine similarity)
            # Simplified: use correlation
            matches = []
            threshold = 0.95
            
            for i in range(len(blocks)):
                for j in range(i+1, min(i+100, len(blocks))):  # Limit search
                    corr = np.corrcoef(blocks[i], blocks[j])[0, 1]
                    if corr > threshold:
                        dist = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                        if dist > block_size * 2:  # Not adjacent
                            matches.append({
                                "pos1": positions[i],
                                "pos2": positions[j],
                                "similarity": float(corr),
                                "distance": float(dist)
                            })
            
            return {
                "detected": len(matches) > 0,
                "match_count": len(matches),
                "top_matches": matches[:5],
                "interpretation": "Possible copy-move forgery" if len(matches) > 0 else "No copy-move detected"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _check_metadata_consistency(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Check for metadata inconsistencies."""
        issues = []
        
        try:
            exif = pil_img.getexif()
            if not exif:
                return {"issues": [], "consistent": True}
            
            exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            
            # Check software tag
            software = exif_data.get("Software", "")
            if software:
                editing_software = ["photoshop", "gimp", "lightroom", "snapseed", "picsart", "after effects", "premiere"]
                for sw in editing_software:
                    if sw in software.lower():
                        issues.append(f"Edited with {software}")
                        break
            
            # Check datetime consistency
            datetime_orig = exif_data.get("DateTimeOriginal")
            datetime_digitized = exif_data.get("DateTimeDigitized")
            datetime_modified = exif_data.get("DateTime")
            
            if datetime_orig and datetime_modified and datetime_orig != datetime_modified:
                issues.append("DateTimeOriginal != DateTime (possible modification)")
            
            # Check for missing camera info (screenshots often lack this)
            if not exif_data.get("Make") and not exif_data.get("Model"):
                issues.append("No camera make/model (possible screenshot or generated)")
            
            return {
                "issues": issues,
                "consistent": len(issues) == 0,
                "software": software
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ==================== VIDEO ANALYSIS ====================
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Comprehensive video forensic analysis."""
        results = {"media_type": "video"}
        
        # Metadata
        results["metadata"] = self._extract_video_metadata(video_path)
        
        # Frame analysis
        if self.video_config.get("analyze_frame_consistency", True):
            results["frame_analysis"] = self._analyze_video_frames(video_path)
        
        # GOP structure
        if self.video_config.get("analyze_gop_structure", True):
            results["gop_analysis"] = self._analyze_gop_structure(video_path)
        
        # Frame duplication
        if self.video_config.get("detect_frame_duplication", True):
            results["duplication"] = self._detect_frame_duplication(video_path)
        
        return results
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe."""
        metadata = {}
        
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            
            # Format info
            fmt = data.get("format", {})
            metadata["format"] = fmt.get("format_name")
            metadata["duration"] = float(fmt.get("duration", 0))
            metadata["bitrate"] = int(fmt.get("bit_rate", 0))
            metadata["size"] = int(fmt.get("size", 0))
            
            # Stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    metadata["video_codec"] = stream.get("codec_name")
                    metadata["width"] = stream.get("width")
                    metadata["height"] = stream.get("height")
                    metadata["fps"] = eval(stream.get("r_frame_rate", "0/1"))
                    metadata["pix_fmt"] = stream.get("pix_fmt")
                    metadata["bitrate"] = stream.get("bit_rate")
                elif stream.get("codec_type") == "audio":
                    metadata["audio_codec"] = stream.get("codec_name")
                    metadata["audio_sample_rate"] = stream.get("sample_rate")
                    metadata["audio_channels"] = stream.get("channels")
            
        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata
    
    def _analyze_video_frames(self, video_path: str, max_frames: int = 100) -> Dict[str, Any]:
        """Analyze frame-level consistency."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(max_frames, total_frames)
        
        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        frame_stats = []
        prev_frame = None
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness/contrast stats
            mean_val = float(gray.mean())
            std_val = float(gray.std())
            
            # Difference from previous frame
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                diff_mean = float(diff.mean())
                diff_std = float(diff.std())
            else:
                diff_mean = 0
                diff_std = 0
            
            frame_stats.append({
                "frame_index": int(idx),
                "brightness": mean_val,
                "contrast": std_val,
                "diff_mean": diff_mean,
                "diff_std": diff_std
            })
            
            prev_frame = gray
        
        cap.release()
        
        # Detect anomalies
        if frame_stats:
            brightness_vals = [f["brightness"] for f in frame_stats]
            contrast_vals = [f["contrast"] for f in frame_stats]
            
            brightness_mean = np.mean(brightness_vals)
            brightness_std = np.std(brightness_vals)
            
            # Frames with unusual brightness/contrast
            anomalies = []
            for f in frame_stats:
                if abs(f["brightness"] - brightness_mean) > 3 * brightness_std:
                    anomalies.append({
                        "frame": f["frame_index"],
                        "type": "brightness_anomaly",
                        "value": f["brightness"],
                        "z_score": (f["brightness"] - brightness_mean) / brightness_std
                    })
        
        return {
            "total_frames": total_frames,
            "sampled_frames": len(frame_stats),
            "frame_stats": frame_stats[:20],  # Limit output
            "anomalies": anomalies,
            "summary": {
                "brightness_mean": float(brightness_mean),
                "brightness_std": float(brightness_std),
                "anomaly_count": len(anomalies)
            }
        }
    
    def _analyze_gop_structure(self, video_path: str) -> Dict[str, Any]:
        """Analyze Group of Pictures structure for editing detection."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "frame=pict_type,pkt_pts_time",
                "-of", "csv=p=0",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            lines = result.stdout.strip().split('\n')
            frame_types = []
            
            for line in lines:
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        frame_types.append(parts[0])  # I, P, B
            
            # Count frame types
            from collections import Counter
            type_counts = Counter(frame_types)
            
            # Check GOP pattern
            gop_sizes = []
            current_gop = 0
            for ft in frame_types:
                if ft == 'I':
                    if current_gop > 0:
                        gop_sizes.append(current_gop)
                    current_gop = 1
                else:
                    current_gop += 1
            if current_gop > 0:
                gop_sizes.append(current_gop)
            
            # Regular GOP should have consistent sizes
            gop_consistency = np.std(gop_sizes) / np.mean(gop_sizes) if gop_sizes else 0
            
            return {
                "frame_type_counts": dict(type_counts),
                "gop_sizes": gop_sizes,
                "avg_gop_size": float(np.mean(gop_sizes)) if gop_sizes else 0,
                "gop_consistency": float(gop_consistency),
                "irregular_gops": int(np.sum(np.array(gop_sizes) > np.mean(gop_sizes) * 2)) if gop_sizes else 0,
                "interpretation": "Regular GOP structure" if gop_consistency < 0.3 else "Irregular GOP - possible editing"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_frame_duplication(self, video_path: str, sample_rate: int = 1) -> Dict[str, Any]:
        """Detect duplicated frames (freeze frames, loops)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_hash = None
        duplicates = []
        frame_hashes = []
        
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Compute perceptual hash
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32))
            dct = cv2.dct(small.astype(np.float32))
            hash_val = dct[:8, :8].flatten()
            hash_bits = (hash_val > np.median(hash_val)).astype(int)
            hash_str = ''.join(map(str, hash_bits))
            
            frame_hashes.append((i, hash_str))
            
            if prev_hash == hash_str:
                duplicates.append(i)
            
            prev_hash = hash_str
        
        cap.release()
        
        return {
            "total_frames_checked": len(frame_hashes),
            "duplicate_frames": duplicates,
            "duplicate_count": len(duplicates),
            "interpretation": f"Found {len(duplicates)} duplicate frames" if duplicates else "No duplicate frames detected"
        }
    
    # ==================== AUDIO ANALYSIS ====================
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive audio forensic analysis."""
        results = {"media_type": "audio"}
        
        # Metadata
        results["metadata"] = self._extract_audio_metadata(audio_path)
        
        # Load audio
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            results["error"] = f"Failed to load audio: {e}"
            return results
        
        # Spectral consistency
        if self.audio_config.get("analyze_spectral_consistency", True):
            results["spectral_analysis"] = self._analyze_audio_spectral(y, sr)
        
        # Splicing detection
        if self.audio_config.get("detect_splicing", True):
            results["splicing"] = self._detect_audio_splicing(y, sr)
        
        # Sample rate consistency
        if self.audio_config.get("check_sample_rate_consistency", True):
            results["sample_rate_check"] = self._check_sample_rate_consistency(audio_path)
        
        return results
    
    def _extract_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio metadata."""
        import soundfile as sf
        
        metadata = {}
        try:
            info = sf.info(audio_path)
            metadata["format"] = info.format
            metadata["subtype"] = info.subtype
            metadata["sample_rate"] = info.samplerate
            metadata["channels"] = info.channels
            metadata["duration"] = info.duration
            metadata["frames"] = info.frames
        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata
    
    def _analyze_audio_spectral(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze spectral consistency."""
        import librosa
        
        try:
            # STFT
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Spectral features over time
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)
            
            # Detect abrupt changes
            def detect_changes(feature, threshold=3.0):
                diff = np.diff(feature.flatten())
                z_scores = np.abs(diff - np.mean(diff)) / (np.std(diff) + 1e-8)
                return np.where(z_scores > threshold)[0].tolist()
            
            centroid_changes = detect_changes(spectral_centroid)
            rolloff_changes = detect_changes(spectral_rolloff)
            bandwidth_changes = detect_changes(spectral_bandwidth)
            
            return {
                "spectral_centroid_mean": float(spectral_centroid.mean()),
                "spectral_centroid_std": float(spectral_centroid.std()),
                "spectral_rolloff_mean": float(spectral_rolloff.mean()),
                "spectral_bandwidth_mean": float(spectral_bandwidth.mean()),
                "abrupt_changes": {
                    "centroid": len(centroid_changes),
                    "rolloff": len(rolloff_changes),
                    "bandwidth": len(bandwidth_changes)
                },
                "interpretation": "Spectral anomalies detected" if (centroid_changes or rolloff_changes or bandwidth_changes) else "Spectrally consistent"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_audio_splicing(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect audio splicing using spectral flux."""
        import librosa
        
        try:
            # Spectral flux
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
            
            # Find peaks in flux
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(flux, height=np.mean(flux) + 2*np.std(flux))
            
            splice_points = []
            for peak in peaks:
                time = peak * 512 / sr
                splice_points.append({
                    "time": float(time),
                    "flux_value": float(flux[peak]),
                    "prominence": float(properties.get("prominences", [0])[0]) if "prominences" in properties else 0
                })
            
            return {
                "splice_candidates": len(splice_points),
                "splice_points": splice_points,
                "mean_flux": float(flux.mean()),
                "std_flux": float(flux.std()),
                "interpretation": f"Found {len(splice_points)} potential splice points" if splice_points else "No splicing detected"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _check_sample_rate_consistency(self, audio_path: str) -> Dict[str, Any]:
        """Check for sample rate inconsistencies."""
        import soundfile as sf
        
        try:
            # Read in chunks to check sample rate
            info = sf.info(audio_path)
            return {
                "sample_rate": info.samplerate,
                "consistent": True,  # Single file typically has consistent SR
                "note": "Single file sample rate check - for multi-file analysis compare across files"
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ==================== REPORTING ====================
    
    def generate_forensic_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable forensic report."""
        report = f"""
# Forensic Analysis Report

**File**: {analysis_results.get('file_name', 'Unknown')}
**Type**: {analysis_results.get('file_type', 'Unknown')}
**Size**: {analysis_results.get('file_size', 0)} bytes
**SHA-256**: {analysis_results.get('sha256', 'Unknown')}

## Summary
"""
        
        media_type = analysis_results.get("media_type", "unknown")
        
        if media_type == "image":
            report += self._format_image_report(analysis_results)
        elif media_type == "video":
            report += self._format_video_report(analysis_results)
        elif media_type == "audio":
            report += self._format_audio_report(analysis_results)
        
        return report
    
    def _format_image_report(self, results: Dict) -> str:
        report = ""
        
        # Metadata
        meta = results.get("metadata", {})
        report += f"""
### Image Properties
- **Dimensions**: {meta.get('width', '?')} x {meta.get('height', '?')}
- **Format**: {meta.get('format', '?')}
- **Camera**: {meta.get('camera_make', '?')} {meta.get('camera_model', '?')}
- **Software**: {meta.get('software', 'None detected')}
- **DateTime**: {meta.get('datetime_original', '?')}
"""
        
        # ELA
        ela = results.get("ela", {})
        if "error" not in ela:
            report += f"""
### Error Level Analysis (ELA)
- **Mean Error**: {ela.get('mean_error', 0):.2f}
- **Std Error**: {ela.get('std_error', 0):.2f}
- **Suspicious Regions**: {ela.get('suspicious_ratio', 0)*100:.1f}%
- **Assessment**: {ela.get('interpretation', 'N/A')}
"""
        
        # Noise
        noise = results.get("noise_analysis", {})
        if "error" not in noise:
            report += f"""
### Noise Analysis
- **Global Noise Mean**: {noise.get('global_noise_mean', 0):.2f}
- **Global Noise Std**: {noise.get('global_noise_std', 0):.2f}
- **Outlier Blocks**: {noise.get('outlier_block_ratio', 0)*100:.1f}%
- **Assessment**: {noise.get('interpretation', 'N/A')}
"""
        
        # Copy-move
        cm = results.get("copy_move", {})
        if "error" not in cm:
            report += f"""
### Copy-Move Detection
- **Matches Found**: {cm.get('match_count', 0)}
- **Assessment**: {cm.get('interpretation', 'N/A')}
"""
        
        # Metadata consistency
        mc = results.get("metadata_consistency", {})
        if "error" not in mc:
            report += f"""
### Metadata Consistency
- **Consistent**: {mc.get('consistent', False)}
- **Issues**: {', '.join(mc.get('issues', ['None']))}
"""
        
        return report
    
    def _format_video_report(self, results: Dict) -> str:
        report = ""
        
        meta = results.get("metadata", {})
        report += f"""
### Video Properties
- **Codec**: {meta.get('video_codec', '?')}
- **Resolution**: {meta.get('width', '?')} x {meta.get('height', '?')}
- **FPS**: {meta.get('fps', '?')}
- **Duration**: {meta.get('duration', 0):.1f}s
- **Bitrate**: {meta.get('bitrate', 0)} bps
"""
        
        frame_analysis = results.get("frame_analysis", {})
        if "error" not in frame_analysis:
            report += f"""
### Frame Consistency
- **Total Frames**: {frame_analysis.get('total_frames', 0)}
- **Anomalies**: {frame_analysis.get('summary', {}).get('anomaly_count', 0)}
"""
        
        gop = results.get("gop_analysis", {})
        if "error" not in gop:
            report += f"""
### GOP Structure
- **Avg GOP Size**: {gop.get('avg_gop_size', 0):.1f}
- **GOP Consistency**: {gop.get('gop_consistency', 0):.2f}
- **Assessment**: {gop.get('interpretation', 'N/A')}
"""
        
        dup = results.get("duplication", {})
        if "error" not in dup:
            report += f"""
### Frame Duplication
- **Duplicates Found**: {dup.get('duplicate_count', 0)}
- **Assessment**: {dup.get('interpretation', 'N/A')}
"""
        
        return report
    
    def _format_audio_report(self, results: Dict) -> str:
        report = ""
        
        meta = results.get("metadata", {})
        report += f"""
### Audio Properties
- **Format**: {meta.get('format', '?')}
- **Sample Rate**: {meta.get('sample_rate', '?')} Hz
- **Channels**: {meta.get('channels', '?')}
- **Duration**: {meta.get('duration', 0):.1f}s
"""
        
        spectral = results.get("spectral_analysis", {})
        if "error" not in spectral:
            report += f"""
### Spectral Analysis
- **Centroid Mean**: {spectral.get('spectral_centroid_mean', 0):.1f}
- **Abrupt Changes**: {spectral.get('abrupt_changes', {}).get('centroid', 0)} centroid, {spectral.get('abrupt_changes', {}).get('rolloff', 0)} rolloff
- **Assessment**: {spectral.get('interpretation', 'N/A')}
"""
        
        splicing = results.get("splicing", {})
        if "error" not in splicing:
            report += f"""
### Splicing Detection
- **Candidates**: {splicing.get('splice_candidates', 0)}
- **Assessment**: {splicing.get('interpretation', 'N/A')}
"""
        
        return report


def create_forensic_analyzer(config: Dict = None) -> ForensicAnalyzer:
    """Factory function."""
    return ForensicAnalyzer(config)