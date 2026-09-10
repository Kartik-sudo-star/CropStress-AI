"""
FastAPI Backend for Deepfake Detection
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import hashlib
import json
import tempfile
import shutil
from dataclasses import asdict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models import (
    create_image_detector, load_image_detector,
    create_video_detector, load_video_detector,
    create_audio_detector, load_audio_detector,
    create_multimodal_detector, load_multimodal_detector
)
from ml.preprocessing import (
    create_preprocessor_from_config,
    create_video_preprocessor_from_config,
    create_audio_preprocessor_from_config,
    InferenceImagePreprocessor,
    VideoPreprocessor,
    AudioPreprocessor
)
from ml.explainability import GradCAM, overlay_heatmap, tensor_to_numpy_image
from ml.forensic import create_forensic_analyzer
from ml.risk import create_risk_assessor
from ml.evidence import create_evidence_manager, create_report_generator, AnalysisRecord

# Global state
models = {}
preprocessors = {}
config = {}
device = torch.device("cpu")

# ==================== PYDANTIC MODELS ====================

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    file_type: str
    file_hash: str
    prediction: str
    confidence: float
    risk_level: str
    risk_score: float
    deepfake_probability: float
    media_properties: Dict[str, Any]
    forensic_indicators: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    explanation: Dict[str, Any]
    recommendations: List[str]
    india_guidance: Dict[str, Any]
    model_version: str

class HistoryItem(BaseModel):
    analysis_id: str
    timestamp: str
    file_type: str
    prediction: str
    confidence: float
    risk_level: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    device: str

class ModelsResponse(BaseModel):
    image_detector: Dict[str, Any]
    video_detector: Dict[str, Any]
    audio_detector: Dict[str, Any]
    multimodal_detector: Dict[str, Any]

# ==================== STARTUP/SHUTDOWN ====================

def load_config():
    """Load configuration."""
    global config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """Configure logging."""
    log_path = Path(__file__).parent.parent.parent / config["paths"]["logs_dir"] / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="10 MB", retention="5 days", level="INFO")

def load_models():
    """Load all trained models."""
    global models, preprocessors, device
    
    device = torch.device(config["general"]["device"])
    logger.info(f"Loading models on {device}")
    
    model_dir = Path(config["paths"]["models_dir"])
    
    # Image detector
    image_model_path = model_dir / "image_detector" / "best_model.pth"
    image_preproc_path = model_dir / "image_detector" / "preprocessor.pkl"
    
    if image_model_path.exists():
        try:
            models["image"] = load_image_detector(str(image_model_path), config, device)
            preprocessors["image"] = InferenceImagePreprocessor(str(image_preproc_path))
            logger.info("Loaded image detector")
        except Exception as e:
            logger.error(f"Failed to load image detector: {e}")
    
    # Video detector
    video_model_path = model_dir / "video_detector" / "best_model.pth"
    video_preproc_path = model_dir / "video_detector" / "preprocessor.pkl"
    
    if video_model_path.exists():
        try:
            models["video"] = load_video_detector(str(video_model_path), config, device)
            preprocessors["video"] = VideoPreprocessor.load(str(video_preproc_path))
            logger.info("Loaded video detector")
        except Exception as e:
            logger.error(f"Failed to load video detector: {e}")
    
    # Audio detector
    audio_model_path = model_dir / "audio_detector" / "best_model.pth"
    audio_preproc_path = model_dir / "audio_detector" / "preprocessor.pkl"
    
    if audio_model_path.exists():
        try:
            models["audio"] = load_audio_detector(str(audio_model_path), config, device)
            preprocessors["audio"] = AudioPreprocessor.load(str(audio_preproc_path))
            logger.info("Loaded audio detector")
        except Exception as e:
            logger.error(f"Failed to load audio detector: {e}")
    
    # Multimodal detector
    multimodal_model_path = model_dir / "multimodal_detector" / "best_model.pth"
    
    if multimodal_model_path.exists():
        try:
            models["multimodal"] = load_multimodal_detector(str(multimodal_model_path), config, device)
            # Use video + audio preprocessors
            logger.info("Loaded multimodal detector")
        except Exception as e:
            logger.error(f"Failed to load multimodal detector: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    load_config()
    setup_logging()
    load_models()
    logger.info("Backend started successfully")
    yield
    # Shutdown
    logger.info("Backend shutting down")

# ==================== FASTAPI APP ====================

# Load config eagerly so CORS origins and upload limits are available at
# import time (lifespan re-loads it on startup; load_config is idempotent).
try:
    load_config()
except Exception as e:
    logger.warning(f"Config preload failed (will retry at startup): {e}")

app = FastAPI(
    title="Deepfake Detection API",
    description="AI-Driven Deepfake Detection & Cyber-Crime Risk Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("backend", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HELPER FUNCTIONS ====================

def validate_file(file: UploadFile) -> tuple:
    """Validate uploaded file."""
    # Check content type
    allowed_image = config["backend"]["upload"]["allowed_image_types"]
    allowed_video = config["backend"]["upload"]["allowed_video_types"]
    allowed_audio = config["backend"]["upload"]["allowed_audio_types"]
    
    all_allowed = allowed_image + allowed_video + allowed_audio
    
    if file.content_type not in all_allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    # Determine file type
    if file.content_type in allowed_image:
        file_type = "image"
        max_size = config["backend"]["upload"]["max_image_size_mb"] * 1024 * 1024
    elif file.content_type in allowed_video:
        file_type = "video"
        max_size = config["backend"]["upload"]["max_video_size_mb"] * 1024 * 1024
    else:
        file_type = "audio"
        max_size = config["backend"]["upload"]["max_audio_size_mb"] * 1024 * 1024
    
    return file_type, max_size

def save_upload_file(file: UploadFile, temp_dir: Path) -> Path:
    """Save uploaded file to temp directory."""
    suffix = Path(file.filename).suffix
    temp_file = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return temp_file

def get_file_hash(filepath: Path) -> str:
    """Calculate SHA-256 hash."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def analyze_image(filepath: Path, file_hash: str) -> Dict[str, Any]:
    """Analyze image file."""
    if "image" not in models:
        raise HTTPException(503, "Image detector not available")
    
    model = models["image"]
    preprocessor = preprocessors["image"]
    
    # Preprocess
    tensor = preprocessor.preprocess_file(str(filepath))
    tensor = tensor.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        pred_class = probs.argmax(dim=1).item()
    
    prediction = "Potentially Manipulated" if pred_class == 1 else "Likely Authentic"
    confidence = prob_fake if pred_class == 1 else probs[0, 0].item()
    
    # Grad-CAM
    explanation = {}
    try:
        target_layer = model._get_target_layer()
        if target_layer:
            gradcam = GradCAM(model, target_layer, use_cuda=device.type == "cuda")
            heatmap = gradcam(tensor, target_class=pred_class)
            # Convert to base64 for frontend
            import base64
            from io import BytesIO
            from PIL import Image
            overlay_img = overlay_heatmap(
                tensor_to_numpy_image(tensor[0].cpu()),
                heatmap
            )
            pil_img = Image.fromarray(overlay_img)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            explanation["gradcam"] = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        explanation["gradcam"] = None
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "deepfake_probability": prob_fake,
        "explanation": explanation
    }

def analyze_video(filepath: Path, file_hash: str) -> Dict[str, Any]:
    """Analyze video file."""
    if "video" not in models:
        raise HTTPException(503, "Video detector not available")
    
    model = models["video"]
    preprocessor = preprocessors["video"]
    
    # Preprocess
    video_tensor, frame_indices, metadata = preprocessor.preprocess(str(filepath))
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        pred_class = probs.argmax(dim=1).item()
    
    prediction = "Potentially Manipulated" if pred_class == 1 else "Likely Authentic"
    confidence = prob_fake if pred_class == 1 else probs[0, 0].item()
    
    # Per-frame predictions
    frame_probs = []
    try:
        frame_logits = model.get_per_frame_logits(video_tensor)
        frame_probs = torch.softmax(frame_logits, dim=2)[0, :, 1].cpu().numpy().tolist()
    except:
        pass
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "deepfake_probability": prob_fake,
        "frame_probabilities": frame_probs,
        "metadata": metadata,
        "explanation": {}
    }

def analyze_audio(filepath: Path, file_hash: str) -> Dict[str, Any]:
    """Analyze audio file."""
    if "audio" not in models:
        raise HTTPException(503, "Audio detector not available")
    
    model = models["audio"]
    preprocessor = preprocessors["audio"]
    
    # Preprocess - RawNet2 needs waveform
    arch = config["audio_detector"]["architecture"]
    return_waveform = arch in ["rawnet2", "wav2vec2"]
    
    if return_waveform:
        waveform, mel_spec = preprocessor.preprocess(str(filepath), return_waveform=True)
        inputs = waveform.unsqueeze(0).to(device)
    else:
        mel_spec = preprocessor.preprocess(str(filepath), return_waveform=False)
        inputs = mel_spec.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_embeddings'):
            logits = model(inputs)
        else:
            logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        pred_class = probs.argmax(dim=1).item()
    
    prediction = "Potentially Manipulated" if pred_class == 1 else "Likely Authentic"
    confidence = prob_fake if pred_class == 1 else probs[0, 0].item()
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "deepfake_probability": prob_fake,
        "explanation": {}
    }

def analyze_multimodal(video_path: Path, audio_path: Path, file_hash: str) -> Dict[str, Any]:
    """Analyze video+audio multimodal."""
    if "multimodal" not in models:
        raise HTTPException(503, "Multimodal detector not available")
    
    model = models["multimodal"]
    
    # Preprocess video
    video_preproc = preprocessors.get("video")
    video_tensor, _, _ = video_preproc.preprocess(str(video_path))
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    # Preprocess audio
    audio_preproc = preprocessors.get("audio")
    arch = config["audio_detector"]["architecture"]
    return_waveform = arch in ["rawnet2", "wav2vec2"]
    
    if return_waveform:
        waveform, _ = audio_preproc.preprocess(str(audio_path), return_waveform=True)
        audio_input = waveform.unsqueeze(0).to(device)
    else:
        mel_spec = audio_preproc.preprocess(str(audio_path), return_waveform=False)
        audio_input = mel_spec.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(video_tensor, audio_input)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        pred_class = probs.argmax(dim=1).item()
    
    prediction = "Potentially Manipulated" if pred_class == 1 else "Likely Authentic"
    confidence = prob_fake if pred_class == 1 else probs[0, 0].item()
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "deepfake_probability": prob_fake,
        "explanation": {}
    }

def get_model_version(file_type: str) -> str:
    """Read model version from the detector's metadata.json (never invented)."""
    subdir = {
        "image": "image_detector",
        "video": "video_detector",
        "audio": "audio_detector",
        "multimodal": "multimodal_detector",
    }.get(file_type, "image_detector")
    meta_path = Path(config["paths"]["models_dir"]) / subdir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f).get("version", "unknown")
        except Exception:
            return "unknown"
    return "unknown"

# ==================== API ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        models_loaded={k: k in models for k in ["image", "video", "audio", "multimodal"]},
        device=str(device)
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_content: str = Form(""),
    user_context: str = Form("{}")
):
    """
    Analyze uploaded media for deepfake detection and risk assessment.
    
    Supports: image, video, audio files.
    """
    # Validate file
    file_type, max_size = validate_file(file)
    
    # Parse user context
    try:
        context = json.loads(user_context) if user_context else {}
    except:
        context = {}
    
    # Create temp directory
    temp_dir = Path(config["paths"]["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    filepath = save_upload_file(file, temp_dir)
    
    try:
        # Check file size
        file_size = filepath.stat().st_size
        if file_size > max_size:
            raise HTTPException(413, f"File too large: {file_size} > {max_size}")
        
        # Calculate hash
        file_hash = get_file_hash(filepath)
        
        # Get media properties
        media_props = {"filename": file.filename, "size": file_size, "content_type": file.content_type}
        
        # Run appropriate analysis
        if file_type == "image":
            result = analyze_image(filepath, file_hash)
            try:
                from PIL import Image as PILImage
                with PILImage.open(filepath) as _img:
                    media_props.update({
                        "width": _img.width,
                        "height": _img.height,
                        "mode": _img.mode,
                        "format": _img.format,
                    })
            except Exception as e:
                logger.warning(f"Could not read image dimensions: {e}")
        elif file_type == "video":
            result = analyze_video(filepath, file_hash)
            media_props.update(result.get("metadata", {}))
        elif file_type == "audio":
            result = analyze_audio(filepath, file_hash)
        else:
            raise HTTPException(400, f"Unsupported file type: {file_type}")
        
        # Forensic analysis
        forensic_analyzer = create_forensic_analyzer(config)
        forensic_results = forensic_analyzer.analyze_file(str(filepath))
        
        # Risk assessment
        risk_assessor = create_risk_assessor(config)
        risk_result = risk_assessor.assess(
            deepfake_probability=result["deepfake_probability"],
            media_type=file_type,
            text_content=text_content,
            metadata=forensic_results.get("metadata", {}),
            forensic_indicators=forensic_results,
            user_context=context
        )
        
        # Create evidence record
        evidence_manager = create_evidence_manager(config)
        model_version = get_model_version(file_type)
        
        record = evidence_manager.create_analysis_record(
            file_path=str(filepath),
            media_type=file_type,
            media_properties=media_props,
            model_version=model_version,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_level=risk_result.risk_level.value,
            forensic_indicators=forensic_results,
            risk_assessment={
                "risk_score": risk_result.risk_score,
                "deepfake_probability": risk_result.deepfake_probability,
                "indicators": [asdict(ind) for ind in risk_result.indicators]
            },
            explanation_artifacts=result.get("explanation", {}),
            recommendations=risk_result.recommended_actions
        )
        
        # Save record
        evidence_manager.save_record(record)
        evidence_manager.save_human_report(record)
        
        # Prepare response
        response = AnalysisResponse(
            analysis_id=record.analysis_id,
            timestamp=record.timestamp,
            file_type=file_type,
            file_hash=file_hash,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_level=risk_result.risk_level.value,
            risk_score=risk_result.risk_score,
            deepfake_probability=result["deepfake_probability"],
            media_properties=media_props,
            forensic_indicators=forensic_results,
            risk_assessment={
                "risk_score": risk_result.risk_score,
                "deepfake_probability": risk_result.deepfake_probability,
                "indicators": [asdict(ind) for ind in risk_result.indicators]
            },
            explanation=result.get("explanation", {}),
            recommendations=risk_result.recommended_actions,
            india_guidance=risk_result.india_specific_guidance,
            model_version=model_version
        )
        
        return response
        
    finally:
        # Cleanup temp file
        try:
            filepath.unlink()
        except:
            pass

@app.get("/analysis/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis(analysis_id: str):
    """Get analysis by ID."""
    evidence_manager = create_evidence_manager(config)
    record = evidence_manager.load_record(analysis_id)
    
    if not record:
        raise HTTPException(404, "Analysis not found")
    
    return record.to_dict()

@app.get("/history", response_model=List[HistoryItem])
async def get_history(limit: int = 50, offset: int = 0):
    """Get analysis history."""
    evidence_manager = create_evidence_manager(config)
    records = evidence_manager.list_records(limit=limit, offset=offset)
    
    return [
        HistoryItem(
            analysis_id=r["analysis_id"],
            timestamp=r["timestamp"],
            file_type=r["file_type"],
            prediction=r["prediction"],
            confidence=r["confidence"],
            risk_level=r["risk_level"],
            model_version=r["model_version"]
        )
        for r in records
    ]

@app.get("/analysis/{analysis_id}/report")
async def get_report(analysis_id: str, format: str = "json"):
    """Get analysis report in specified format."""
    evidence_manager = create_evidence_manager(config)
    record = evidence_manager.load_record(analysis_id)
    
    if not record:
        raise HTTPException(404, "Analysis not found")
    
    if format == "json":
        return record.to_dict()
    elif format == "text":
        report_gen = create_report_generator()
        return {"report": report_gen.generate_html_report(record)}
    elif format == "html":
        report_gen = create_report_generator()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(report_gen.generate_html_report(record))
    else:
        raise HTTPException(400, f"Unsupported format: {format}")

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get loaded model information."""
    model_dir = Path(config["paths"]["models_dir"])
    
    def get_model_info(name: str, subdir: str) -> Dict:
        path = model_dir / subdir / "best_model.pth"
        meta_path = model_dir / subdir / "metadata.json"
        
        info = {
            "loaded": name in models,
            "path": str(path) if path.exists() else None,
        }
        
        if meta_path.exists():
            with open(meta_path) as f:
                info["metadata"] = json.load(f)
        
        return info
    
    return ModelsResponse(
        image_detector=get_model_info("image", "image_detector"),
        video_detector=get_model_info("video", "video_detector"),
        audio_detector=get_model_info("audio", "audio_detector"),
        multimodal_detector=get_model_info("multimodal", "multimodal_detector")
    )

@app.get("/analytics")
async def get_analytics():
    """Get analytics dashboard data."""
    evidence_manager = create_evidence_manager(config)
    records = evidence_manager.list_records(limit=1000)
    
    if not records:
        return {"message": "No analyses yet"}
    
    # Compute statistics
    total = len(records)
    by_type = {}
    by_prediction = {}
    by_risk = {}
    avg_confidence = 0
    
    for r in records:
        by_type[r["file_type"]] = by_type.get(r["file_type"], 0) + 1
        by_prediction[r["prediction"]] = by_prediction.get(r["prediction"], 0) + 1
        by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1
        avg_confidence += r["confidence"]
    
    avg_confidence /= total
    
    return {
        "total_analyses": total,
        "by_media_type": by_type,
        "by_prediction": by_prediction,
        "by_risk_level": by_risk,
        "average_confidence": avg_confidence,
        "recent_analyses": records[:10]
    }

@app.post("/validate")
async def validate_file_endpoint(file: UploadFile = File(...)):
    """Validate file without full analysis."""
    file_type, max_size = validate_file(file)
    
    temp_dir = Path(config["paths"]["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_upload_file(file, temp_dir)
    
    try:
        file_hash = get_file_hash(filepath)
        size = filepath.stat().st_size
        
        return {
            "valid": True,
            "file_type": file_type,
            "size": size,
            "max_size": max_size,
            "sha256": file_hash,
            "content_type": file.content_type
        }
    finally:
        try:
            filepath.unlink()
        except:
            pass

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=config.get("backend", {}).get("host", "0.0.0.0"),
        port=config.get("backend", {}).get("port", 8000),
        reload=config.get("backend", {}).get("reload", False)
    )