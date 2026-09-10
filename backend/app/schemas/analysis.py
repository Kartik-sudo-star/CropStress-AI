"""
Pydantic Schemas for API Request/Response Validation
"""
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class PredictionLabel(str, Enum):
    MANIPULATED = "Potentially Manipulated"
    AUTHENTIC = "Likely Authentic"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Request schemas
class AnalyzeRequest(BaseModel):
    text_content: Optional[str] = Field("", description="Associated text/caption for context analysis")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context (platform, account info, etc.)")


class ValidateRequest(BaseModel):
    pass  # File is in multipart form


# Response schemas
class MediaProperties(BaseModel):
    filename: str
    size: int
    content_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


class ExplanationArtifacts(BaseModel):
    gradcam: Optional[str] = None  # Base64 encoded image
    frame_probabilities: Optional[List[float]] = None
    attention_weights: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None


class ForensicIndicators(BaseModel):
    metadata: Optional[Dict[str, Any]] = None
    ela: Optional[Dict[str, Any]] = None
    noise_analysis: Optional[Dict[str, Any]] = None
    copy_move: Optional[Dict[str, Any]] = None
    metadata_consistency: Optional[Dict[str, Any]] = None
    frame_analysis: Optional[Dict[str, Any]] = None
    gop_analysis: Optional[Dict[str, Any]] = None
    duplication: Optional[Dict[str, Any]] = None
    spectral_analysis: Optional[Dict[str, Any]] = None
    splicing: Optional[Dict[str, Any]] = None
    sample_rate_check: Optional[Dict[str, Any]] = None


class RiskIndicatorSchema(BaseModel):
    category: str
    name: str
    description: str
    weight: float
    confidence: float
    evidence: str = ""
    matched_keywords: List[str] = []


class RiskAssessmentResponse(BaseModel):
    risk_score: float
    deepfake_probability: float
    indicators: List[RiskIndicatorSchema]
    recommended_actions: List[str]


class IndiaGuidance(BaseModel):
    reporting_channels: List[Dict[str, str]] = []
    legal_references: List[str] = []
    evidence_preservation: List[str] = []
    helplines: List[Dict[str, str]] = []
    urgent_actions: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    file_type: FileType
    file_hash: str
    prediction: PredictionLabel
    confidence: float
    risk_level: RiskLevel
    risk_score: float
    deepfake_probability: float
    media_properties: MediaProperties
    forensic_indicators: ForensicIndicators
    risk_assessment: RiskAssessmentResponse
    explanation: ExplanationArtifacts
    recommendations: List[str]
    india_guidance: IndiaGuidance
    model_version: str


class HistoryItem(BaseModel):
    analysis_id: str
    timestamp: datetime
    file_type: FileType
    prediction: PredictionLabel
    confidence: float
    risk_level: RiskLevel
    model_version: str


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int


class ReportFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    HTML = "html"


class ReportResponse(BaseModel):
    analysis_id: str
    format: ReportFormat
    content: str
    generated_at: datetime


class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    image_detector: ModelInfo
    video_detector: ModelInfo
    audio_detector: ModelInfo
    multimodal_detector: ModelInfo


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    device: str


class AnalyticsResponse(BaseModel):
    total_analyses: int
    by_media_type: Dict[str, int]
    by_prediction: Dict[str, int]
    by_risk_level: Dict[str, int]
    average_confidence: float
    recent_analyses: List[HistoryItem]


class ValidationResponse(BaseModel):
    valid: bool
    file_type: FileType
    size: int
    max_size: int
    sha256: str
    content_type: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)