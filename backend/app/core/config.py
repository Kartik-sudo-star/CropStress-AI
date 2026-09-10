"""
Core Configuration Management
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import yaml
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from config.yaml and environment variables."""
    
    # General
    PROJECT_NAME: str = "Deepfake Detection API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///./deepfake_analysis.db"
    
    # File Upload
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_AUDIO_SIZE_MB: int = 20
    
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]
    
    ALLOWED_VIDEO_TYPES: List[str] = [
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo"
    ]
    
    ALLOWED_AUDIO_TYPES: List[str] = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/x-m4a"
    ]
    
    # Model Settings
    CONFIDENCE_THRESHOLD: float = 0.5
    ENABLE_EXPLAINABILITY: bool = True
    ENABLE_FORENSIC: bool = True
    ENABLE_RISK_ASSESSMENT: bool = True
    
    # Paths
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    TEMP_DIR: str = "temp"
    REPORTS_DIR: str = "reports"
    
    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Device
    DEVICE: str = "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    # Load from config.yaml first, then override with env vars
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    yaml_config = {}
    
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}
    
    # Map YAML config to settings
    settings_dict = {}
    
    if "backend" in yaml_config:
        b = yaml_config["backend"]
        settings_dict.update({
            "HOST": b.get("host", "0.0.0.0"),
            "PORT": b.get("port", 8000),
            "DEBUG": b.get("debug", False),
            "CORS_ORIGINS": b.get("cors_origins", []),
            "DATABASE_URL": b.get("database", {}).get("url", "sqlite:///./deepfake_analysis.db"),
        })
    
    if "general" in yaml_config:
        settings_dict["DEVICE"] = yaml_config["general"].get("device", "cpu")
    
    if "paths" in yaml_config:
        p = yaml_config["paths"]
        settings_dict.update({
            "MODELS_DIR": p.get("models_dir", "models"),
            "LOGS_DIR": p.get("logs_dir", "logs"),
            "TEMP_DIR": p.get("temp_dir", "temp"),
            "REPORTS_DIR": p.get("reports_dir", "reports"),
        })
    
    return Settings(**settings_dict)


settings = get_settings()