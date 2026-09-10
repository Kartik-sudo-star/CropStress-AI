"""
Database Models for Deepfake Detection System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class AnalysisRecord(Base):
    """Database model for analysis records."""
    __tablename__ = "analysis_records"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # File info
    file_name = Column(String(255))
    file_type = Column(String(32), index=True)  # image, video, audio
    file_hash = Column(String(64), index=True)  # SHA-256
    file_size = Column(Integer)
    content_type = Column(String(64))
    
    # Media properties
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    
    # Model info
    model_version = Column(String(32))
    model_architecture = Column(String(64))
    
    # Prediction results
    prediction = Column(String(64), index=True)  # Potentially Manipulated, Likely Authentic
    confidence = Column(Float)
    deepfake_probability = Column(Float)
    risk_level = Column(String(16), index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score = Column(Float)
    
    # JSON fields for complex data
    forensic_indicators = Column(JSON, nullable=True)
    risk_assessment = Column(JSON, nullable=True)
    explanation_artifacts = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    india_guidance = Column(JSON, nullable=True)
    
    # Status
    status = Column(String(32), default="completed")
    processing_time_ms = Column(Integer, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_analysis_timestamp_type", "timestamp", "file_type"),
        Index("ix_analysis_risk_level", "risk_level"),
    )


class ModelMetadata(Base):
    """Database model for model version tracking."""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(64), unique=True, index=True, nullable=False)
    version = Column(String(32), nullable=False)
    architecture = Column(String(64))
    training_date = Column(DateTime, default=func.now())
    dataset_version = Column(String(32))
    preprocessing_version = Column(String(32))
    
    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    eer = Column(Float, nullable=True)
    
    # Config
    class_mapping = Column(JSON)
    hyperparameters = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=True)
    deployed_at = Column(DateTime, nullable=True)


class DatasetRecord(Base):
    """Database model for dataset tracking."""
    __tablename__ = "dataset_records"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(64), index=True)
    version = Column(String(32))
    split = Column(String(16))  # train, validation, test
    media_type = Column(String(16))  # image, video, audio
    num_samples = Column(Integer)
    class_distribution = Column(JSON)
    source = Column(String(255))
    license = Column(String(128))
    created_at = Column(DateTime, default=func.now())
    manifest_path = Column(String(512))


class SystemLog(Base):
    """Database model for system logs."""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    level = Column(String(16))
    module = Column(String(64))
    message = Column(Text)
    details = Column(JSON, nullable=True)
    request_id = Column(String(64), nullable=True, index=True)


def create_tables(engine):
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all tables."""
    Base.metadata.drop_all(bind=engine)