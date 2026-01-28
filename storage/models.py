"""
SQLAlchemy Models for Camera Health Monitoring System
ADD THIS CLASS TO YOUR EXISTING storage/models.py

File: storage/models.py
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


# ============================================
# EXISTING MODELS (NO CHANGES)
# ============================================

# storage/models.py

class Camera(Base):
    """CAMERA Table - Stores camera configuration"""
    __tablename__ = 'cameras'
    
    camera_id = Column(String(64), primary_key=True)
    ip = Column(String(45), nullable=False)
    rtsp_port = Column(Integer, nullable=False)  # Match your DB!
    rtsp_url = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)  # If DB has created_at
    # OR
    # started_at = Column(DateTime, default=datetime.utcnow, nullable=False)  # If DB has started_at
    enabled = Column(Boolean, default=True, nullable=False)
    
    interval_ip = Column(Integer, default=60)
    interval_port = Column(Integer, default=15)
    interval_vision = Column(Integer, default=120)
    
    health_logs = relationship("HealthLog", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")

class HealthLog(Base):
    __tablename__ = 'health_logs'

    event_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(
        String(64),
        ForeignKey('cameras.camera_id', ondelete='CASCADE'),
        nullable=False
    )

    status = Column(String(20), nullable=False)
    event_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_description = Column(Text)
    meta_data = Column(JSON)
    last_checked = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="health_logs")
    alerts = relationship("Alert", back_populates="health_log", cascade="all, delete-orphan")


class Alert(Base):
    """ALERT Table - Stores alerts generated from health checks"""
    __tablename__ = 'alerts'
    
    alert_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(64), ForeignKey('cameras.camera_id', ondelete='CASCADE'), nullable=False)
    event_id = Column(String(36), ForeignKey('health_logs.event_id', ondelete='CASCADE'), nullable=False)
    
    alert_type = Column(String(50), nullable=False)
    alert_description = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    
    camera = relationship("Camera", back_populates="alerts")
    health_log = relationship("HealthLog", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert(id={self.alert_id}, camera={self.camera_id}, type={self.alert_type})>"


# ============================================
# NEW MODEL - IP VALIDATION
# ============================================

class IPValidation(Base):
    """
    IP_VALIDATION Table - Stores IP validation results
    Used to track all IP validation attempts during camera registration
    """
    __tablename__ = 'ip_validations'
    
    validation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ip = Column(String(45), nullable=False, index=True)
    is_valid_format = Column(Boolean, nullable=False)
    is_unique = Column(Boolean, nullable=False)
    validation_result = Column(String(20), nullable=False, index=True)  # SUCCESS, INVALID_FORMAT, DUPLICATE
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<IPValidation(id={self.validation_id}, ip={self.ip}, result={self.validation_result})>"