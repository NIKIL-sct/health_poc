"""
SQLAlchemy Models for Camera Health Monitoring System
ADD THIS CLASS TO YOUR EXISTING storage/models.py

File: storage/models.py
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


# ============================================
# EXISTING MODELS (NO CHANGES)
# ============================================

class Camera(Base):
    """CAMERA Table - Stores camera configuration"""
    __tablename__ = 'cameras'
    
    camera_id = Column(String(64), primary_key=True)
    ip = Column(String(45), nullable=False)
    port = Column(Integer, nullable=False)
    rtsp_url = Column(Text, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    
    interval_ip = Column(Integer, default=60)
    interval_port = Column(Integer, default=15)
    interval_vision = Column(Integer, default=120)
    
    health_logs = relationship("HealthLog", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Camera(id={self.camera_id}, ip={self.ip}, enabled={self.enabled})>"


class HealthLog(Base):
    """HEALTH_LOG Table - Stores all health check events"""
    __tablename__ = 'health_logs'
    
    event_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(64), ForeignKey('cameras.camera_id', ondelete='CASCADE'), nullable=False)
    
    status = Column(String(20), nullable=False)
    event_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_description = Column(Text)
    meta_data = Column(JSON)
    last_checked = Column(DateTime, default=datetime.utcnow)
    
    camera = relationship("Camera", back_populates="health_logs")
    alerts = relationship("Alert", back_populates="health_log", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<HealthLog(id={self.event_id}, camera={self.camera_id}, type={self.event_type}, status={self.status})>"


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
    
# ============================================
# NEW MODEL - CAMERA LATENCY (CONNECTIVITY CHECKS)
# ============================================

class CameraLatency(Base):
    """
    CAMERA_LATENCY Table - Stores latency and packet loss metrics
    Used for connectivity checks: latency measurement and packet loss calculation
    """
    __tablename__ = 'camera_latency'
    
    # Primary Key
    latency_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign Key to Camera
    camera_id = Column(String(64), nullable=False, index=True)
    
    # Latency Metrics (in milliseconds)
    rtt_avg = Column(Float, nullable=True, comment="Average Round Trip Time in ms")
    rtt_min = Column(Float, nullable=True, comment="Minimum Round Trip Time in ms")
    rtt_max = Column(Float, nullable=True, comment="Maximum Round Trip Time in ms")
    
    # Packet Loss Metrics
    packet_loss_percent = Column(Float, nullable=True, comment="Packet loss percentage")
    packets_sent = Column(Integer, nullable=True, comment="Number of packets sent")
    packets_received = Column(Integer, nullable=True, comment="Number of packets received")
    
    # Connectivity Status
    is_reachable = Column(Boolean, default=True, comment="Whether camera is reachable")
    
    # Additional Metadata
    check_type = Column(String(20), default="PING", comment="Type of connectivity check (PING, TCP, etc.)")
    error_message = Column(Text, nullable=True, comment="Error message if check failed")
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return (
            f"<CameraLatency(camera_id={self.camera_id}, "
            f"rtt_avg={self.rtt_avg}ms, loss={self.packet_loss_percent}%)>"
        )
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "latency_id": self.latency_id,
            "camera_id": self.camera_id,
            "rtt_avg": self.rtt_avg,
            "rtt_min": self.rtt_min,
            "rtt_max": self.rtt_max,
            "packet_loss_percent": self.packet_loss_percent,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "is_reachable": self.is_reachable,
            "check_type": self.check_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
