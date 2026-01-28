"""
DTO Validators for Vigil-X Camera Health Service
All request validation & error normalization lives here
"""

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
from typing import Optional
import re
from urllib.parse import urlparse
import ipaddress
from fastapi import HTTPException, status


# -------------------------------------------------
# Base DTO (Common behavior)
# -------------------------------------------------

class BaseDTO(BaseModel):
    class Config:
        anystr_strip_whitespace = True
        validate_assignment = True


# -------------------------------------------------
# Camera Request DTO
# -------------------------------------------------

class CameraRequestDTO(BaseDTO):
    id: str = Field(..., min_length=3, max_length=64, example="cam_0001")
    ip: str = Field(..., example="10.10.2.11")
    rtsp_port: int = Field(..., ge=1, le=65535, example=554)
    rtsp_url: str = Field(..., example="rtsp://user:pass@10.10.2.11:554/stream")

    

    # -------------------------
    # IP Validation
    # -------------------------
    @validator("ip")
    def validate_ip(cls, v):
        try:
            ipaddress.IPv4Address(v)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_IP_ADDRESS",
                    "message": "Invalid IPv4 address",
                    "field": "ip"
                }
            )
        return v

    # -------------------------
    # RTSP URL Validation
    # -------------------------
    @validator("rtsp_url")
    def validate_rtsp_url(cls, v, values):
        parsed = urlparse(v)

        if parsed.scheme != "rtsp":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_RTSP_URL",
                    "message": "RTSP URL must start with rtsp://",
                    "field": "rtsp_url"
                }
            )

        if not parsed.hostname:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_RTSP_HOST",
                    "message": "RTSP URL must contain a valid hostname",
                    "field": "rtsp_url"
                }
            )

        # Optional: Cross-check port
        rtsp_port = values.get("rtsp_port")
        if parsed.port and rtsp_port and parsed.port != rtsp_port:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "RTSP_PORT_MISMATCH",
                    "message": "RTSP URL port does not match rtsp_port field",
                    "field": "rtsp_port"
                }
            )

        return v

"""
Data Transfer Objects (DTOs) for Camera Health Monitoring - ENHANCED
Includes new DTOs for connectivity metrics and latency tracking
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================
# EXISTING DTOs (No Changes)
# ============================================

class CameraRequestDTO(BaseModel):
    """Request DTO for camera registration"""
    id: str = Field(..., description="Unique camera identifier")
    ip: str = Field(..., description="Camera IP address")
    rtsp_port: int = Field(554, description="RTSP port number")
    rtsp_url: str = Field(..., description="Full RTSP URL")


class CameraResponseDTO(BaseModel):
    """Response DTO for camera operations"""
    camera_id: str
    ip: str
    rtsp_port: int
    rtsp_url: str
    enabled: bool
    started_at: datetime
    interval_ip: int
    interval_port: int
    interval_vision: int


class HealthLogDTO(BaseModel):
    """DTO for health check event logs"""
    event_id: str
    camera_id: str
    status: str
    event_type: str
    timestamp: datetime
    event_description: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    last_checked: Optional[datetime] = None


class AlertDTO(BaseModel):
    """DTO for alerts"""
    alert_id: str
    camera_id: str
    event_id: str
    alert_type: str
    alert_description: str
    timestamp: datetime
    resolved: bool
    resolved_at: Optional[datetime] = None


class IPValidationRequestDTO(BaseModel):
    """Request DTO for IP validation"""
    ip: str = Field(..., description="IP address to validate")


class IPValidationResponseDTO(BaseModel):
    """Response DTO for IP validation"""
    validation_id: Optional[str]
    ip: str
    is_valid_format: bool
    is_unique: bool
    validation_result: str  # SUCCESS, INVALID_FORMAT, DUPLICATE
    message: str
    timestamp: Optional[str]


# ============================================
# NEW DTOs FOR CONNECTIVITY METRICS
# ============================================

class PingMetricsDTO(BaseModel):
    """DTO for ping metrics (latency and packet loss)"""
    is_reachable: bool = Field(..., description="Whether IP is reachable")
    packets_sent: int = Field(..., description="Number of packets sent")
    packets_received: int = Field(..., description="Number of packets received")
    packet_loss_percent: float = Field(..., description="Packet loss percentage")
    rtt_min_ms: float = Field(..., description="Minimum round-trip time in milliseconds")
    rtt_avg_ms: float = Field(..., description="Average round-trip time in milliseconds")
    rtt_max_ms: float = Field(..., description="Maximum round-trip time in milliseconds")
    rtt_stddev_ms: float = Field(..., description="Standard deviation of RTT in milliseconds")
    error: Optional[str] = Field(None, description="Error message if check failed")


class PortMetricsDTO(BaseModel):
    """DTO for port connection metrics"""
    is_accessible: bool = Field(..., description="Whether port is accessible")
    attempts: int = Field(..., description="Number of connection attempts")
    successful_connections: int = Field(..., description="Number of successful connections")
    connection_success_rate: float = Field(..., description="Connection success rate percentage")
    latency_min_ms: float = Field(..., description="Minimum connection latency in milliseconds")
    latency_avg_ms: float = Field(..., description="Average connection latency in milliseconds")
    latency_max_ms: float = Field(..., description="Maximum connection latency in milliseconds")
    error: Optional[str] = Field(None, description="Error message if check failed")


class ConnectivityResultDTO(BaseModel):
    """DTO for complete connectivity verification result"""
    network_reachable: bool = Field(..., description="Whether network layer is reachable")
    service_accessible: bool = Field(..., description="Whether service port is accessible")
    overall_status: str = Field(
        ..., 
        description="Overall status: HEALTHY, DEGRADED, SERVICE_DOWN, PING_BLOCKED, DOWN"
    )
    ping_metrics: PingMetricsDTO = Field(..., description="Detailed ping metrics")
    port_metrics: PortMetricsDTO = Field(..., description="Detailed port metrics")
    timestamp: str = Field(..., description="Timestamp of check")


class ConnectivityCheckRequestDTO(BaseModel):
    """Request DTO for on-demand connectivity check"""
    camera_id: str = Field(..., description="Camera ID to check")
    ping_count: int = Field(5, description="Number of ping packets to send", ge=1, le=10)
    port_attempts: int = Field(3, description="Number of port connection attempts", ge=1, le=5)


class ConnectivityCheckResponseDTO(BaseModel):
    """Response DTO for connectivity check"""
    camera_id: str
    target_ip: str
    rtsp_ip: str
    rtsp_port: int
    connectivity: ConnectivityResultDTO
    checked_at: datetime


class CameraSummaryDTO(BaseModel):
    """Enhanced camera summary with connectivity metrics"""
    camera_id: str
    ip_status: str = Field(..., description="IP status: UP, DOWN, UNKNOWN")
    port_status: str = Field(..., description="Port status: UP, DOWN, UNKNOWN")
    vision_status: str = Field(..., description="Vision status")
    overall_status: str = Field(..., description="Overall connectivity status")
    network_reachable: bool
    service_accessible: bool
    alert_active: bool
    last_ip_check: Optional[str] = None
    last_port_check: Optional[str] = None
    last_vision_check: Optional[str] = None
    last_connectivity_check: Optional[str] = None
    ip_metrics: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Latest IP metrics (packet_loss, rtt_avg, etc.)"
    )
    port_metrics: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Latest port metrics (success_rate, latency_avg, etc.)"
    )
    connectivity_metrics: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Latest full connectivity metrics"
    )


class MetricsHistoryDTO(BaseModel):
    """DTO for historical metrics data"""
    camera_id: str
    metric_type: str  # "ip_check", "port_check", "connectivity_check"
    time_range: str
    data_points: List[Dict[str, Any]]
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistical summary (avg, min, max, etc.)"
    )


class NetworkHealthReportDTO(BaseModel):
    """DTO for comprehensive network health report"""
    camera_id: str
    report_timestamp: datetime
    ip_health: Dict[str, Any]
    port_health: Dict[str, Any]
    overall_health: Dict[str, Any]
    recommendations: List[str] = Field(
        default_factory=list,
        description="System recommendations based on metrics"
    )