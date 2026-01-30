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
    port: int = Field(..., ge=1, le=65535, example=554)
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
        port = values.get("port")
        if parsed.port and port and parsed.port != port:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "port_MISMATCH",
                    "message": "RTSP URL port does not match port field",
                    "field": "port"
                }
            )

        return v

class ScheduleConfigRequest(BaseModel):
    """Request model for scheduling configuration"""
    interval_ip: Optional[int] = Field(None, ge=10, le=3600, description="IP check interval in seconds (10-3600)")
    interval_port: Optional[int] = Field(None, ge=5, le=1800, description="Port check interval in seconds (5-1800)")
    interval_vision: Optional[int] = Field(None, ge=30, le=7200, description="Vision check interval in seconds (30-7200)")
    
    @validator('interval_ip', 'interval_port', 'interval_vision')
    def validate_intervals(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Interval must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "interval_ip": 120,
                "interval_port": 30,
                "interval_vision": 300
            }
        }


class TriggerCheckRequest(BaseModel):
    """Request model for triggering immediate checks"""
    check_types: Optional[list[str]] = Field(
        default=None,
        description="Types of checks to trigger: ['ip', 'port', 'vision']. If None, triggers all"
    )
    
    @validator('check_types')
    def validate_check_types(cls, v):
        if v is not None:
            valid_types = {'ip', 'port', 'vision'}
            invalid = set(v) - valid_types
            if invalid:
                raise ValueError(f'Invalid check types: {invalid}. Must be one of {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "check_types": ["ip", "port"]
            }
        }
