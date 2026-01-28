"""
DTO for IP Validation Requests
"""

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import ipaddress


class IPValidationRequestDTO(BaseModel):
    """DTO for IP validation request"""
    ip: str = Field(..., example="192.168.1.100", description="IPv4 address to validate")
    
    class Config:
        anystr_strip_whitespace = True
        validate_assignment = True
    
    @validator("ip")
    def validate_ip_format(cls, v):
        """Validate IP format"""
        if not v or not v.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "EMPTY_IP",
                    "message": "IP address cannot be empty",
                    "field": "ip"
                }
            )
        
        try:
            ipaddress.IPv4Address(v.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_IP_FORMAT",
                    "message": f"'{v}' is not a valid IPv4 address",
                    "field": "ip"
                }
            )
        return v.strip()


class IPValidationResponseDTO(BaseModel):
    """DTO for IP validation response"""
    ip: str
    is_valid_format: bool
    is_unique: bool
    validation_id: str
    timestamp: str
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "ip": "192.168.1.100",
                "is_valid_format": True,
                "is_unique": False,
                "validation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-01-24T10:30:00.000000",
                "message": "IP address is already in use by camera: cam_0001"
            }
        }