"""
IP Validation Service
File: core/ip_validation_service.py
"""

import ipaddress
import logging
from typing import Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from storage.db_repositary import IPValidationRepository
from storage.redis_client import get_async_redis, RedisKeys

logger = logging.getLogger(__name__)


class IPValidationService:
    """Service for IP validation operations"""
    
    @staticmethod
    def validate_ip_format(ip: str) -> Tuple[bool, str]:
        """
        Validate IP address format
        
        Args:
            ip: IP address string
        
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            # Strip whitespace
            ip = ip.strip()
            
            if not ip:
                return False, "IP address cannot be empty"
            
            # Validate IPv4 format
            ipaddress.IPv4Address(ip)
            return True, "Valid IPv4 address format"
            
        except ValueError as e:
            return False, f"Invalid IPv4 format: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error validating IP format: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    async def check_ip_uniqueness_db(
        db: AsyncSession,
        ip: str
    ) -> Tuple[bool, str]:
        """
        Check IP uniqueness against database
        
        Args:
            db: Database session
            ip: IP address to check
        
        Returns:
            Tuple of (is_unique: bool, message: str)
        """
        try:
            is_unique, camera_id = await IPValidationRepository.check_ip_uniqueness_async(db, ip)
            
            if is_unique:
                return True, "IP address is unique"
            else:
                return False, f"IP address is already in use by camera: {camera_id}"
                
        except Exception as e:
            logger.error(f"Error checking IP uniqueness in DB: {e}")
            # Return unique on error to not block operations
            return True, f"Could not verify uniqueness (DB error): {str(e)}"
    
    @staticmethod
    async def check_ip_uniqueness_redis(ip: str) -> Tuple[bool, str]:
        """
        Check IP uniqueness against Redis cache
        
        Args:
            ip: IP address to check
        
        Returns:
            Tuple of (is_unique: bool, message: str)
        """
        try:
            r = await get_async_redis()
            
            # Get all active cameras from Redis
            camera_ids = await r.smembers(RedisKeys.active_cameras())
            
            if not camera_ids:
                return True, "IP address is unique (no cameras in Redis)"
            
            # Check each camera's IP
            for camera_id in camera_ids:
                camera_key = RedisKeys.camera(camera_id)
                camera_ip = await r.hget(camera_key, "ip")
                
                if camera_ip and camera_ip == ip:
                    return False, f"IP address is already in use by camera: {camera_id} (Redis)"
            
            return True, "IP address is unique"
            
        except Exception as e:
            logger.error(f"Error checking IP uniqueness in Redis: {e}")
            # Return unique on error to not block operations
            return True, f"Could not verify uniqueness (Redis error): {str(e)}"
    
    @staticmethod
    async def validate_and_store(
        db: AsyncSession,
        ip: str
    ) -> Dict:
        """
        Complete IP validation: format check + uniqueness check + storage
        
        Args:
            db: Database session
            ip: IP address to validate
        
        Returns:
            Dict with validation results
        """
        # Step 1: Validate format
        is_valid_format, format_message = IPValidationService.validate_ip_format(ip)
        
        # Step 2: Check uniqueness (both DB and Redis)
        is_unique_db = True
        is_unique_redis = True
        uniqueness_message = "IP is unique"
        
        if is_valid_format:
            # Check database first
            is_unique_db, db_message = await IPValidationService.check_ip_uniqueness_db(db, ip)
            
            # Check Redis as secondary source
            is_unique_redis, redis_message = await IPValidationService.check_ip_uniqueness_redis(ip)
            
            # Use DB as primary source of truth
            is_unique = is_unique_db
            uniqueness_message = db_message if not is_unique_db else redis_message
        else:
            is_unique = True  # Not applicable if format is invalid
            uniqueness_message = "Format validation failed"
        
        # Step 3: Determine overall result
        if not is_valid_format:
            validation_result = "INVALID_FORMAT"
            final_message = format_message
        elif not is_unique:
            validation_result = "DUPLICATE"
            final_message = uniqueness_message
        else:
            validation_result = "SUCCESS"
            final_message = "IP address is valid and unique"
        
        # Step 4: Store validation result in database
        try:
            validation_record = await IPValidationRepository.create_validation_async(
                db=db,
                ip=ip,
                is_valid_format=is_valid_format,
                is_unique=is_unique,
                validation_result=validation_result,
                message=final_message
            )
            
            # Commit the transaction
            await db.commit()
            
            return {
                "validation_id": validation_record.validation_id,
                "ip": ip,
                "is_valid_format": is_valid_format,
                "is_unique": is_unique,
                "validation_result": validation_result,
                "message": final_message,
                "timestamp": validation_record.timestamp.isoformat(),
                "storage": "PostgreSQL"
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error storing validation result: {e}")
            
            # Return validation results even if storage failed
            return {
                "validation_id": None,
                "ip": ip,
                "is_valid_format": is_valid_format,
                "is_unique": is_unique,
                "validation_result": validation_result,
                "message": final_message,
                "timestamp": None,
                "storage_error": str(e)
            }