"""
Camera Logger with Database Integration
DUAL WRITE: Redis (cache) + PostgreSQL (persistent storage)
Optional JSON logging based on ENABLE_CAMERA_JSON_LOGS flag
FIXED: Consistent use of 'meta_data' instead of 'metadata'
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict

from storage.db_config import get_sync_db
from storage.db_repositary import (
    CameraRepository,
    HealthLogRepository,
    AlertRepository
)
from storage.redis_client import get_sync_redis, RedisKeys
from storage.redis_client import get_sync_redis, RedisData


logger = logging.getLogger(__name__)

# Import feature flag for JSON logging

from storage.vision_storage import ENABLE_CAMERA_JSON_LOGS
from storage.vision_storage import VisionStorage

camera_logger = VisionStorage()

from storage.camera_worker import camera_logger as json_logger


# ======================================================
# Helper Functions
# ======================================================

def create_event(
    camera_id: str,
    event_type: str,
    status: str,
    description: str,
    meta_data: dict = None,
    timestamp: str = None
) -> Dict:
    """Create a standard event structure"""
    return {
        "camera_id": camera_id,
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "event_description": description,
        "timestamp": timestamp or datetime.now().isoformat(),
        "last_checked": timestamp or datetime.now().isoformat(),
        "status": status,
        "meta_data": meta_data or {}  # FIXED: Use 'meta_data' consistently
    }


def categorize_alert(alert_text: str) -> str:
    """Categorize alert text into event type"""
    alert_lower = alert_text.lower()
    
    if any(kw in alert_lower for kw in ["displacement", "angle changed", "position changed", "orientation"]):
        return "POSITION"
    elif any(kw in alert_lower for kw in ["blur", "focus", "out of focus"]):
        return "BLUR"
    elif any(kw in alert_lower for kw in ["dark", "lighting", "overexposed", "brightness"]):
        return "LIGHTING"
    elif any(kw in alert_lower for kw in ["obstruction", "blocked", "smear", "clarity degraded"]):
        return "OBSTRUCTION"
    else:
        return "VISION_ALERT"


# ======================================================
# DUAL WRITE FUNCTIONS (Redis + DB + Optional JSON)
# ======================================================

def log_ip_check(camera_id: str, ip: str, is_reachable: bool):
    """
    Log IP check with FK violation protection
    """
    event = create_event(
        camera_id=camera_id,
        event_type="IP_CHECK",
        status="PASS" if is_reachable else "FAIL",
        description=f"IP {ip} is {'reachable' if is_reachable else 'not reachable'}",
        meta_data={"ip": ip, "reachable": is_reachable}
    )
    
    summary_updates = {
        "Ip_status": "UP" if is_reachable else "DOWN",
        "Last_ip_check": event["timestamp"]
    }
    
    # === WRITE TO DATABASE ===
    try:
        with get_sync_db() as db:
            HealthLogRepository.create_sync(db, event)
            logger.debug(f"[{camera_id}] IP check logged to DB")
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a FK violation
        if "ForeignKeyViolation" in error_msg or "foreign key constraint" in error_msg:
            logger.error(
                f"[{camera_id}] Camera not found in DB! "
                f"Camera must be registered via POST /health/camera/ first. "
                f"Error: {error_msg}"
            )
            
            # Try to auto-create camera in DB from Redis data
            try:
                from storage.redis_client import RedisKeys
                r = get_sync_redis()
                camera_data = r.hgetall(RedisKeys.camera(camera_id))
                
                if camera_data:
                    logger.info(f"[{camera_id}] Attempting auto-recovery: creating camera in DB")
                    
                    db_data = {
                        "id": camera_id,
                        "ip": camera_data.get("ip", "0.0.0.0"),
                        "port": int(camera_data.get("port", 554)),
                        "rtsp_url": camera_data.get("rtsp_url", f"rtsp://{camera_id}"),
                        "interval_ip": int(camera_data.get("interval_ip", 60)),
                        "interval_port": int(camera_data.get("interval_port", 15)),
                        "interval_vision": int(camera_data.get("interval_vision", 120))
                    }
                    
                    with get_sync_db() as recovery_db:
                        CameraRepository.create_sync(recovery_db, db_data)
                        logger.info(f"[{camera_id}] ✓ Camera auto-created in DB")
                        
                        # Retry the log write
                        HealthLogRepository.create_sync(recovery_db, event)
                        logger.info(f"[{camera_id}] ✓ IP check logged after recovery")
                        
            except Exception as recovery_error:
                logger.error(f"[{camera_id}] Auto-recovery failed: {recovery_error}")
        else:
            logger.error(f"[{camera_id}] Failed to write IP check to DB: {e}")
    
    # === WRITE TO REDIS (always works) ===
    try:
        r = get_sync_redis()
        RedisData.log_event_sync(r, camera_id, event)
        RedisData.update_summary_sync(r, camera_id, summary_updates)
    except Exception as e:
        logger.error(f"[{camera_id}] Failed to write IP check to Redis: {e}")


def log_port_check(camera_id: str, ip: str, port: int, is_accessible: bool):
    """
    Log PORT check to DB + Redis + Optional JSON
    """
    event = create_event(
        camera_id=camera_id,
        event_type="PORT_CHECK",
        status="PASS" if is_accessible else "FAIL",
        description=f"Port {port} on {ip} is {'accessible' if is_accessible else 'not accessible'}",
        meta_data={"ip": ip, "port": port, "accessible": is_accessible}
    )
    
    summary_updates = {
        "Port_status": "UP" if is_accessible else "DOWN",
        "Last_port_check": event["timestamp"]
    }
    
    # === WRITE TO DATABASE ===
    try:
        with get_sync_db() as db:
            HealthLogRepository.create_sync(db, event)
            logger.debug(f"[{camera_id}] PORT check logged to DB")
    except Exception as e:
        logger.error(f"[{camera_id}] Failed to write PORT check to DB: {e}")
    
    # === WRITE TO REDIS ===
    try:
        r = get_sync_redis()
        RedisData.log_event_sync(r, camera_id, event)
        RedisData.update_summary_sync(r, camera_id, summary_updates)
    except Exception as e:
        logger.error(f"[{camera_id}] Failed to write PORT check to Redis: {e}")
    
    # === WRITE TO JSON (optional) ===
    if ENABLE_CAMERA_JSON_LOGS:
        try:
            json_logger.log_event(camera_id, event, summary_updates)
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write PORT check to JSON: {e}")


def log_vision_check(camera_id: str, vision_result: Dict):
    """
    Log vision check to DB + Redis + Optional JSON
    Creates alerts in DB if issues detected
    """
    alerts = vision_result.get("alerts", [])
    checks = vision_result.get("checks", {})
    
    if isinstance(alerts, str):
        alerts = [alerts]
    elif not isinstance(alerts, list):
        alerts = []
    
    meta_data = {
        "similarity": checks.get("baseline_comparison", {}).get("similarity"),
        "difference_percent": checks.get("baseline_comparison", {}).get("difference_percent"),
        "brightness": checks.get("brightness", {}).get("value"),
        "blur_score": checks.get("blur", {}).get("value"),
    }
    
    summary_updates = {
        "Vision_status": vision_result["status"],
        "Alert_active": bool(alerts),
        "Last_vision_check": datetime.now().isoformat()
    }
    
    # Get Redis client
    r = get_sync_redis()
    
    # === NO ALERTS - Baseline Match ===
    if not alerts:
        event = create_event(
            camera_id=camera_id,
            event_type="BASELINE_MATCH",
            status="PASS",
            description="Captured frame matches baseline - no anomalies detected",
            meta_data=meta_data
        )
        
        # Write to DB
        try:
            with get_sync_db() as db:
                HealthLogRepository.create_sync(db, event)
                logger.debug(f"[{camera_id}] BASELINE_MATCH logged to DB")
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write vision check to DB: {e}")
        
        # Write to Redis
        try:
            RedisData.log_event_sync(r, camera_id, event)
            RedisData.update_summary_sync(r, camera_id, summary_updates)
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write vision check to Redis: {e}")
        
        # Write to JSON (optional)
        if ENABLE_CAMERA_JSON_LOGS and json_logger:
            try:
                json_logger.log_event(camera_id, event, summary_updates)
            except Exception as e:
                logger.error(f"[{camera_id}] Failed to write vision check to JSON: {e}")
        
        return
    
    # === PROCESS ALERTS ===
    for alert_text in alerts:
        if isinstance(alert_text, str):
            alert_type = categorize_alert(alert_text)
            event = create_event(
                camera_id=camera_id,
                event_type=alert_type,
                status="FAIL",
                description=alert_text,
                meta_data=meta_data
            )
        elif isinstance(alert_text, dict):
            event = create_event(
                camera_id=camera_id,
                event_type=alert_text.get("type", "UNKNOWN_ALERT"),
                status="FAIL",
                description=alert_text.get("message", "Vision alert triggered"),
                meta_data=alert_text.get("meta_data", {})
            )
            alert_type = alert_text.get("type", "UNKNOWN_ALERT")
        else:
            continue
        
        # === WRITE TO DATABASE (with Alert) ===
        try:
            with get_sync_db() as db:
                # Create health log
                health_log = HealthLogRepository.create_sync(db, event)
                
                # Create alert entry
                alert_data = {
                    "camera_id": camera_id,
                    "event_id": health_log.event_id,
                    "alert_type": alert_type,
                    "alert_description": event["event_description"]
                }
                AlertRepository.create_sync(db, alert_data)
                
                logger.debug(f"[{camera_id}] Alert '{alert_type}' logged to DB")
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write alert to DB: {e}")
        
        # === WRITE TO REDIS ===
        try:
            RedisData.log_event_sync(r, camera_id, event)
            RedisData.update_summary_sync(r, camera_id, summary_updates)
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write alert to Redis: {e}")
        
        # === WRITE TO JSON (optional) ===
        if ENABLE_CAMERA_JSON_LOGS and json_logger:
            try:
                json_logger.log_event(camera_id, event, summary_updates)
            except Exception as e:
                logger.error(f"[{camera_id}] Failed to write alert to JSON: {e}")


def log_startup(camera_id: str, camera_info: Dict):
    """
    Log camera startup to DB + Redis + Optional JSON
    """
    event = create_event(
        camera_id=camera_id,
        event_type="STARTUP",
        status="PASS",
        description="Camera monitoring worker started",
        meta_data={"camera_info": camera_info}
    )
    
    # === WRITE TO DATABASE ===
    try:
        with get_sync_db() as db:
            HealthLogRepository.create_sync(db, event)
            logger.info(f"[{camera_id}] Startup event logged to DB")
    except Exception as e:
        logger.error(f"[{camera_id}] Failed to write startup event to DB: {e}")
    
    # === WRITE TO REDIS ===
    try:
        r = get_sync_redis()
        RedisData.log_event_sync(r, camera_id, event)
    except Exception as e:
        logger.error(f"[{camera_id}] Failed to write startup event to Redis: {e}")
    
    # === WRITE TO JSON (optional) ===
    if ENABLE_CAMERA_JSON_LOGS and json_logger:
        try:
            json_logger.log_event(camera_id, event)
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to write startup event to JSON: {e}")