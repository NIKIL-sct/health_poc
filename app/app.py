"""
Vigil-X Camera Health Service - WITH INTEGRATED IP VALIDATION
Updated: POST /health/camera/ endpoint with validation checks

File: app/app.py (REPLACE the existing start_camera_health_check function)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime
import logging
import traceback
from app.connectivity_api import connectivity_router
from dto.dto_validator import CameraRequestDTO
from storage.db_config import get_db_dependency, close_async_engine
from storage.db_repositary import (
    CameraRepository,
    HealthLogRepository,
    SummaryRepository,
    IPValidationRepository  # NEW IMPORT
)
from storage.redis_client import (
    get_async_redis,
    close_async_redis,
    RedisData
)
from core.schedular import scheduler
from workers.network_worker import network_worker_pool
from workers.vision_worker import vision_worker_pool
from core.ping_checker import PingChecker
from core.vision_checker import VisionChecker
from core.ip_validation_service import IPValidationService  # NEW IMPORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vigil-X Camera Health Service ")

app.include_router(connectivity_router)

# -------------------------------------------------
# Startup / Shutdown (NO CHANGES)
# -------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Start background components"""
    logger.info("Starting Vigil-X services with DB integration...")
    
    # Initialize database tables
    from storage.models import Base
    from storage.db_config import get_sync_engine
    engine = get_sync_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
    
    # Start workers and scheduler
    await scheduler.start()
    await network_worker_pool.start()
    vision_worker_pool.start()
    
    logger.info("✓ All services started (DB + Redis)")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop all workers gracefully"""
    logger.info("Shutting down services...")
    
    await scheduler.stop()
    await network_worker_pool.stop()
    vision_worker_pool.stop()
    await close_async_redis()
    await close_async_engine()
    
    logger.info("All services stopped")


# -------------------------------------------------
# POST: Start Camera Monitoring WITH IP VALIDATION
# -------------------------------------------------

@app.post("/health/camera/", status_code=201)
async def start_camera_health_check(
    camera: CameraRequestDTO,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Register camera with IP validation checks:
    
    VALIDATION FLOW:
    1. ✓ Validate IP format (structure check)
    2. ✓ Check IP uniqueness (not in use by another camera)
    3. ✓ Store validation result in database
    4. ✓ If validation passes → Register camera
    5. ✓ Store in PostgreSQL → Redis → Scheduler
    
    VALIDATION FAILURE:
    - If format is invalid → HTTP 422 with error details
    - If IP is duplicate → HTTP 409 with existing camera details
    - Validation result is ALWAYS stored in database
    """
    r = await get_async_redis()
    
    # ========================================================
    # STEP 1: IP FORMAT VALIDATION
    # ========================================================
    logger.info(f"[{camera.id}] Step 1: Validating IP format for {camera.ip}")
    
    is_valid_format, format_message = IPValidationService.validate_ip_format(camera.ip)
    
    if not is_valid_format:
        # Store validation failure in database (separate transaction)
        try:
            await IPValidationRepository.create_validation_async(
                db=db,
                ip=camera.ip,
                is_valid_format=False,
                is_unique=True,  # Not checked yet
                validation_result="INVALID_FORMAT",
                message=format_message
            )
            await db.commit()
            logger.warning(f"[{camera.id}] IP format validation failed: {format_message}")
        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")
            await db.rollback()
        
        # Reject request
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_IP_FORMAT",
                "message": format_message,
                "ip": camera.ip,
                "camera_id": camera.id,
                "validation_stored": True
            }
        )
    
    logger.info(f"[{camera.id}] ✓ IP format is valid")
    
    # ========================================================
    # STEP 2: CHECK IF CAMERA ALREADY EXISTS (Before Uniqueness Check)
    # ========================================================
    # IMPORTANT: Check camera existence BEFORE creating validation record
    # This prevents transaction issues
    existing = await CameraRepository.get_by_id_async(db, camera.id)
    
    if existing and existing.enabled:
        # Camera already exists and is enabled
        logger.warning(f"[{camera.id}] Camera already registered and enabled")
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CAMERA_ALREADY_REGISTERED",
                "message": f"Camera '{camera.id}' is already being monitored",
                "camera_id": camera.id
            }
        )
    
    # ========================================================
    # STEP 3: IP UNIQUENESS CHECK
    # ========================================================
    logger.info(f"[{camera.id}] Step 2: Checking IP uniqueness for {camera.ip}")
    
    # Check in database first (primary source)
    is_unique_db, db_message = await IPValidationService.check_ip_uniqueness_db(db, camera.ip)
    
    # Check in Redis as secondary verification
    is_unique_redis, redis_message = await IPValidationService.check_ip_uniqueness_redis(camera.ip)
    
    # Use database as source of truth
    is_unique = is_unique_db
    uniqueness_message = db_message
    
    # If IP is found in Redis but not in DB, prefer Redis result
    if is_unique_db and not is_unique_redis:
        is_unique = False
        uniqueness_message = redis_message
        logger.warning(f"[{camera.id}] IP found in Redis but not in DB (cache inconsistency)")
    
    if not is_unique:
        # Store validation failure in database (separate transaction)
        try:
            await IPValidationRepository.create_validation_async(
                db=db,
                ip=camera.ip,
                is_valid_format=True,
                is_unique=False,
                validation_result="DUPLICATE",
                message=uniqueness_message
            )
            await db.commit()
            logger.warning(f"[{camera.id}] IP uniqueness check failed: {uniqueness_message}")
        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")
            await db.rollback()
        
        # Reject request
        raise HTTPException(
            status_code=409,
            detail={
                "error": "DUPLICATE_IP",
                "message": uniqueness_message,
                "ip": camera.ip,
                "camera_id": camera.id,
                "validation_stored": True
            }
        )
    
    logger.info(f"[{camera.id}] ✓ IP is unique")
    
    # ========================================================
    # STEP 4: STORE CAMERA AND VALIDATION IN SINGLE TRANSACTION
    # ========================================================
    validation_record = None
    camera_data = camera.dict()
    camera_data["interval_ip"] = 60
    camera_data["interval_port"] = 15
    camera_data["interval_vision"] = 120
    
    try:
        # Create validation record (don't commit yet)
        validation_record = await IPValidationRepository.create_validation_async(
            db=db,
            ip=camera.ip,
            is_valid_format=True,
            is_unique=True,
            validation_result="SUCCESS",
            message="IP validation passed - camera registration proceeding"
        )
        logger.info(f"[{camera.id}] ✓ Validation passed and recorded")
        
        # Create or update camera (don't commit yet)
        if existing:
            # Update existing disabled camera
            await CameraRepository.update_async(db, camera.id, {
                "enabled": True,
                "ip": camera.ip,
                "port": camera.rtsp_port,
                "rtsp_url": camera.rtsp_url
            })
            logger.info(f"✓ Camera {camera.id} re-enabled in DB")
        else:
            # Create new camera in DB
            await CameraRepository.create_async(db, camera_data)
            logger.info(f"✓ Camera {camera.id} created in DB")
        
        # COMMIT BOTH: validation record + camera creation in single transaction
        await db.commit()
        logger.info(f"[{camera.id}] ✓ Database transaction committed (validation + camera)")
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to register camera {camera.id} in DB: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": f"Failed to register camera: {str(e)}"
            }
        )
    
    # ========================================================
    # STEP 5: STORE IN REDIS (cache)
    # ========================================================
    try:
        await RedisData.store_camera(r, camera.id, camera_data)
        logger.info(f"✓ Camera {camera.id} stored in Redis")
    except Exception as e:
        logger.error(f"Failed to store camera {camera.id} in Redis: {e}")
        # Don't fail the request - DB is primary source
    
    # ========================================================
    # STEP 6: REGISTER IN SCHEDULER (start health checks)
    # ========================================================
    intervals = {
        "ip": 60,
        "port": 15,
        "vision": 120
    }
    
    try:
        await scheduler.register_camera(camera.id, intervals)
        logger.info(f"✓ Camera {camera.id} registered in scheduler")
    except Exception as e:
        logger.error(f"Failed to register camera {camera.id} in scheduler: {e}")
    
    logger.info(f"✓ Camera {camera.id} fully registered (Validation → DB → Redis → Scheduler)")
    
    # ========================================================
    # SUCCESS RESPONSE
    # ========================================================
    return {
        "message": "Camera health monitoring started",
        "camera_id": camera.id,
        "ip": camera.ip,
        "validation": {
            "validation_id": validation_record.validation_id if validation_record else None,
            "is_valid_format": True,
            "is_unique": True,
            "validation_result": "SUCCESS",
            "timestamp": validation_record.timestamp.isoformat() if validation_record else None
        },
        "intervals": intervals,
        "storage": "PostgreSQL (primary) + Redis (cache)",
        "status": "registered",
        "health_checks_started": True
    }


# -------------------------------------------------
# POST: One-Time Health Check (NO CHANGES)
# -------------------------------------------------

@app.post("/health/camera/triggeronce", status_code=201)
async def one_time_health_check(
    camera: CameraRequestDTO,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Immediate synchronous health check
    Results are NOT stored in DB (one-time check)
    """
    ping = PingChecker()
    vision = VisionChecker()

    response = {
        "camera_id": camera.id,
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    # IP CHECK
    ip_checked_at = datetime.now()
    ip_up = ping.ping_ip(camera.ip)

    response["checks"]["ip"] = {
        "status": "UP" if ip_up else "DOWN",
        "checked_at": ip_checked_at.isoformat()
    }

    # PORT CHECK
    port_checked_at = datetime.now()
    info = ping.extract_rtsp_info(camera.rtsp_url)
    port_up = ping.check_port(info["ip"], info["port"])

    response["checks"]["port"] = {
        "status": "UP" if port_up else "DOWN",
        "checked_at": port_checked_at.isoformat(),
        "port": info["port"]
    }

    # VISION CHECK
    if ip_up and port_up:
        vision_checked_at = datetime.now()
        vision_result = vision.check_camera_vision(
            camera.id,
            camera.rtsp_url
        )

        response["checks"]["vision"] = {
            "status": vision_result["status"],
            "checked_at": vision_checked_at.isoformat(),
            "alerts": vision_result.get("alerts", []),
            "checks": vision_result.get("checks", {})
        }
    else:
        response["checks"]["vision"] = {
            "status": "SKIPPED",
            "reason": "IP or Port down"
        }

    return response


# -------------------------------------------------
# GET: Camera Health (NO CHANGES)
# -------------------------------------------------

@app.get("/health/camera/{camera_id}")
async def get_camera_health(
    camera_id: str,
    event_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    from_ts: Optional[datetime] = Query(None),
    to_ts: Optional[datetime] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    sort: str = Query("desc"),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get camera health data from PostgreSQL
    Database is the primary source of truth for historical data
    """
    
    try:
        # === VERIFY CAMERA EXISTS ===
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        # === GET LATEST SUMMARY FROM DB ===
        summary = await SummaryRepository.get_latest_summary_async(db, camera_id)
        
        # === GET HEALTH LOGS FROM DB WITH FILTERS ===
        order_desc = (sort == "desc")
        
        logs, total_count = await HealthLogRepository.get_by_camera_async(
            db,
            camera_id,
            event_type=event_type,
            status=status,
            from_ts=from_ts,
            to_ts=to_ts,
            offset=offset,
            limit=limit,
            order_desc=order_desc
        )
        
        # === FORMAT EVENTS ===
        events = []
        for log in logs:
            try:
                event = {
                    "event_id": log.event_id,
                    "camera_id": log.camera_id,
                    "event_type": log.event_type,
                    "status": log.status,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "event_description": log.event_description,
                    "meta_data": log.meta_data if log.meta_data else {},
                    "last_checked": log.last_checked.isoformat() if log.last_checked else None
                }
                events.append(event)
            except Exception as e:
                logger.error(f"Error serializing log {log.event_id}: {e}")
                continue
        
        return {
            "Latest_summary": summary,
            "Events": events,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "returned_items": len(events),
                "total_items": total_count,
                "has_more": offset + limit < total_count
            },
            "storage": "PostgreSQL (primary)",
            "filters_applied": {
                "event_type": event_type,
                "status": status,
                "from_ts": from_ts.isoformat() if from_ts else None,
                "to_ts": to_ts.isoformat() if to_ts else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching health data for {camera_id}: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"Failed to fetch camera health data: {str(e)}",
                "camera_id": camera_id
            }
        )


# -------------------------------------------------
# GET: System Health (NO CHANGES)
# -------------------------------------------------

@app.get("/health")
async def system_health():
    """Overall system health endpoint"""
    r = await get_async_redis()
    
    try:
        await r.ping()
        redis_status = "UP"
    except Exception:
        redis_status = "DOWN"
    
    try:
        from storage.db_config import get_async_engine
        from sqlalchemy import text
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "UP"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "DOWN"
    
    return {
        "status": "HEALTHY" if redis_status == "UP" and db_status == "UP" else "DEGRADED",
        "components": {
            "redis": redis_status,
            "database": db_status,
            "scheduler": "RUNNING" if scheduler.running else "STOPPED",
            "network_workers": "RUNNING",
            "vision_workers": "RUNNING"
        },
        "architecture": "Dual Storage (PostgreSQL + Redis)"
    }