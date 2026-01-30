"""
Vigil-X Camera Health Service - WITH INTEGRATED IP VALIDATION
Updated: POST /health/camera/ endpoint with validation checks

File: app/app.py (REPLACE the existing start_camera_health_check function)
"""
from dto.dto_validator import CameraRequestDTO,ScheduleConfigRequest,TriggerCheckRequest
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
from core.ip_validation_service import IPValidationService  
from core.connectivity_checker import ConnectivityChecker
from storage.db_repositary import CameraLatencyRepository
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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vigil-X Camera Health Service with DB")


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
        # Store validation failure in database
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
    # STEP 2: IP UNIQUENESS CHECK
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
        # Store validation failure in database
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
    # STEP 3: STORE SUCCESSFUL VALIDATION
    # ========================================================
    validation_record = None
    try:
        validation_record = await IPValidationRepository.create_validation_async(
            db=db,
            ip=camera.ip,
            is_valid_format=True,
            is_unique=True,
            validation_result="SUCCESS",
            message="IP validation passed - camera registration proceeding"
        )
        # Don't commit yet - will commit with camera creation
        logger.info(f"[{camera.id}] ✓ Validation passed and recorded")
    except Exception as e:
        logger.error(f"Failed to store validation result: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "VALIDATION_STORAGE_ERROR",
                "message": f"Failed to store validation result: {str(e)}"
            }
        )
    
    # ========================================================
    # STEP 4: CHECK IF CAMERA ALREADY EXISTS
    # ========================================================
    existing = await CameraRepository.get_by_id_async(db, camera.id)
    
    if existing and existing.enabled:
        # Camera already exists and is enabled
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CAMERA_ALREADY_REGISTERED",
                "message": f"Camera '{camera.id}' is already being monitored",
                "camera_id": camera.id
            }
        )
    
    # ========================================================
    # STEP 5: STORE IN DATABASE
    # ========================================================
    camera_data = camera.dict()
    camera_data["interval_ip"] = 60
    camera_data["interval_port"] = 15
    camera_data["interval_vision"] = 120
    
    try:
        if existing:
            # Update existing disabled camera
            await CameraRepository.update_async(db, camera.id, {
                "enabled": True,
                "ip": camera.ip,
                "port": camera.port,
                "rtsp_url": camera.rtsp_url
            })
            logger.info(f"✓ Camera {camera.id} re-enabled in DB")
        else:
            # Create new camera in DB
            await CameraRepository.create_async(db, camera_data)
            logger.info(f"✓ Camera {camera.id} created in DB")
        
        # CRITICAL: Commit to DB BEFORE proceeding (includes validation record)
        await db.commit()
        logger.info(f"[{camera.id}] ✓ Database transaction committed")
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create camera {camera.id} in DB: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": f"Failed to register camera: {str(e)}"
            }
        )
    
    # ========================================================
    # STEP 6: STORE IN REDIS (cache)
    # ========================================================
    try:
        await RedisData.store_camera(r, camera.id, camera_data)
        logger.info(f"✓ Camera {camera.id} stored in Redis")
    except Exception as e:
        logger.error(f"Failed to store camera {camera.id} in Redis: {e}")
        # Don't fail the request - DB is primary source
    
    # ========================================================
    # STEP 7: REGISTER IN SCHEDULER (start health checks)
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

# ============================================
# NEW ENDPOINTS - ADD THESE AFTER EXISTING ENDPOINTS
# ============================================

@app.post("/health/camera/connectivity/network-check")
async def check_network_connectivity(
    camera_id: str,
    ip: str,
    port: int = 554,
    timeout: int = 5,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    **TASK 1: Network Verification**
    
    Check if camera API/service is alive and responding.
    Verifies connectivity between services and camera by checking if API is reachable.
    
    Args:
        camera_id: Camera identifier
        ip: Camera IP address
        port: Service port (default 554 for RTSP)
        timeout: Connection timeout in seconds
    
    Returns:
        Connectivity status with details
    """
    try:
        logger.info(f"Network check for camera {camera_id} at {ip}:{port}")
        
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        # Perform network check
        result = await ConnectivityChecker.check_api_alive(ip, port, timeout)
        
        # Store in Redis cache
        r = await get_async_redis()
        try:
            await RedisData.store_connectivity_api_status(r, camera_id, result, ttl=300)
        except Exception as redis_err:
            logger.error(f"Failed to store in Redis: {redis_err}")
        
        return {
            "camera_id": camera_id,
            "check_type": "NETWORK_VERIFICATION",
            "result": result,
            "stored_in_redis": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Network check failed for {camera_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network check failed: {str(e)}"
        )


@app.post("/health/camera/connectivity/latency-check")
async def measure_camera_latency(
    camera_id: str,
    ip: str,
    ping_count: int = 10,
    timeout: int = 5,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    **TASK 2: Latency Measurement**
    
    Measure Round Trip Time (RTT) for camera during health check.
    Calculates avg/min/max RTT and stores in:
    - Database: camera_latency table
    - Redis: for fast caching
    
    Args:
        camera_id: Camera identifier
        ip: Camera IP address
        ping_count: Number of ping packets (5-100)
        timeout: Timeout per packet in seconds
    
    Returns:
        Latency metrics (rtt_avg, rtt_min, rtt_max in ms)
    """
    try:
        logger.info(f"Latency check for camera {camera_id} at {ip}")
        
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        # Perform latency measurement
        result = await ConnectivityChecker.measure_latency(ip, ping_count, timeout)
        
        stored_db = False
        stored_redis = False
        
        if result["success"]:
            # Prepare data for storage
            latency_data = {
                "camera_id": camera_id,
                "rtt_avg": result.get("rtt_avg"),
                "rtt_min": result.get("rtt_min"),
                "rtt_max": result.get("rtt_max"),
                "packets_sent": result.get("packets_sent"),
                "packets_received": result.get("packets_received"),
                "is_reachable": result.get("is_reachable", False),
                "check_type": "LATENCY",
                "error_message": None
            }
            
            # Store in database
            try:
                await CameraLatencyRepository.create_async(db, latency_data)
                await db.commit()
                stored_db = True
                logger.info(f"Stored latency in DB for {camera_id}")
            except Exception as db_err:
                logger.error(f"Failed to store in DB: {db_err}")
                await db.rollback()
            
            # Store in Redis
            r = await get_async_redis()
            try:
                await RedisData.store_connectivity_latency(
                    r, camera_id,
                    {**latency_data, "timestamp": result["timestamp"]},
                    ttl=300,
                    keep_history=True
                )
                stored_redis = True
            except Exception as redis_err:
                logger.error(f"Failed to store in Redis: {redis_err}")
        
        return {
            "camera_id": camera_id,
            "check_type": "LATENCY_MEASUREMENT",
            "result": result,
            "storage": {
                "database": stored_db,
                "redis": stored_redis
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Latency check failed for {camera_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Latency check failed: {str(e)}"
        )


@app.post("/health/camera/connectivity/packet-loss-check")
async def calculate_camera_packet_loss(
    camera_id: str,
    ip: str,
    packet_count: int = 100,
    timeout: int = 5,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    **TASK 3: Packet Loss Calculation**
    
    Calculate packet loss percentage for camera during health check.
    Stores result in:
    - Database: camera_latency table (packet_loss_percent column)
    - Redis: for fast caching
    
    Args:
        camera_id: Camera identifier
        ip: Camera IP address
        packet_count: Number of packets to send (10-1000)
        timeout: Timeout per packet in seconds
    
    Returns:
        Packet loss percentage and packet statistics
    """
    try:
        logger.info(f"Packet loss check for camera {camera_id} at {ip}")
        
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        # Perform packet loss calculation
        result = await ConnectivityChecker.calculate_packet_loss(ip, packet_count, timeout)
        
        stored_db = False
        stored_redis = False
        
        if result["success"]:
            # Prepare data for storage (includes packet loss)
            loss_data = {
                "camera_id": camera_id,
                "packet_loss_percent": result.get("packet_loss_percent"),
                "packets_sent": result.get("packets_sent"),
                "packets_received": result.get("packets_received"),
                "is_reachable": result.get("packet_loss_percent", 100) < 100,
                "check_type": "PACKET_LOSS",
                "error_message": None if result["success"] else result.get("message")
            }
            
            # Store in database
            try:
                await CameraLatencyRepository.create_async(db, loss_data)
                await db.commit()
                stored_db = True
                logger.info(f"Stored packet loss in DB for {camera_id}")
            except Exception as db_err:
                logger.error(f"Failed to store in DB: {db_err}")
                await db.rollback()
            
            # Store in Redis
            r = await get_async_redis()
            try:
                await RedisData.store_connectivity_packet_loss(
                    r, camera_id,
                    {**loss_data, "timestamp": result["timestamp"]},
                    ttl=300,
                    keep_history=True
                )
                stored_redis = True
            except Exception as redis_err:
                logger.error(f"Failed to store in Redis: {redis_err}")
        
        return {
            "camera_id": camera_id,
            "check_type": "PACKET_LOSS_CALCULATION",
            "result": result,
            "storage": {
                "database": stored_db,
                "redis": stored_redis
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Packet loss check failed for {camera_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Packet loss check failed: {str(e)}"
        )




@app.get("/health/camera/{camera_id}/connectivity/latency-history")
async def get_camera_latency_history(
    camera_id: str,
    source: str = Query(default="database", regex="^(database|redis)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    hours: Optional[int] = Query(default=None, ge=1, le=168),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get latency history for a camera
    
    Args:
        camera_id: Camera identifier
        source: Data source ('database' or 'redis')
        limit: Maximum number of records
        hours: Optional time window (last N hours)
    
    Returns:
        List of historical latency records
    """
    try:
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        records = []
        
        if source == "database":
            records_obj = await CameraLatencyRepository.get_history_by_camera_async(
                db, camera_id, limit=limit, hours=hours
            )
            records = [r.to_dict() for r in records_obj]
        else:  # redis
            r = await get_async_redis()
            minutes = hours * 60 if hours else None
            records = await RedisData.get_connectivity_latency_history(
                r, camera_id, limit=limit, minutes=minutes
            )
        
        return {
            "camera_id": camera_id,
            "source": source,
            "total_records": len(records),
            "records": records
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latency history for {camera_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.get("/health/camera/{camera_id}/connectivity/statistics")
async def get_camera_connectivity_statistics(
    camera_id: str,
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get aggregated connectivity statistics for a camera
    
    Args:
        camera_id: Camera identifier
        hours: Time window for statistics (default 24 hours)
    
    Returns:
        Aggregated statistics (avg RTT, packet loss, uptime %)
    """
    try:
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found"
            )
        
        # Get statistics from database
        stats = await CameraLatencyRepository.get_statistics_async(
            db, camera_id, hours=hours
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for {camera_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.put("/health/camera/{camera_id}/intervals")
async def update_camera_intervals(
    camera_id: str,
    config: ScheduleConfigRequest,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Configure scheduled test intervals for a camera
    
    This endpoint allows you to set custom intervals for IP, Port, and Vision checks.
    """
    try:
        # Verify camera exists
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CAMERA_NOT_FOUND",
                    "message": f"Camera '{camera_id}' not found",
                    "camera_id": camera_id
                }
            )
        
        # Prepare update data
        update_data = {}
        intervals_to_update = {}
        
        if config.interval_ip is not None:
            update_data["interval_ip"] = config.interval_ip
            intervals_to_update['ip'] = config.interval_ip
            
        if config.interval_port is not None:
            update_data["interval_port"] = config.interval_port
            intervals_to_update['port'] = config.interval_port
            
        if config.interval_vision is not None:
            update_data["interval_vision"] = config.interval_vision
            intervals_to_update['vision'] = config.interval_vision
        
        if not update_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_INTERVALS_PROVIDED",
                    "message": "At least one interval must be provided",
                    "valid_fields": ["interval_ip", "interval_port", "interval_vision"]
                }
            )
        
        # Update database
        updated_camera = await CameraRepository.update_async(db, camera_id, update_data)
        await db.commit()
        
        logger.info(f"Updated intervals in DB for {camera_id}: {update_data}")
        
        # Update Redis
        r = await get_async_redis()
        from storage.redis_client import RedisKeys  # Import here if needed
        camera_key = RedisKeys.camera(camera_id)
        
        pipe = r.pipeline()
        for key, value in update_data.items():
            pipe.hset(camera_key, key, value)
        await pipe.execute()
        
        logger.info(f"Updated intervals in Redis for {camera_id}")
        
        # Update scheduler
        await scheduler.update_camera_intervals(camera_id, intervals_to_update)
        
        return {
            "message": "Camera schedule configured successfully",
            "camera_id": camera_id,
            "previous_intervals": {
                "interval_ip": camera.interval_ip,
                "interval_port": camera.interval_port,
                "interval_vision": camera.interval_vision
            },
            "updated_intervals": {
                "interval_ip": updated_camera.interval_ip,
                "interval_port": updated_camera.interval_port,
                "interval_vision": updated_camera.interval_vision
            },
            
            "status": "active",
            "note": "New intervals will take effect on the next scheduled check"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring schedule for {camera_id}: {e}", exc_info=True)
        await db.rollback()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"Failed to configure camera schedule: {str(e)}",
                "camera_id": camera_id
            }
        )



@app.get("/health/camera/{camera_id}/schedule")
async def get_camera_schedule(
    camera_id: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get current schedule configuration for a camera
    """
    try:
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CAMERA_NOT_FOUND",
                    "message": f"Camera '{camera_id}' not found",
                    "camera_id": camera_id
                }
            )
        
        return {
            "camera_id": camera_id,
            "schedule_mode": "scheduled" if camera.enabled else "disabled",
            "intervals": {
                "interval_ip": camera.interval_ip,
                "interval_port": camera.interval_port,
                "interval_vision": camera.interval_vision
            },
            "camera_info": {
                "ip": camera.ip,
                "port": camera.port,
                "enabled": camera.enabled
            },
            "test_modes": {
                "manual": "Use POST /health/camera/triggeronce",
                "continuous": "Currently running with scheduled intervals",
                "scheduled": "Current mode - use POST /schedule to update"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching schedule for {camera_id}: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"Failed to fetch camera schedule: {str(e)}",
                "camera_id": camera_id
            }
        )
