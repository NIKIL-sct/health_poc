"""
Database Repository Layer
Provides clean abstraction for all database operations
Supports both sync (workers) and async (FastAPI) contexts
FIXED: Consistent use of 'meta_data' and proper timestamp handling
"""

from sqlalchemy import select, and_, or_, desc, asc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging

from storage.models import IPValidation, Camera


from storage.models import Camera, HealthLog, Alert

logger = logging.getLogger(__name__)


# ============================================
# CAMERA REPOSITORY
# ============================================

class CameraRepository:
    """Repository for Camera operations"""
    
    # ===== ASYNC METHODS (FastAPI) =====
    
    @staticmethod
    async def create_async(db: AsyncSession, camera_data: Dict) -> Camera:
        """Create new camera"""
        camera = Camera(
            camera_id=camera_data["id"],
            ip=camera_data["ip"],
            port=camera_data["port"],
            rtsp_url=camera_data["rtsp_url"],
            enabled=True,
            interval_ip=camera_data.get("interval_ip", 60),
            interval_port=camera_data.get("interval_port", 15),
            interval_vision=camera_data.get("interval_vision", 120)
        )
        
        db.add(camera)
        await db.flush()
        logger.info(f"Camera {camera.camera_id} created in DB")
        return camera
    
    @staticmethod
    async def get_by_id_async(db: AsyncSession, camera_id: str) -> Optional[Camera]:
        """Get camera by ID"""
        result = await db.execute(
            select(Camera).where(Camera.camera_id == camera_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_async(db: AsyncSession, camera_id: str, updates: Dict) -> Optional[Camera]:
        """Update camera"""
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            return None
        
        for key, value in updates.items():
            if hasattr(camera, key):
                setattr(camera, key, value)
        
        await db.flush()
        return camera
    
    @staticmethod
    async def delete_async(db: AsyncSession, camera_id: str) -> bool:
        """Delete camera (cascade deletes logs and alerts)"""
        camera = await CameraRepository.get_by_id_async(db, camera_id)
        if not camera:
            return False
        
        await db.delete(camera)
        await db.flush()
        logger.info(f"Camera {camera_id} deleted from DB")
        return True
    
    @staticmethod
    async def list_all_async(db: AsyncSession, enabled_only: bool = False) -> List[Camera]:
        """List all cameras"""
        query = select(Camera)
        
        if enabled_only:
            query = query.where(Camera.enabled == True)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    # ===== SYNC METHODS (Workers) =====
    
    @staticmethod
    def create_sync(db: Session, camera_data: Dict) -> Camera:
        camera = Camera(
            camera_id=camera_data["id"],
            ip=camera_data["ip"],
            port=camera_data["port"],
            rtsp_url=camera_data["rtsp_url"],
            enabled=True,
            interval_ip=camera_data.get("interval_ip", 60),
            interval_port=camera_data.get("interval_port", 15),
            interval_vision=camera_data.get("interval_vision", 120)
        )

        db.add(camera)
        db.commit()              
        db.refresh(camera)
        logger.info(f"Camera {camera.camera_id} created in DB")
        return camera


# ============================================
# HEALTH LOG REPOSITORY
# ============================================

class HealthLogRepository:
    """Repository for HealthLog operations"""
    
    # ===== ASYNC METHODS =====
    
    @staticmethod
    async def create_async(db: AsyncSession, log_data: Dict) -> HealthLog:
        """Create health log entry"""
        # Parse timestamp if it's a string
        timestamp = log_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        # Parse last_checked if it's a string
        last_checked = log_data.get("last_checked")
        if isinstance(last_checked, str):
            last_checked = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
        elif last_checked is None:
            last_checked = datetime.utcnow()
        
        health_log = HealthLog(
            event_id=log_data.get("event_id"),
            camera_id=log_data["camera_id"],
            status=log_data["status"],
            event_type=log_data["event_type"],
            timestamp=timestamp,
            event_description=log_data.get("event_description", ""),
            meta_data=log_data.get("meta_data", {}),
            last_checked=last_checked
        )
        
        db.add(health_log)
        await db.flush()
        return health_log
    
    @staticmethod
    async def get_by_camera_async(
        db: AsyncSession,
        camera_id: str,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        from_ts: Optional[datetime] = None,
        to_ts: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100,
        order_desc: bool = True
    ) -> tuple[List[HealthLog], int]:
        """
        Get health logs for a camera with filters and pagination
        Returns: (logs, total_count)
        """
        # Build base query
        query = select(HealthLog).where(HealthLog.camera_id == camera_id)
        
        # Apply filters
        if event_type:
            query = query.where(HealthLog.event_type == event_type)
        
        if status:
            query = query.where(HealthLog.status == status)
        
        if from_ts:
            query = query.where(HealthLog.timestamp >= from_ts)
        
        if to_ts:
            query = query.where(HealthLog.timestamp <= to_ts)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total_count = total_result.scalar()
        
        # Apply ordering
        if order_desc:
            query = query.order_by(desc(HealthLog.timestamp))
        else:
            query = query.order_by(asc(HealthLog.timestamp))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute
        result = await db.execute(query)
        logs = list(result.scalars().all())
        
        return logs, total_count
    
    @staticmethod
    async def get_latest_by_type_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str
    ) -> Optional[HealthLog]:
        """Get latest log entry for a specific event type"""
        query = select(HealthLog).where(
            and_(
                HealthLog.camera_id == camera_id,
                HealthLog.event_type == event_type
            )
        ).order_by(desc(HealthLog.timestamp)).limit(1)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    # ===== SYNC METHODS =====
    
    @staticmethod
    def create_sync(db: Session, log_data: Dict) -> HealthLog:
        """Create health log entry (sync)"""
        # Parse timestamp if it's a string
        timestamp = log_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        # Parse last_checked if it's a string
        last_checked = log_data.get("last_checked")
        if isinstance(last_checked, str):
            last_checked = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
        elif last_checked is None:
            last_checked = datetime.utcnow()
        
        health_log = HealthLog(
            event_id=log_data.get("event_id"),
            camera_id=log_data["camera_id"],
            status=log_data["status"],
            event_type=log_data["event_type"],
            timestamp=timestamp,
            event_description=log_data.get("event_description", ""),
            meta_data=log_data.get("meta_data", {}),
            last_checked=last_checked
        )
        
        db.add(health_log)
        db.commit()                 
        db.refresh(health_log)
        return health_log



# ============================================
# ALERT REPOSITORY
# ============================================

class AlertRepository:
    """Repository for Alert operations"""
    
    # ===== ASYNC METHODS =====
    
    @staticmethod
    async def create_async(db: AsyncSession, alert_data: Dict) -> Alert:
        """Create alert"""
        alert = Alert(
            alert_id=alert_data.get("alert_id"),
            camera_id=alert_data["camera_id"],
            event_id=alert_data["event_id"],
            alert_type=alert_data["alert_type"],
            alert_description=alert_data["alert_description"],
            timestamp=alert_data.get("timestamp", datetime.utcnow()),
            resolved=False
        )
        
        db.add(alert)
        await db.flush()
        return alert
    
    @staticmethod
    async def get_active_alerts_async(
        db: AsyncSession,
        camera_id: Optional[str] = None
    ) -> List[Alert]:
        """Get all unresolved alerts"""
        query = select(Alert).where(Alert.resolved == False)
        
        if camera_id:
            query = query.where(Alert.camera_id == camera_id)
        
        query = query.order_by(desc(Alert.timestamp))
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def resolve_alert_async(
        db: AsyncSession,
        alert_id: str
    ) -> Optional[Alert]:
        """Mark alert as resolved"""
        result = await db.execute(
            select(Alert).where(Alert.alert_id == alert_id)
        )
        alert = result.scalar_one_or_none()
        
        if alert:
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            await db.flush()
        
        return alert
    
    # ===== SYNC METHODS =====
    
    @staticmethod
    def create_sync(db: Session, alert_data: Dict) -> Alert:
        """Create alert (sync)"""
        alert = Alert(
            alert_id=alert_data.get("alert_id"),
            camera_id=alert_data["camera_id"],
            event_id=alert_data["event_id"],
            alert_type=alert_data["alert_type"],
            alert_description=alert_data["alert_description"],
            timestamp=alert_data.get("timestamp", datetime.utcnow()),
            resolved=False
        )
        
        db.add(alert)
        db.commit()                 
        db.refresh(alert)
        return alert


class IPValidationRepository:
    """Repository for IP Validation operations"""
    
    @staticmethod
    async def create_validation_async(
        db: AsyncSession,
        ip: str,
        is_valid_format: bool,
        is_unique: bool,
        validation_result: str,
        message: str
    ) -> IPValidation:
        """
        Create IP validation record
        
        Args:
            db: Database session
            ip: IP address being validated
            is_valid_format: Whether IP format is valid
            is_unique: Whether IP is unique (not in use)
            validation_result: Overall result (SUCCESS, INVALID_FORMAT, DUPLICATE)
            message: Descriptive message
        
        Returns:
            Created IPValidation object
        """
        validation = IPValidation(
            ip=ip,
            is_valid_format=is_valid_format,
            is_unique=is_unique,
            validation_result=validation_result,
            message=message,
            timestamp=datetime.utcnow()
        )
        
        db.add(validation)
        await db.flush()
        logger.info(f"IP validation record created for {ip}: {validation_result}")
        return validation
    
    @staticmethod
    async def get_by_id_async(
        db: AsyncSession,
        validation_id: str
    ) -> Optional[IPValidation]:
        """Get validation record by ID"""
        result = await db.execute(
            select(IPValidation).where(IPValidation.validation_id == validation_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_validations_by_ip_async(
        db: AsyncSession,
        ip: str,
        limit: int = 10
    ) -> List[IPValidation]:
        """Get validation history for a specific IP"""
        query = select(IPValidation).where(
            IPValidation.ip == ip
        ).order_by(IPValidation.timestamp.desc()).limit(limit)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def check_ip_uniqueness_async(
        db: AsyncSession,
        ip: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if IP is unique (not used by any camera)
        
        Returns:
            Tuple of (is_unique: bool, camera_id: Optional[str])
            - If unique: (True, None)
            - If in use: (False, camera_id)
        """
        result = await db.execute(
            select(Camera).where(Camera.ip == ip)
        )
        camera = result.scalar_one_or_none()
        
        if camera:
            return False, camera.camera_id
        return True, None
    
    @staticmethod
    async def get_latest_validation_for_ip_async(
        db: AsyncSession,
        ip: str
    ) -> Optional[IPValidation]:
        """Get the most recent validation record for an IP"""
        query = select(IPValidation).where(
            IPValidation.ip == ip
        ).order_by(IPValidation.timestamp.desc()).limit(1)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()

# ============================================
# SUMMARY HELPERS
# ============================================

class SummaryRepository:
    """Helper methods for generating summaries from DB"""
    
    @staticmethod
    async def get_latest_summary_async(db: AsyncSession, camera_id: str) -> Dict:
        """
        Generate latest summary from most recent health checks
        Mimics Redis summary structure
        """
        summary = {
            "Ip_status": "UNKNOWN",
            "Port_status": "UNKNOWN",
            "Vision_status": "UNKNOWN",
            "Alert_active": False,
            "Last_ip_check": None,
            "Last_port_check": None,
            "Last_vision_check": None
        }
        
        # Get latest IP check
        ip_log = await HealthLogRepository.get_latest_by_type_async(db, camera_id, "IP_CHECK")
        if ip_log:
            summary["Ip_status"] = ip_log.status
            summary["Last_ip_check"] = ip_log.timestamp.isoformat()
        
        # Get latest PORT check
        port_log = await HealthLogRepository.get_latest_by_type_async(db, camera_id, "PORT_CHECK")
        if port_log:
            summary["Port_status"] = port_log.status
            summary["Last_port_check"] = port_log.timestamp.isoformat()
        
        # Get latest VISION check (any vision-related type)
        vision_query = select(HealthLog).where(
            and_(
                HealthLog.camera_id == camera_id,
                or_(
                    HealthLog.event_type == "BASELINE_MATCH",
                    HealthLog.event_type == "VISION_ALERT",
                    HealthLog.event_type == "OBSTRUCTION",
                    HealthLog.event_type == "BLUR",
                    HealthLog.event_type == "LIGHTING",
                    HealthLog.event_type == "POSITION"
                )
            )
        ).order_by(desc(HealthLog.timestamp)).limit(1)
        
        result = await db.execute(vision_query)
        vision_log = result.scalar_one_or_none()
        
        if vision_log:
            summary["Vision_status"] = vision_log.status
            summary["Last_vision_check"] = vision_log.timestamp.isoformat()
        
        # Check for active alerts
        active_alerts = await AlertRepository.get_active_alerts_async(db, camera_id)
        summary["Alert_active"] = len(active_alerts) > 0
        
        return summary
    
# ============================================
# CAMERA LATENCY REPOSITORY (NEW - CONNECTIVITY CHECKS)
# ============================================

class CameraLatencyRepository:
    """Repository for CameraLatency operations - Connectivity Checks"""
    
    # ===== ASYNC METHODS =====
    
    @staticmethod
    async def create_async(db: AsyncSession, latency_data: Dict):
        """
        Create a new latency record
        
        Args:
            db: Database session
            latency_data: Dictionary containing latency metrics
                - camera_id (required)
                - rtt_avg, rtt_min, rtt_max (optional)
                - packet_loss_percent, packets_sent, packets_received (optional)
                - is_reachable (optional, default True)
                - check_type (optional, default 'PING')
                - error_message (optional)
        
        Returns:
            Created CameraLatency object
        """
        from storage.models import CameraLatency
        
        latency = CameraLatency(
            camera_id=latency_data["camera_id"],
            rtt_avg=latency_data.get("rtt_avg"),
            rtt_min=latency_data.get("rtt_min"),
            rtt_max=latency_data.get("rtt_max"),
            packet_loss_percent=latency_data.get("packet_loss_percent"),
            packets_sent=latency_data.get("packets_sent"),
            packets_received=latency_data.get("packets_received"),
            is_reachable=latency_data.get("is_reachable", True),
            check_type=latency_data.get("check_type", "PING"),
            error_message=latency_data.get("error_message"),
            timestamp=latency_data.get("timestamp", datetime.utcnow())
        )
        
        db.add(latency)
        await db.flush()
        logger.info(f"Latency record created for camera {latency_data['camera_id']}")
        return latency
    
    @staticmethod
    async def get_by_id_async(db: AsyncSession, latency_id: str):
        """Get latency record by ID"""
        from storage.models import CameraLatency
        
        result = await db.execute(
            select(CameraLatency).where(CameraLatency.latency_id == latency_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_latest_by_camera_async(db: AsyncSession, camera_id: str):
        """Get the most recent latency record for a camera"""
        from storage.models import CameraLatency
        
        query = select(CameraLatency).where(
            CameraLatency.camera_id == camera_id
        ).order_by(desc(CameraLatency.timestamp)).limit(1)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_history_by_camera_async(
        db: AsyncSession,
        camera_id: str,
        limit: int = 100,
        hours: Optional[int] = None
    ) -> List:
        """
        Get latency history for a camera
        
        Args:
            db: Database session
            camera_id: Camera identifier
            limit: Maximum number of records to return
            hours: Optional time window (only get records from last N hours)
        
        Returns:
            List of CameraLatency objects
        """
        from storage.models import CameraLatency
        from datetime import timedelta
        
        query = select(CameraLatency).where(
            CameraLatency.camera_id == camera_id
        )
        
        if hours:
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            query = query.where(CameraLatency.timestamp >= time_threshold)
        
        query = query.order_by(desc(CameraLatency.timestamp)).limit(limit)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def get_statistics_async(
        db: AsyncSession,
        camera_id: str,
        hours: int = 24
    ) -> Dict:
        """
        Calculate statistics for a camera over a time period
        
        Args:
            db: Database session
            camera_id: Camera identifier
            hours: Time window in hours (default 24)
        
        Returns:
            Dictionary with aggregated statistics
        """
        records = await CameraLatencyRepository.get_history_by_camera_async(
            db, camera_id, limit=1000, hours=hours
        )
        
        if not records:
            return {
                "camera_id": camera_id,
                "period_hours": hours,
                "total_checks": 0,
                "avg_rtt": None,
                "min_rtt": None,
                "max_rtt": None,
                "avg_packet_loss": None,
                "uptime_percent": None,
                "message": "No data available"
            }
        
        # Calculate statistics
        rtts = [r.rtt_avg for r in records if r.rtt_avg is not None]
        packet_losses = [r.packet_loss_percent for r in records if r.packet_loss_percent is not None]
        reachable_count = sum(1 for r in records if r.is_reachable)
        
        return {
            "camera_id": camera_id,
            "period_hours": hours,
            "total_checks": len(records),
            "avg_rtt": sum(rtts) / len(rtts) if rtts else None,
            "min_rtt": min(rtts) if rtts else None,
            "max_rtt": max(rtts) if rtts else None,
            "avg_packet_loss": sum(packet_losses) / len(packet_losses) if packet_losses else None,
            "uptime_percent": (reachable_count / len(records) * 100) if records else None,
            "latest_check": records[0].timestamp.isoformat() if records else None
        }
    
    @staticmethod
    async def get_all_cameras_latest_async(
        db: AsyncSession,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Get latest latency record for all cameras
        
        Returns:
            Dictionary mapping camera_id to latest CameraLatency object
        """
        from storage.models import CameraLatency
        
        # Get latest record per camera using subquery
        subquery = select(
            CameraLatency.camera_id,
            func.max(CameraLatency.timestamp).label('max_timestamp')
        ).group_by(CameraLatency.camera_id).subquery()
        
        query = select(CameraLatency).join(
            subquery,
            and_(
                CameraLatency.camera_id == subquery.c.camera_id,
                CameraLatency.timestamp == subquery.c.max_timestamp
            )
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        records = result.scalars().all()
        
        return {record.camera_id: record for record in records}
    
    # ===== SYNC METHODS =====
    
    @staticmethod
    def create_sync(db: Session, latency_data: Dict):
        """Create latency record (sync version)"""
        from storage.models import CameraLatency
        
        latency = CameraLatency(
            camera_id=latency_data["camera_id"],
            rtt_avg=latency_data.get("rtt_avg"),
            rtt_min=latency_data.get("rtt_min"),
            rtt_max=latency_data.get("rtt_max"),
            packet_loss_percent=latency_data.get("packet_loss_percent"),
            packets_sent=latency_data.get("packets_sent"),
            packets_received=latency_data.get("packets_received"),
            is_reachable=latency_data.get("is_reachable", True),
            check_type=latency_data.get("check_type", "PING"),
            error_message=latency_data.get("error_message"),
            timestamp=latency_data.get("timestamp", datetime.utcnow())
        )
        
        db.add(latency)
        db.commit()
        db.refresh(latency)
        logger.info(f"Latency record created (sync) for camera {latency_data['camera_id']}")
        return latency
    
    @staticmethod
    def get_latest_by_camera_sync(db: Session, camera_id: str):
        """Get latest latency record (sync version)"""
        from storage.models import CameraLatency
        
        return db.query(CameraLatency).filter(
            CameraLatency.camera_id == camera_id
        ).order_by(desc(CameraLatency.timestamp)).first()
