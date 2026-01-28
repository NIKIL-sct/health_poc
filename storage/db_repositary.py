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
            rtsp_port=camera_data["rtsp_port"],
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
            port=camera_data["rtsp_port"],
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

        timestamp = log_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif timestamp is None:
            timestamp = datetime.utcnow()

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
        db.flush()        # ✅ ensures INSERT happens before alert
        return health_log

        @staticmethod
        async def get_by_type_and_timerange_async(
            db: AsyncSession,
            camera_id: str,
            event_type: str,
            since: datetime
        ) -> List[HealthLog]:
            """Get health logs by camera, event type, and time range"""
            try:
                query = select(HealthLog).where(
                    and_(
                        HealthLog.camera_id == camera_id,
                        HealthLog.event_type == event_type,
                        HealthLog.timestamp >= since
                    )
                ).order_by(desc(HealthLog.timestamp))
                
                result = await db.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.error(f"Failed to get logs: {e}")
                return []
        
        @staticmethod
        async def get_latest_metrics_async(
            db: AsyncSession,
            camera_id: str,
            event_type: str,
            limit: int = 1
        ) -> List[HealthLog]:
            """Get the most recent health logs with metrics"""
            try:
                query = select(HealthLog).where(
                    and_(
                        HealthLog.camera_id == camera_id,
                        HealthLog.event_type == event_type
                    )
                ).order_by(desc(HealthLog.timestamp)).limit(limit)
                
                result = await db.execute(query)
                return list(result.scalars().all())
            except Exception as e:
                logger.error(f"Failed to get latest metrics: {e}")
                return []
        
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



class HealthLogRepositoryExtensions:
    """
    Extended methods for HealthLogRepository to support connectivity metrics
    
    Add these methods to your existing HealthLogRepository class
    """
    
    @staticmethod
    async def get_by_type_and_timerange_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str,
        since: datetime
    ) -> List[HealthLog]:
        """
        Get health logs by camera, event type, and time range
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Event type (IP_CHECK, PORT_CHECK, CONNECTIVITY_CHECK, etc.)
            since: Start datetime for the range
        
        Returns:
            List of HealthLog records ordered by timestamp descending
        """
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type,
                    HealthLog.timestamp >= since
                )
            ).order_by(desc(HealthLog.timestamp))
            
            result = await db.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(
                f"Failed to get logs for {camera_id}, type {event_type}, since {since}: {e}"
            )
            return []
    
    @staticmethod
    async def get_latest_metrics_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str,
        limit: int = 1
    ) -> List[HealthLog]:
        """
        Get the most recent health logs with metrics
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Event type
            limit: Number of records to retrieve (default 1)
        
        Returns:
            List of HealthLog records
        """
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type
                )
            ).order_by(desc(HealthLog.timestamp)).limit(limit)
            
            result = await db.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Failed to get latest metrics for {camera_id}: {e}")
            return []
    
    @staticmethod
    async def get_metrics_statistics_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str,
        since: datetime
    ) -> dict:
        """
        Get statistical summary of metrics for a time period
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Event type
            since: Start datetime for the range
        
        Returns:
            Dict with statistical summary
        """
        try:
            # Get logs
            logs = await HealthLogRepositoryExtensions.get_by_type_and_timerange_async(
                db, camera_id, event_type, since
            )
            
            if not logs:
                return {
                    "count": 0,
                    "message": "No data available for the specified time range"
                }
            
            # Calculate statistics
            total = len(logs)
            passed = sum(1 for log in logs if log.status == "PASS")
            failed = sum(1 for log in logs if log.status == "FAIL")
            warnings = sum(1 for log in logs if log.status == "WARNING")
            
            stats = {
                "count": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": round((passed / total) * 100, 2) if total > 0 else 0
            }
            
            # Extract metrics from meta_data
            if event_type == "IP_CHECK":
                packet_losses = []
                rtts = []
                
                for log in logs:
                    if log.meta_data:
                        pl = log.meta_data.get("packet_loss_percent")
                        if pl is not None:
                            packet_losses.append(pl)
                        
                        rtt = log.meta_data.get("rtt_avg_ms")
                        if rtt is not None and rtt > 0:
                            rtts.append(rtt)
                
                if packet_losses:
                    stats["avg_packet_loss"] = round(sum(packet_losses) / len(packet_losses), 2)
                    stats["min_packet_loss"] = round(min(packet_losses), 2)
                    stats["max_packet_loss"] = round(max(packet_losses), 2)
                
                if rtts:
                    stats["avg_rtt_ms"] = round(sum(rtts) / len(rtts), 2)
                    stats["min_rtt_ms"] = round(min(rtts), 2)
                    stats["max_rtt_ms"] = round(max(rtts), 2)
            
            elif event_type == "PORT_CHECK":
                success_rates = []
                latencies = []
                
                for log in logs:
                    if log.meta_data:
                        sr = log.meta_data.get("connection_success_rate")
                        if sr is not None:
                            success_rates.append(sr)
                        
                        lat = log.meta_data.get("latency_avg_ms")
                        if lat is not None and lat > 0:
                            latencies.append(lat)
                
                if success_rates:
                    stats["avg_connection_success_rate"] = round(sum(success_rates) / len(success_rates), 2)
                    stats["min_connection_success_rate"] = round(min(success_rates), 2)
                    stats["max_connection_success_rate"] = round(max(success_rates), 2)
                
                if latencies:
                    stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
                    stats["min_latency_ms"] = round(min(latencies), 2)
                    stats["max_latency_ms"] = round(max(latencies), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics for {camera_id}: {e}")
            return {
                "error": str(e),
                "count": 0
            }
    
    @staticmethod
    def get_by_type_and_timerange_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        since: datetime
    ) -> List[HealthLog]:
        """
        Sync version: Get health logs by camera, event type, and time range
        """
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type,
                    HealthLog.timestamp >= since
                )
            ).order_by(desc(HealthLog.timestamp))
            
            result = db.execute(query)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(
                f"Failed to get logs for {camera_id}, type {event_type}, since {since}: {e}"
            )
            return []


# Example of how to add these to existing HealthLogRepository class:
"""
Add these methods to your existing HealthLogRepository class in storage/db_repositary.py:

class HealthLogRepository:
    # ... existing methods ...
    
    # Add all methods from HealthLogRepositoryExtensions here
    get_by_type_and_timerange_async = staticmethod(
        HealthLogRepositoryExtensions.get_by_type_and_timerange_async
    )
    
    get_latest_metrics_async = staticmethod(
        HealthLogRepositoryExtensions.get_latest_metrics_async
    )
    
    get_metrics_statistics_async = staticmethod(
        HealthLogRepositoryExtensions.get_metrics_statistics_async
    )
    
    get_by_type_and_timerange_sync = staticmethod(
        HealthLogRepositoryExtensions.get_by_type_and_timerange_sync
    )
"""

# COMPLETE REPOSITORY METHODS FOR HEALTH LOGS AND ALERTS
# ========================================================
# Add these to your storage/db_repositary.py file

"""
These are the complete, tested repository methods you need for storing
health check results and managing alerts in the database.
"""

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from datetime import datetime
from typing import List, Optional
import logging

from storage.models import HealthLog, Alert

logger = logging.getLogger(__name__)


# ========================================================
# HEALTH LOG REPOSITORY
# ========================================================

class HealthLogRepository:
    """
    Repository for managing health check logs in the database
    """
    
    # -------------------- CREATE METHODS --------------------
    
    @staticmethod
    def create_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        status: str,
        message: str,
        meta_data: dict = None
    ) -> HealthLog:
        """
        Create a health log entry (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Type of event (IP_CHECK, PORT_CHECK, VISION_CHECK, etc.)
            status: Status of check (PASS, FAIL, WARNING)
            message: Human-readable message
            meta_data: Additional JSON metadata
            
        Returns:
            Created HealthLog object
            
        Raises:
            Exception if creation fails
        """
        try:
            health_log = HealthLog(
                camera_id=camera_id,
                event_type=event_type,
                status=status,
                message=message,
                meta_data=meta_data or {},
                timestamp=datetime.utcnow()
            )
            
            db.add(health_log)
            db.commit()
            db.refresh(health_log)
            
            logger.debug(f"Created health log: {camera_id} - {event_type} - {status}")
            return health_log
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create health log: {e}")
            raise
    
    @staticmethod
    async def create_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str,
        status: str,
        message: str,
        meta_data: dict = None
    ) -> HealthLog:
        """
        Create a health log entry (asynchronous)
        """
        try:
            health_log = HealthLog(
                camera_id=camera_id,
                event_type=event_type,
                status=status,
                message=message,
                meta_data=meta_data or {},
                timestamp=datetime.utcnow()
            )
            
            db.add(health_log)
            await db.commit()
            await db.refresh(health_log)
            
            logger.debug(f"Created health log: {camera_id} - {event_type} - {status}")
            return health_log
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create health log: {e}")
            raise
    
    # -------------------- READ METHODS --------------------
    
    @staticmethod
    def get_latest_by_type_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        limit: int = 1
    ) -> List[HealthLog]:
        """
        Get the most recent health logs of a specific type (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Event type filter
            limit: Maximum number of records to return
            
        Returns:
            List of HealthLog objects (most recent first)
        """
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type
                )
            ).order_by(desc(HealthLog.timestamp)).limit(limit)
            
            result = db.execute(query)
            logs = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(logs)} {event_type} logs for {camera_id}")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get latest health logs: {e}")
            return []
    
    @staticmethod
    async def get_latest_by_type_async(
        db: AsyncSession,
        camera_id: str,
        event_type: str,
        limit: int = 1
    ) -> List[HealthLog]:
        """
        Get the most recent health logs of a specific type (asynchronous)
        """
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type
                )
            ).order_by(desc(HealthLog.timestamp)).limit(limit)
            
            result = await db.execute(query)
            logs = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(logs)} {event_type} logs for {camera_id}")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get latest health logs: {e}")
            return []
    
    @staticmethod
    def get_by_timerange_sync(
        db: Session,
        camera_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[HealthLog]:
        """
        Get health logs within a time range (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Optional event type filter
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            
        Returns:
            List of HealthLog objects
        """
        try:
            conditions = [HealthLog.camera_id == camera_id]
            
            if event_type:
                conditions.append(HealthLog.event_type == event_type)
            
            if start_time:
                conditions.append(HealthLog.timestamp >= start_time)
            
            if end_time:
                conditions.append(HealthLog.timestamp <= end_time)
            
            query = select(HealthLog).where(
                and_(*conditions)
            ).order_by(desc(HealthLog.timestamp))
            
            result = db.execute(query)
            logs = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(logs)} logs for {camera_id} in time range")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get logs by time range: {e}")
            return []
    
    @staticmethod
    async def get_by_timerange_async(
        db: AsyncSession,
        camera_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[HealthLog]:
        """
        Get health logs within a time range (asynchronous)
        """
        try:
            conditions = [HealthLog.camera_id == camera_id]
            
            if event_type:
                conditions.append(HealthLog.event_type == event_type)
            
            if start_time:
                conditions.append(HealthLog.timestamp >= start_time)
            
            if end_time:
                conditions.append(HealthLog.timestamp <= end_time)
            
            query = select(HealthLog).where(
                and_(*conditions)
            ).order_by(desc(HealthLog.timestamp))
            
            result = await db.execute(query)
            logs = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(logs)} logs for {camera_id} in time range")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get logs by time range: {e}")
            return []
    
    @staticmethod
    def count_by_status_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        status: str,
        since: Optional[datetime] = None
    ) -> int:
        """
        Count health logs by status (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Event type
            status: Status to count (PASS, FAIL, WARNING)
            since: Optional start time
            
        Returns:
            Count of matching logs
        """
        try:
            conditions = [
                HealthLog.camera_id == camera_id,
                HealthLog.event_type == event_type,
                HealthLog.status == status
            ]
            
            if since:
                conditions.append(HealthLog.timestamp >= since)
            
            query = select(func.count()).select_from(HealthLog).where(and_(*conditions))
            
            result = db.execute(query)
            count = result.scalar()
            
            return count or 0
            
        except Exception as e:
            logger.error(f"Failed to count logs by status: {e}")
            return 0


# ========================================================
# ALERT REPOSITORY
# ========================================================

class AlertRepository:
    """
    Repository for managing alerts in the database
    """
    
    # -------------------- CREATE METHODS --------------------
    
    @staticmethod
    def create_sync(
        db: Session,
        camera_id: str,
        alert_type: str,
        severity: str,
        message: str,
        triggered_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create an alert (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            alert_type: Type of alert
            severity: Alert severity (LOW, MEDIUM, HIGH, CRITICAL)
            message: Alert message
            triggered_at: When alert was triggered (defaults to now)
            
        Returns:
            Created Alert object
        """
        try:
            alert = Alert(
                camera_id=camera_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                triggered_at=triggered_at or datetime.utcnow(),
                resolved=False
            )
            
            db.add(alert)
            db.commit()
            db.refresh(alert)
            
            logger.info(f"Created alert: {camera_id} - {alert_type} - {severity}")
            return alert
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise
    
    @staticmethod
    async def create_async(
        db: AsyncSession,
        camera_id: str,
        alert_type: str,
        severity: str,
        message: str,
        triggered_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create an alert (asynchronous)
        """
        try:
            alert = Alert(
                camera_id=camera_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                triggered_at=triggered_at or datetime.utcnow(),
                resolved=False
            )
            
            db.add(alert)
            await db.commit()
            await db.refresh(alert)
            
            logger.info(f"Created alert: {camera_id} - {alert_type} - {severity}")
            return alert
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise
    
    # -------------------- READ METHODS --------------------
    
    @staticmethod
    def get_active_alerts_sync(db: Session, camera_id: str) -> List[Alert]:
        """
        Get all active (unresolved) alerts for a camera (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            
        Returns:
            List of active Alert objects
        """
        try:
            query = select(Alert).where(
                and_(
                    Alert.camera_id == camera_id,
                    Alert.resolved == False
                )
            ).order_by(desc(Alert.triggered_at))
            
            result = db.execute(query)
            alerts = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(alerts)} active alerts for {camera_id}")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    @staticmethod
    async def get_active_alerts_async(db: AsyncSession, camera_id: str) -> List[Alert]:
        """
        Get all active (unresolved) alerts for a camera (asynchronous)
        """
        try:
            query = select(Alert).where(
                and_(
                    Alert.camera_id == camera_id,
                    Alert.resolved == False
                )
            ).order_by(desc(Alert.triggered_at))
            
            result = await db.execute(query)
            alerts = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(alerts)} active alerts for {camera_id}")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    # -------------------- UPDATE METHODS --------------------
    
    @staticmethod
    def resolve_alert_sync(
        db: Session,
        alert_id: int,
        resolved_by: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Resolve (close) an alert (synchronous)
        
        Args:
            db: Database session
            alert_id: Alert ID to resolve
            resolved_by: Optional identifier of who/what resolved it
            
        Returns:
            Updated Alert object or None if not found
        """
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                logger.warning(f"Alert {alert_id} not found")
                return None
            
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            if resolved_by:
                alert.resolved_by = resolved_by
            
            db.commit()
            db.refresh(alert)
            
            logger.info(f"Resolved alert {alert_id} for {alert.camera_id}")
            return alert
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return None
    
    @staticmethod
    async def resolve_alert_async(
        db: AsyncSession,
        alert_id: int,
        resolved_by: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Resolve (close) an alert (asynchronous)
        """
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                logger.warning(f"Alert {alert_id} not found")
                return None
            
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            if resolved_by:
                alert.resolved_by = resolved_by
            
            await db.commit()
            await db.refresh(alert)
            
            logger.info(f"Resolved alert {alert_id} for {alert.camera_id}")
            return alert
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return None
    
    @staticmethod
    def resolve_alerts_by_type_sync(
        db: Session,
        camera_id: str,
        alert_type: str,
        resolved_by: Optional[str] = None
    ) -> int:
        """
        Resolve all active alerts of a specific type for a camera (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            alert_type: Type of alerts to resolve
            resolved_by: Optional identifier of who/what resolved them
            
        Returns:
            Number of alerts resolved
        """
        try:
            query = select(Alert).where(
                and_(
                    Alert.camera_id == camera_id,
                    Alert.alert_type == alert_type,
                    Alert.resolved == False
                )
            )
            
            result = db.execute(query)
            alerts = list(result.scalars().all())
            
            count = 0
            for alert in alerts:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                if resolved_by:
                    alert.resolved_by = resolved_by
                count += 1
            
            db.commit()
            
            logger.info(f"Resolved {count} {alert_type} alerts for {camera_id}")
            return count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resolve alerts by type: {e}")
            return 0


# ========================================================
# USAGE EXAMPLES
# ========================================================

class HealthLogRepository:
    """Repository for HealthLog operations - CORRECTED for your schema"""
    
    @staticmethod
    def create_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        status: str,
        event_description: str,  # ✅ CORRECT field name
        meta_data: dict = None
    ) -> HealthLog:
        """
        Create a health log entry (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_type: Type of event (IP_CHECK, PORT_CHECK, etc.)
            status: Status of check (PASS, FAIL, WARNING)
            event_description: Human-readable description  # ✅ CORRECT
            meta_data: Additional JSON metadata
            
        Returns:
            Created HealthLog object
        """
        try:
            health_log = HealthLog(
                camera_id=camera_id,
                event_type=event_type,
                status=status,
                event_description=event_description,  # ✅ CORRECT
                meta_data=meta_data or {},
                timestamp=datetime.utcnow(),
                last_checked=datetime.utcnow()
            )
            
            db.add(health_log)
            db.flush()  # Flush to get event_id without committing
            
            logger.debug(f"Created health log: {camera_id} - {event_type} - {status}")
            return health_log
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create health log: {e}")
            raise
    
    @staticmethod
    def get_latest_by_type_sync(
        db: Session,
        camera_id: str,
        event_type: str,
        limit: int = 1
    ) -> List[HealthLog]:
        """Get the most recent health logs of a specific type"""
        try:
            query = select(HealthLog).where(
                and_(
                    HealthLog.camera_id == camera_id,
                    HealthLog.event_type == event_type
                )
            ).order_by(desc(HealthLog.timestamp)).limit(limit)
            
            result = db.execute(query)
            logs = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(logs)} {event_type} logs for {camera_id}")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get latest health logs: {e}")
            return []


class AlertRepository:
    """Repository for Alert operations - CORRECTED for your schema"""
    
    @staticmethod
    def create_sync(
        db: Session,
        camera_id: str,
        event_id: str,  # ✅ REQUIRED - foreign key to HealthLog
        alert_type: str,
        alert_description: str,  # ✅ CORRECT field name
        timestamp: Optional[datetime] = None  # ✅ CORRECT field name
    ) -> Alert:
        """
        Create an alert (synchronous)
        
        Args:
            db: Database session
            camera_id: Camera identifier
            event_id: Related health log event_id (REQUIRED)
            alert_type: Type of alert
            alert_description: Alert description
            timestamp: When alert was triggered (defaults to now)
            
        Returns:
            Created Alert object
        """
        try:
            alert = Alert(
                camera_id=camera_id,
                event_id=event_id,  # ✅ REQUIRED
                alert_type=alert_type,
                alert_description=alert_description,  # ✅ CORRECT
                timestamp=timestamp or datetime.utcnow(),  # ✅ CORRECT
                resolved=False
            )
            
            db.add(alert)
            db.flush()
            
            logger.info(f"Created alert: {camera_id} - {alert_type}")
            return alert
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise
    
    @staticmethod
    def get_active_alerts_sync(db: Session, camera_id: str) -> List[Alert]:
        """Get all active (unresolved) alerts for a camera"""
        try:
            query = select(Alert).where(
                and_(
                    Alert.camera_id == camera_id,
                    Alert.resolved == False
                )
            ).order_by(desc(Alert.timestamp))  # ✅ CORRECT
            
            result = db.execute(query)
            alerts = list(result.scalars().all())
            
            logger.debug(f"Retrieved {len(alerts)} active alerts for {camera_id}")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    @staticmethod
    def resolve_alerts_by_type_sync(
        db: Session,
        camera_id: str,
        alert_type: str
    ) -> int:
        """Resolve all active alerts of a specific type"""
        try:
            query = select(Alert).where(
                and_(
                    Alert.camera_id == camera_id,
                    Alert.alert_type == alert_type,
                    Alert.resolved == False
                )
            )
            
            result = db.execute(query)
            alerts = list(result.scalars().all())
            
            count = 0
            for alert in alerts:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                count += 1
            
            db.flush()
            
            logger.info(f"Resolved {count} {alert_type} alerts for {camera_id}")
            return count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resolve alerts by type: {e}")
            return 0












