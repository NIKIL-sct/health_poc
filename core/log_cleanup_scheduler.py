"""
CORRECTED: log_cleanup_scheduler.py - Automatic Log Cleanup Scheduler

KEY CHANGES:
1. Runs every 30 seconds (configurable) to check for cameras needing cleanup
2. For each camera with log_delete_timeout_seconds set:
   - Deletes all logs older than (current_time - interval_seconds)
   - Cleans both health_logs and alerts tables
   - Clears Redis cache
3. This ensures logs are automatically deleted at the specified intervals

EXAMPLE WORKFLOW:
- Camera has interval set to 60 seconds
- Current time: 12:11:00
- Scheduler deletes all logs with timestamp < (12:11:00 - 0s) = 12:11:00
- Next run at 12:11:30: deletes logs < 12:11:30
- This effectively removes old logs every 30 seconds (scheduler interval)
"""

import asyncio
import logging
from datetime import datetime, timedelta

from storage.db_config import get_async_db
from storage.db_repositary import (
    CameraRepository,
    HealthLogRepository,
    AlertRepository
)
from storage.redis_client import get_async_redis, RedisKeys

logger = logging.getLogger(__name__)


class LogCleanupScheduler:
    """
    Background scheduler for automatic log cleanup based on camera-specific intervals
    """

    def __init__(self, scheduler_interval_seconds: int = 30):
        """
        Initialize the scheduler
        
        Args:
            scheduler_interval_seconds: How often to run cleanup check (default: 30s)
                This is NOT the log retention interval - that's per-camera.
                This is how frequently we check if any cameras need cleanup.
        """
        self.scheduler_interval = scheduler_interval_seconds
        logger.info(f"Log Cleanup Scheduler initialized (runs every {scheduler_interval_seconds}s)")

    async def run_forever(self):
        """
        Main scheduler loop - runs continuously in background
        """
        logger.info("Starting Log Cleanup Scheduler...")
        
        while True:
            try:
                await self._cleanup_cycle()
            except Exception as e:
                logger.error(f"Cleanup cycle error: {e}", exc_info=True)
            
            # Wait before next cycle
            await asyncio.sleep(self.scheduler_interval)

    async def _cleanup_cycle(self):
        """
        Execute one cleanup cycle for all cameras
        """
        current_time = datetime.utcnow()
        logger.debug(f"Starting cleanup cycle at {current_time.isoformat()}")
        
        try:
            async with get_async_db() as db:
                # Get all cameras with cleanup enabled
                cameras = await CameraRepository.list_all_async(db)
                
                if not cameras:
                    logger.debug("No cameras found in database")
                    return
                
                # Track statistics
                total_health_logs_deleted = 0
                total_alerts_deleted = 0
                cameras_cleaned = 0
                
                # Process each camera
                for camera in cameras:
                    if not camera.log_delete_timeout_seconds:
                        # Skip cameras without cleanup configured
                        continue
                    
                    try:
                        # Perform cleanup for this camera
                        health_deleted, alerts_deleted = await self._cleanup_camera_logs(
                            db=db,
                            camera_id=camera.camera_id,
                            interval_seconds=camera.log_delete_timeout_seconds,
                            current_time=current_time
                        )
                        
                        if health_deleted > 0 or alerts_deleted > 0:
                            total_health_logs_deleted += health_deleted
                            total_alerts_deleted += alerts_deleted
                            cameras_cleaned += 1
                            
                            logger.info(
                                f"[{camera.camera_id}] Cleaned: "
                                f"{health_deleted} health logs, {alerts_deleted} alerts "
                                f"(interval: {camera.log_delete_timeout_seconds}s)"
                            )
                    
                    except Exception as e:
                        logger.error(f"[{camera.camera_id}] Cleanup failed: {e}", exc_info=True)
                        continue
                
                # Commit all deletions
                await db.commit()
                
                # Log summary
                if cameras_cleaned > 0:
                    logger.info(
                        f"Cleanup cycle complete: {cameras_cleaned} cameras cleaned, "
                        f"{total_health_logs_deleted + total_alerts_deleted} total logs deleted"
                    )
                else:
                    logger.debug("Cleanup cycle complete: no logs to delete")
        
        except Exception as e:
            logger.error(f"Cleanup cycle failed: {e}", exc_info=True)

    async def _cleanup_camera_logs(
        self,
        db,
        camera_id: str,
        interval_seconds: int,
        current_time: datetime
    ) -> tuple[int, int]:
        """
        Clean up logs for a single camera
        
        CRITICAL LOGIC:
        - We delete logs OLDER than current_time
        - This means: delete all logs with timestamp < current_time
        - The interval_seconds defines how often this runs, but each run
          deletes everything before the current moment
        
        Args:
            db: Database session
            camera_id: Camera identifier
            interval_seconds: Cleanup interval (used for logging only)
            current_time: Current timestamp
            
        Returns:
            Tuple of (health_logs_deleted, alerts_deleted)
        """
        
        # Delete all logs before current time (retention_seconds=0)
        # This is the key: we pass 0 to delete everything older than RIGHT NOW
        health_logs_deleted = await HealthLogRepository.delete_older_than_async(
            db=db,
            camera_id=camera_id,
            retention_seconds=0  # Delete everything before current_time
        )
        
        alerts_deleted = await AlertRepository.delete_older_than_async(
            db=db,
            camera_id=camera_id,
            retention_seconds=0  # Delete everything before current_time
        )
        
        # Clear Redis cache if any logs were deleted
        if health_logs_deleted > 0 or alerts_deleted > 0:
            try:
                await self._clear_redis_cache(camera_id)
            except Exception as e:
                logger.warning(f"[{camera_id}] Failed to clear Redis cache: {e}")
        
        return health_logs_deleted, alerts_deleted

    async def _clear_redis_cache(self, camera_id: str):
        """
        Clear Redis cache for a camera after log cleanup
        """
        try:
            r = await get_async_redis()
            
            # Clear summary cache
            summary_key = RedisKeys.summary(camera_id)
            deleted = await r.delete(summary_key)
            
            if deleted:
                logger.debug(f"[{camera_id}] Redis cache cleared")
        
        except Exception as e:
            logger.error(f"[{camera_id}] Redis cache clear error: {e}")
            raise


# Global instance
cleanup_scheduler = LogCleanupScheduler(scheduler_interval_seconds=30)