"""
If you already have cameras in Redis but not in DB,
run this migration script ONCE to sync them.

File: migrate_redis_to_db.py
"""

import asyncio
from storage.redis_client import get_async_redis, RedisKeys
from storage.db_config import get_async_db
from storage.db_repositary import CameraRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_cameras_to_db():
    """
    One-time migration: Copy all cameras from Redis to PostgreSQL
    """
    r = await get_async_redis()
    
    # Get all active camera IDs from Redis
    camera_ids = await r.smembers(RedisKeys.active_cameras())
    
    logger.info(f"Found {len(camera_ids)} cameras in Redis")
    
    migrated = 0
    skipped = 0
    failed = 0
    
    async with get_async_db() as db:
        for camera_id in camera_ids:
            try:
                # Check if already exists in DB
                existing = await CameraRepository.get_by_id_async(db, camera_id)
                
                if existing:
                    logger.info(f"Camera {camera_id} already in DB - skipping")
                    skipped += 1
                    continue
                
                # Get camera data from Redis
                camera_key = RedisKeys.camera(camera_id)
                camera_data = await r.hgetall(camera_key)
                
                if not camera_data:
                    logger.warning(f"Camera {camera_id} has no data in Redis - skipping")
                    skipped += 1
                    continue
                
                # Prepare data for DB
                db_data = {
                    "id": camera_id,
                    "ip": camera_data.get("ip", "0.0.0.0"),
                    "port": int(camera_data.get("port", 554)),
                    "rtsp_url": camera_data.get("rtsp_url", f"rtsp://{camera_id}"),
                    "interval_ip": int(camera_data.get("interval_ip", 60)),
                    "interval_port": int(camera_data.get("interval_port", 15)),
                    "interval_vision": int(camera_data.get("interval_vision", 120))
                }
                
                # Create in DB
                await CameraRepository.create_async(db, db_data)
                await db.commit()
                
                logger.info(f"✓ Migrated camera {camera_id} to DB")
                migrated += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate camera {camera_id}: {e}")
                failed += 1
                continue
    
    logger.info(f"""
    Migration complete:
    - Migrated: {migrated}
    - Skipped: {skipped}
    - Failed: {failed}
    """)


if __name__ == "__main__":
    asyncio.run(migrate_cameras_to_db())
