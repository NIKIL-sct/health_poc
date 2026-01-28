"""
Redis Client Configuration
Shared Redis connection pool for all components
"""

import redis.asyncio as aioredis
import redis
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)

# Redis connection configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_DECODE_RESPONSES = True

# Global connection pools
_async_pool: Optional[aioredis.ConnectionPool] = None
_sync_pool: Optional[redis.ConnectionPool] = None


# ============================================
# ASYNC CLIENT (for FastAPI & Scheduler)
# ============================================

async def get_async_redis() -> aioredis.Redis:
    """Get async Redis client for FastAPI handlers"""
    global _async_pool
    
    if _async_pool is None:
        _async_pool = aioredis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=REDIS_DECODE_RESPONSES,
            max_connections=50
        )
    
    return aioredis.Redis(connection_pool=_async_pool)


async def close_async_redis():
    """Cleanup async pool on shutdown"""
    global _async_pool
    if _async_pool:
        await _async_pool.disconnect()
        _async_pool = None


# ============================================
# SYNC CLIENT (for Workers)
# ============================================

def get_sync_redis() -> redis.Redis:
    """Get sync Redis client for worker processes"""
    global _sync_pool
    
    if _sync_pool is None:
        _sync_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=REDIS_DECODE_RESPONSES,
            max_connections=20
        )
    
    return redis.Redis(connection_pool=_sync_pool)


# ============================================
# REDIS SCHEMA HELPERS
# ============================================

class RedisKeys:
    """Centralized Redis key patterns"""
    
    # Camera registry
    @staticmethod
    def camera(camera_id: str) -> str:
        """camera:{id} - Hash of camera config"""
        return f"camera:{camera_id}"
    
    # Scheduling timestamps
    @staticmethod
    def next_ip(camera_id: str) -> str:
        """next:ip:{id} - Unix timestamp for next IP check"""
        return f"next:ip:{camera_id}"
    
    @staticmethod
    def next_port(camera_id: str) -> str:
        """next:port:{id} - Unix timestamp for next port check"""
        return f"next:port:{camera_id}"
    
    @staticmethod
    def next_vision(camera_id: str) -> str:
        """next:vision:{id} - Unix timestamp for next vision check"""
        return f"next:vision:{camera_id}"
    
    # Task queues
    @staticmethod
    def queue_network() -> str:
        """queue:network - List of network check tasks"""
        return "queue:network"
    
    @staticmethod
    def queue_vision() -> str:
        """queue:vision - List of vision check tasks"""
        return "queue:vision"
    
    # Event logs
    @staticmethod
    def logs(camera_id: str) -> str:
        """logs:{id} - List of health events (capped)"""
        return f"logs:{camera_id}"
    
    # Latest summary
    @staticmethod
    def summary(camera_id: str) -> str:
        """summary:{id} - Hash of latest health state"""
        return f"summary:{camera_id}"
    
    # Active cameras set
    @staticmethod
    def active_cameras() -> str:
        """active:cameras - Set of enabled camera IDs"""
        return "active:cameras"


# ============================================
# REDIS DATA HELPERS
# ============================================

class RedisData:
    """Helper methods for Redis data operations"""
    
    @staticmethod
    async def store_camera(r: aioredis.Redis, camera_id: str, config: dict):
        """Store camera configuration in Redis"""
        pipe = r.pipeline()
        
        # Store camera config as hash
        pipe.hset(
            RedisKeys.camera(camera_id),
            mapping={
                "id": config["id"],
                "ip": config["ip"],
                "rtsp_port": str(config["rtsp_port"]),
                "rtsp_url": config["rtsp_url"],
                "enabled": "1",
                "interval_ip": "60",
                "interval_port": "15",
                "interval_vision": "120"
            }
        ) 
        
        # Add to active cameras set
        pipe.sadd(RedisKeys.active_cameras(), camera_id)
        
        # Initialize next run timestamps to now (immediate execution)
        import time
        now = int(time.time())
        pipe.set(RedisKeys.next_ip(camera_id), now)
        pipe.set(RedisKeys.next_port(camera_id), now)
        pipe.set(RedisKeys.next_vision(camera_id), now)
        
        await pipe.execute()
        logger.info(f"Stored camera {camera_id} in Redis")
    
    @staticmethod
    async def get_camera(r: aioredis.Redis, camera_id: str) -> Optional[dict]:
        """Retrieve camera configuration"""
        try:
            data = await r.hgetall(RedisKeys.camera(camera_id))
            if not data:
                return None
            
            return {
                "id": data["id"],
                "ip": data["ip"],
                "rtsp_port": int(data["rtsp_port"]),
                "rtsp_url": data["rtsp_url"],
                "enabled": data["enabled"] == "1",
                "interval_ip": int(data.get("interval_ip", 60)),
                "interval_port": int(data.get("interval_port", 15)),
                "interval_vision": int(data.get("interval_vision", 120))
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid camera data for {camera_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving camera {camera_id}: {e}")
            return None
    
    @staticmethod
    def get_camera_sync(r: redis.Redis, camera_id: str) -> Optional[dict]:
        """Sync version - retrieve camera configuration"""
        try:
            data = r.hgetall(RedisKeys.camera(camera_id))
            if not data:
                return None
            
            return {
                "id": data["id"],
                "ip": data["ip"],
                "rtsp_port": int(data["rtsp_port"]),
                "rtsp_url": data["rtsp_url"],
                "enabled": data["enabled"] == "1",
                "interval_ip": int(data.get("interval_ip", 60)),
                "interval_port": int(data.get("interval_port", 15)),
                "interval_vision": int(data.get("interval_vision", 120))
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid camera data for {camera_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving camera {camera_id}: {e}")
            return None
    
    @staticmethod
    async def enqueue_task(r: aioredis.Redis, queue: str, task: dict):
        """Push task to Redis queue"""
        try:
            await r.lpush(queue, json.dumps(task))
        except Exception as e:
            logger.error(f"Failed to enqueue task to {queue}: {e}")
            raise
    
    @staticmethod
    def enqueue_task_sync(r: redis.Redis, queue: str, task: dict):
        """Sync version for workers"""
        try:
            r.lpush(queue, json.dumps(task))
        except Exception as e:
            logger.error(f"Failed to enqueue task to {queue}: {e}")
            raise
    
    @staticmethod
    async def log_event(r: aioredis.Redis, camera_id: str, event: dict, max_events: int = 500):
        """
        Store event in Redis with automatic capping
        Uses LPUSH + LTRIM to maintain fixed-size list
        """
        try:
            pipe = r.pipeline()
            
            # Push event to front of list
            logs_key = RedisKeys.logs(camera_id)
            pipe.lpush(logs_key, json.dumps(event))
            
            # Keep only last N events
            pipe.ltrim(logs_key, 0, max_events - 1)
            
            await pipe.execute()
        except Exception as e:
            logger.error(f"Failed to log event for {camera_id}: {e}")
    
    @staticmethod
    def log_event_sync(r: redis.Redis, camera_id: str, event: dict, max_events: int = 500):
        """Sync version for workers"""
        try:
            pipe = r.pipeline()
            logs_key = RedisKeys.logs(camera_id)
            pipe.lpush(logs_key, json.dumps(event))
            pipe.ltrim(logs_key, 0, max_events - 1)
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to log event for {camera_id}: {e}")
    
    @staticmethod
    async def update_summary(r: aioredis.Redis, camera_id: str, summary: dict):
        """Update latest health summary"""
        try:
            await r.hset(
                RedisKeys.summary(camera_id),
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in summary.items()}
            )
        except Exception as e:
            logger.error(f"Failed to update summary for {camera_id}: {e}")
    
    @staticmethod
    def update_summary_sync(r: redis.Redis, camera_id: str, summary: dict):
        """Sync version for workers"""
        try:
            r.hset(
                RedisKeys.summary(camera_id),
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in summary.items()}
            )
        except Exception as e:
            logger.error(f"Failed to update summary for {camera_id}: {e}")
    
    @staticmethod
    async def get_logs(r: aioredis.Redis, camera_id: str, 
                      offset: int = 0, limit: int = 10) -> list:
        """Retrieve paginated event logs"""
        try:
            logs_key = RedisKeys.logs(camera_id)
            
            # LRANGE is 0-indexed, inclusive
            end = offset + limit - 1
            raw_logs = await r.lrange(logs_key, offset, end)
            
            return [json.loads(log) for log in raw_logs]
        except Exception as e:
            logger.error(f"Failed to get logs for {camera_id}: {e}")
            return []
    
    @staticmethod
    async def get_total_logs(r: aioredis.Redis, camera_id: str) -> int:
        """Get total number of stored events"""
        try:
            return await r.llen(RedisKeys.logs(camera_id))
        except Exception as e:
            logger.error(f"Failed to get total logs for {camera_id}: {e}")
            return 0
    
    @staticmethod
    async def get_summary(r: aioredis.Redis, camera_id: str) -> Optional[dict]:
        """Retrieve latest health summary"""
        try:
            data = await r.hgetall(RedisKeys.summary(camera_id))
            if not data:
                return None
            
            # Parse JSON fields
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed[k] = v
            
            return parsed
        except Exception as e:
            logger.error(f"Failed to get summary for {camera_id}: {e}")
            return None
    
    @staticmethod
    def get_summary_sync(r: redis.Redis, camera_id: str) -> Optional[dict]:
        """Sync version - retrieve latest health summary"""
        try:
            data = r.hgetall(RedisKeys.summary(camera_id))
            if not data:
                return None
            
            # Parse JSON fields
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed[k] = v
            
            return parsed
        except Exception as e:
            logger.error(f"Failed to get summary for {camera_id}: {e}")
            return None