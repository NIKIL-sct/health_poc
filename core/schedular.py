"""
Redis-Based Task Scheduler - OPTIMIZED WITH ZSET TIME-WHEEL
Uses sorted sets for O(log N) scheduling instead of O(N) loops
Eliminates per-camera Redis calls through batch processing
"""

import asyncio
import time
import logging
import json
from typing import Set, List, Tuple, Dict
from storage.redis_client import get_async_redis, RedisKeys, RedisData

logger = logging.getLogger(__name__)


class HealthCheckScheduler:
    """
    High-performance scheduler using Redis ZSET time-wheel pattern
    
    ARCHITECTURE:
    1. Three ZSETs: schedule:ip, schedule:port, schedule:vision
    2. Scores are Unix timestamps (next run time)
    3. ZRANGEBYSCORE fetches all due tasks in O(log N + M)
    4. Batch enqueue and reschedule in pipelines
    
    PERFORMANCE:
    - Old: O(N * 6) Redis calls per cycle (N = cameras)
    - New: O(1) ZRANGEBYSCORE + O(M) pipeline (M = due tasks)
    """
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.running = False
        self._task = None
        
        # ZSET keys for time-wheel scheduling
        self.schedule_keys = {
            'ip': 'schedule:ip',
            'port': 'schedule:port',
            'vision': 'schedule:vision'
        }
    
    async def start(self):
        """Start scheduler background task"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("ZSET-based scheduler started")
    
    async def stop(self):
        """Stop scheduler gracefully"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _run(self):
        """Main scheduler loop with time-wheel pattern"""
        r = await get_async_redis()
        
        while self.running:
            try:
                await self._process_time_wheel(r)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _process_time_wheel(self, r):
        """
        Process all three time-wheels (IP, PORT, VISION)
        Uses ZRANGEBYSCORE for O(log N) retrieval of due tasks
        """
        now = int(time.time())
        
        # Fetch all due tasks from each ZSET
        for check_type, schedule_key in self.schedule_keys.items():
            try:
                # Get all tasks with score <= now (due tasks)
                due_camera_ids = await r.zrangebyscore(
                    schedule_key, 
                    '-inf', 
                    now
                )
                
                if not due_camera_ids:
                    continue
                
                logger.debug(f"Processing {len(due_camera_ids)} due {check_type} checks")
                
                # Process in batches to avoid overwhelming Redis
                batch_size = 50
                for i in range(0, len(due_camera_ids), batch_size):
                    batch = due_camera_ids[i:i + batch_size]
                    await self._process_batch(r, check_type, batch, now)
                    
            except Exception as e:
                logger.error(f"Error processing {check_type} time-wheel: {e}")
    
    async def _process_batch(self, r, check_type: str, camera_ids: List[str], now: int):
        """
        Process a batch of due tasks efficiently
        Single pipeline for: fetch configs, enqueue tasks, reschedule
        """
        try:
            # === PHASE 1: Batch fetch camera configs and summaries ===
            pipe = r.pipeline()
            for camera_id in camera_ids:
                pipe.hgetall(RedisKeys.camera(camera_id))
                if check_type == 'vision':
                    # Vision needs network status to decide
                    pipe.hgetall(RedisKeys.summary(camera_id))
            
            results = await pipe.execute()
            
            # === PHASE 2: Prepare tasks and reschedule ===
            tasks_to_enqueue = []
            reschedule_ops = []
            
            idx = 0
            for camera_id in camera_ids:
                try:
                    camera_data = results[idx]
                    idx += 1
                    
                    if not camera_data or camera_data.get("enabled") != "1":
                        # Remove from schedule if disabled
                        reschedule_ops.append(('rem', camera_id))
                        continue
                    
                    # Parse camera config
                    camera = {
                        "id": camera_data["id"],
                        "ip": camera_data["ip"],
                        "rtsp_port": int(camera_data["rtsp_port"]),
                        "rtsp_url": camera_data["rtsp_url"],
                        "enabled": True,
                        "interval_ip": int(camera_data.get("interval_ip", 60)),
                        "interval_port": int(camera_data.get("interval_port", 15)),
                        "interval_vision": int(camera_data.get("interval_vision", 120))
                    }
                    
                    # Vision check: verify network health
                    if check_type == 'vision':
                        summary_data = results[idx]
                        idx += 1
                        
                        # Parse summary
                        network_healthy = True
                        if summary_data:
                            try:
                                ip_status = summary_data.get("Ip_status", "UNKNOWN")
                                port_status = summary_data.get("Port_status", "UNKNOWN")
                                network_healthy = (ip_status == "UP" and port_status == "UP")
                            except Exception:
                                network_healthy = False
                        
                        if not network_healthy:
                            # Network down - reschedule vision for later, don't execute
                            interval = camera["interval_vision"]
                            reschedule_ops.append(('add', camera_id, now + interval))
                            continue
                    
                    # Create task
                    task = {
                        "camera_id": camera_id,
                        "check_type": check_type,
                        "scheduled_at": now,
                        "camera": camera
                    }
                    tasks_to_enqueue.append(task)
                    
                    # Schedule next run
                    interval_key = f"interval_{check_type}"
                    interval = camera[interval_key]
                    reschedule_ops.append(('add', camera_id, now + interval))
                    
                except Exception as e:
                    logger.error(f"Error processing {camera_id}: {e}")
                    continue
            
            # === PHASE 3: Execute batch operations ===
            if tasks_to_enqueue or reschedule_ops:
                pipe = r.pipeline()
                
                # Enqueue tasks
                queue = RedisKeys.queue_network() if check_type in ['ip', 'port'] else RedisKeys.queue_vision()
                for task in tasks_to_enqueue:
                    pipe.lpush(queue, json.dumps(task))
                
                # Reschedule in ZSET
                schedule_key = self.schedule_keys[check_type]
                for op in reschedule_ops:
                    if op[0] == 'add':
                        _, camera_id, next_time = op
                        pipe.zadd(schedule_key, {camera_id: next_time})
                    else:  # 'rem'
                        _, camera_id = op
                        pipe.zrem(schedule_key, camera_id)
                
                await pipe.execute()
                logger.debug(f"Enqueued {len(tasks_to_enqueue)} {check_type} tasks")
                
        except Exception as e:
            logger.error(f"Batch processing error for {check_type}: {e}", exc_info=True)
    
    async def register_camera(self, camera_id: str, intervals: Dict[str, int]):
        """
        Register a new camera in all time-wheels
        Called when camera is added via API
        """
        r = await get_async_redis()
        now = int(time.time())
        
        try:
            pipe = r.pipeline()
            
            # Add to all three ZSETs with immediate execution (score = now)
            pipe.zadd(self.schedule_keys['ip'], {camera_id: now})
            pipe.zadd(self.schedule_keys['port'], {camera_id: now})
            pipe.zadd(self.schedule_keys['vision'], {camera_id: now})
            
            await pipe.execute()
            logger.info(f"Registered {camera_id} in time-wheels")
            
        except Exception as e:
            logger.error(f"Failed to register {camera_id}: {e}")
    
    async def unregister_camera(self, camera_id: str):
        """Remove camera from all time-wheels"""
        r = await get_async_redis()
        
        try:
            pipe = r.pipeline()
            
            for schedule_key in self.schedule_keys.values():
                pipe.zrem(schedule_key, camera_id)
            
            await pipe.execute()
            logger.info(f"Unregistered {camera_id} from time-wheels")
            
        except Exception as e:
            logger.error(f"Failed to unregister {camera_id}: {e}")


# ============================================
# Scheduler Instance
# ============================================

scheduler = HealthCheckScheduler()