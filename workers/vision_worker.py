"""
Vision Worker - FIXED: Parallel processing with multiprocessing based on CPU cores
"""

import multiprocessing as mp
import logging
import time
from datetime import datetime
from storage.redis_client import get_sync_redis

from storage.db_config import get_sync_db

from storage.db_repositary import HealthLogRepository, AlertRepository
from core.vision_checker import VisionChecker
import json
from storage.redis_client import RedisKeys, RedisData


logger = logging.getLogger(__name__)


def vision_worker_process(worker_id: int, stop_event: mp.Event):
    """Vision worker process - runs in separate process"""
    logger.info(f"✓ Vision worker {worker_id} started (PID={mp.current_process().pid})")
    
    redis_client = get_sync_redis()
    vision_checker = VisionChecker()
    
    while not stop_event.is_set():
        try:
            # Get camera from queue
            result = redis_client.blpop(RedisKeys.queue_vision(), timeout=1)

            if not result:
                continue

            _, raw_task = result
            task = json.loads(raw_task)

            camera_id = task["camera_id"]

            camera = RedisData.get_camera_sync(redis_client, camera_id)

            if not camera:
                logger.warning(f"Camera {camera_id} not found in Redis")
                continue

            if not camera:
                logger.warning(f"Camera {camera_id} not found")
                continue
            
            # Perform vision check
            result = vision_checker.check_vision(camera_id, camera["rtsp_url"])
            
            # Store result in DB
            store_vision_result(camera_id, result)
            
            logger.debug(f"✓ Vision check completed: {camera_id}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            time.sleep(1)
    
    logger.info(f"Vision worker {worker_id} stopped")


def store_vision_result(camera_id: str, result: dict):
    """Store vision result in database - FIXED"""
    db = None
    try:
        db = get_sync_db()
        
        # Store vision log
        vision_log = {
            "camera_id": camera_id,
            "event_type": "VISION_CHECK",
            "status": result["status"],
            "message": result["message"],
            "timestamp": datetime.utcnow(),
            "meta_data": {
                k: v for k, v in result.items()
                if k not in ["status", "message"]
            }
        }
        HealthLogRepository.create_sync(db, vision_log)
        
        # Handle alerts
        if result["status"] == "FAIL":
            alert = {
                "camera_id": camera_id,
                "alert_type": "VISION_FAILURE",
                "severity": "CRITICAL",
                "message": result["message"]
            }
            AlertRepository.create_sync(db, alert)
        else:
            AlertRepository.resolve_alerts_by_type_sync(db, camera_id, "VISION_FAILURE")
        
        if result.get("is_black"):
            alert = {
                "camera_id": camera_id,
                "alert_type": "BLACK_FRAME",
                "severity": "MEDIUM",
                "message": "Black frame detected"
            }
            AlertRepository.create_sync(db, alert)
        else:
            AlertRepository.resolve_alerts_by_type_sync(db, camera_id, "BLACK_FRAME")
        
    except Exception as e:
        logger.error(f"Failed to store vision result for {camera_id}: {e}")
    finally:
        if db:
            db.close()


class VisionWorkerPool:
    """Vision worker pool manager"""
    
    def __init__(self, num_workers: int = None):
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)  # Based on CPU cores
        
        self.num_workers = num_workers
        self.processes = []
        self.stop_event = mp.Event()
        logger.info(f"Vision pool: {num_workers} workers (CPU cores: {mp.cpu_count()})")
    
    def start(self):
        """Start all workers"""
        for i in range(self.num_workers):
            p = mp.Process(
                target=vision_worker_process,
                args=(i, self.stop_event),
                daemon=True
            )
            p.start()
            self.processes.append(p)
        
        logger.info(f"✓ Started {self.num_workers} vision workers")
    
    def stop(self):
        """Stop all workers"""
        self.stop_event.set()
        
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self.processes.clear()
        logger.info("✓ All vision workers stopped")


vision_worker_pool = VisionWorkerPool()