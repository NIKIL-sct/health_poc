"""
Vision Worker - Process Pool Consumer with Per-Camera JSON Logging
Consumes vision tasks from Redis and processes them reliably
Updates both Redis and local JSON logs
"""

import json
import logging
import multiprocessing as mp
import time
from datetime import datetime

from storage.redis_client import get_sync_redis, RedisKeys, RedisData
from core.vision_checker import VisionChecker

# Import camera logger
from storage.camera_worker import log_vision_check, log_startup

# ======================================================
# Logging Setup
# ======================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-worker")


# ======================================================
# Worker Process
# ======================================================

def vision_worker_process(worker_id: int, stop_event: mp.Event):
    logger.info(
        f"Vision worker {worker_id} started (PID={mp.current_process().pid})"
    )

    r = get_sync_redis()
    vision_checker = VisionChecker()
    queue_key = RedisKeys.queue_vision()

    while not stop_event.is_set():
        try:
            result = r.brpop(queue_key, timeout=1)

            if not result:
                continue

            _, task_json = result
            if not task_json or not task_json.strip():
                logger.warning(f"Worker {worker_id} received empty task")
                continue

            try:
                task = json.loads(task_json)
                
                # Validate task structure
                if not isinstance(task, dict) or "camera_id" not in task or "camera" not in task:
                    logger.error(f"Worker {worker_id} received invalid task: {task}")
                    continue
                
                process_vision_task(r, vision_checker, task)
                
            except json.JSONDecodeError as e:
                logger.error(f"Worker {worker_id} JSON decode error: {e}")
                continue

        except Exception as e:
            logger.error(
                f"Worker {worker_id} error: {e}",
                exc_info=True
            )
            time.sleep(1)

    logger.info(f"Vision worker {worker_id} stopped")


# ======================================================
# Vision Task Processing
# ======================================================

def process_vision_task(r, vision_checker: VisionChecker, task: dict):
    camera_id = task["camera_id"]
    camera = task["camera"]

    logger.info(f"[{camera_id}] Vision check started")

    try:
        vision_result = vision_checker.check_camera_vision(
            camera_id,
            camera["rtsp_url"]
        )

        # 🔹 Log to per-camera JSON file
        log_vision_check(camera_id, vision_result)

        # 🔹 Update Redis summary
        update_redis_summary(r, camera_id, vision_result)

        logger.info(
            f"[{camera_id}] Vision result: {vision_result['status']}"
        )
        
    except Exception as e:
        logger.error(f"[{camera_id}] Vision check failed: {e}", exc_info=True)


# ======================================================
# Redis Summary Update
# ======================================================

def update_redis_summary(r, camera_id: str, vision_result: dict):
    """Update Redis summary (preserve network status)"""
    try:
        summary_key = RedisKeys.summary(camera_id)
        existing = RedisData.get_summary_sync(r, camera_id)

        summary = {
            "Ip_status": "UNKNOWN",
            "Port_status": "UNKNOWN",
            "Vision_status": vision_result["status"],
            "Alert_active": bool(vision_result.get("alerts")),
            "Last_ip_check": None,
            "Last_port_check": None,
            "Last_vision_check": datetime.now().isoformat()
        }

        if existing:
            # Preserve network status from Redis
            summary["Ip_status"] = existing.get("Ip_status", "UNKNOWN")
            summary["Port_status"] = existing.get("Port_status", "UNKNOWN")
            summary["Last_ip_check"] = existing.get("Last_ip_check")
            summary["Last_port_check"] = existing.get("Last_port_check")

        RedisData.update_summary_sync(r, camera_id, summary)
        
    except Exception as e:
        logger.error(f"Failed to update Redis summary for {camera_id}: {e}")


# ======================================================
# Vision Worker Pool
# ======================================================

class VisionWorkerPool:
    """
    CPU-bound worker pool
    Workers = CPU cores - 1
    """

    def __init__(self, num_workers: int = None):
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        self.num_workers = num_workers
        self.processes = []
        self.stop_event = mp.Event()

    def start(self):
        logger.info(f"Starting {self.num_workers} vision workers")

        for i in range(self.num_workers):
            p = mp.Process(
                target=vision_worker_process,
                args=(i, self.stop_event),
                daemon=True
            )
            p.start()
            self.processes.append(p)

    def stop(self):
        logger.info("Stopping vision workers")
        self.stop_event.set()

        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        logger.info("All vision workers stopped")


# Global instance
vision_worker_pool = VisionWorkerPool()
