"""
Network Worker - Fair Task Distribution with Internal Dispatcher
FIXED: Variable name consistency bug (ip_up vs is_up)
"""

import asyncio
import json
import logging
from datetime import datetime
from storage.redis_client import get_async_redis, RedisKeys, RedisData
from core.ping_checker import PingChecker
from storage.camera_worker import log_ip_check, log_port_check

logger = logging.getLogger(__name__)


class FairNetworkWorker:
    """
    Fair task distribution architecture:
    
    PROBLEM: Multiple workers doing BRPOP on same queue causes:
    - Uneven distribution (Redis round-robin is unreliable)
    - Some workers idle while others overloaded
    
    SOLUTION:
    1. Single POLLER task continuously drains Redis queue
    2. Internal asyncio.Queue distributes to worker pool
    3. Workers pull from internal queue (guaranteed fair)
    
    RESULT:
    - Even distribution across all workers
    - No idle workers
    - Predictable performance
    """
    
    def __init__(self, worker_id: int, shared_queue: asyncio.Queue, concurrency: int = 10):
        self.worker_id = worker_id
        self.shared_queue = shared_queue
        self.concurrency = concurrency
        self.running = False
        self.ping_checker = PingChecker()
        self._tasks = []
    
    async def start(self):
        """Start worker pool consuming from shared queue"""
        if self.running:
            return
        
        self.running = True
        
        # Spawn concurrent consumers
        for i in range(self.concurrency):
            task = asyncio.create_task(self._consume_loop(i))
            self._tasks.append(task)
        
        logger.info(f"Network worker {self.worker_id} started with {self.concurrency} consumers")
    
    async def stop(self):
        """Stop worker gracefully"""
        self.running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info(f"Network worker {self.worker_id} stopped")
    
    async def _consume_loop(self, consumer_id: int):
        """Consumer loop - pulls from shared internal queue"""
        r = await get_async_redis()
        
        while self.running:
            try:
                # Pull from internal queue (fair distribution guaranteed)
                task = await asyncio.wait_for(
                    self.shared_queue.get(),
                    timeout=1.0
                )
                
                if task:
                    await self._process_task(r, task)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id}-{consumer_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_task(self, r, task: dict):
        """Process a single network check task"""
        camera_id = task["camera_id"]
        check_type = task["check_type"]
        camera = task["camera"]
        
        try:
            if check_type == "ip":
                await self._check_ip(r, camera_id, camera)
            elif check_type == "port":
                await self._check_port(r, camera_id, camera)
            else:
                logger.warning(f"Unknown check type: {check_type}")
        except Exception as e:
            logger.error(f"Error processing {check_type} for {camera_id}: {e}", exc_info=True)
    
    async def _check_ip(self, r, camera_id: str, camera: dict):
        """IP reachability check"""
        try:
            loop = asyncio.get_event_loop()
            ip_up = await loop.run_in_executor(
                None,
                self.ping_checker.ping_ip,
                camera["ip"]
            )
            
            # Log to storage (dual-write: Redis + optional JSON)
            await loop.run_in_executor(
                None,
                log_ip_check,
                camera_id,
                camera["ip"],
                ip_up
            )
            
            # Update Redis summary
            await self._update_redis_summary(r, camera_id, ip_status="UP" if ip_up else "DOWN")
            
            logger.info(f"[{camera_id}] IP check: {'UP' if ip_up else 'DOWN'}")
            
        except Exception as e:
            logger.error(f"IP check failed for {camera_id}: {e}", exc_info=True)
    
    async def _check_port(self, r, camera_id: str, camera: dict):
        """Port check"""
        try:
            info = self.ping_checker.extract_rtsp_info(camera["rtsp_url"])
            
            loop = asyncio.get_event_loop()
            port_ok = await loop.run_in_executor(
                None,
                self.ping_checker.check_port,
                info["ip"],
                info["port"]
            )
            
            # Log to storage (dual-write: Redis + optional JSON)
            await loop.run_in_executor(
                None,
                log_port_check,
                camera_id,
                info["ip"],
                info["port"],
                port_ok
            )
            
            # Update Redis summary
            await self._update_redis_summary(r, camera_id, port_status="UP" if port_ok else "DOWN")
            
            logger.info(f"[{camera_id}] PORT check: {'UP' if port_ok else 'DOWN'}")
            
        except Exception as e:
            logger.error(f"PORT check failed for {camera_id}: {e}", exc_info=True)
    
    async def _update_redis_summary(self, r, camera_id: str, **kwargs):
        """Update Redis summary with partial data"""
        try:
            summary = await RedisData.get_summary(r, camera_id)
            if not summary:
                summary = {
                    "Ip_status": "UNKNOWN",
                    "Port_status": "UNKNOWN",
                    "Vision_status": "UNKNOWN",
                    "Alert_active": False,
                    "Last_ip_check": None,
                    "Last_port_check": None,
                    "Last_vision_check": None
                }
            
            now = datetime.now().isoformat()
            
            if "ip_status" in kwargs:
                summary["Ip_status"] = kwargs["ip_status"]
                summary["Last_ip_check"] = now
            
            if "port_status" in kwargs:
                summary["Port_status"] = kwargs["port_status"]
                summary["Last_port_check"] = now
            
            await RedisData.update_summary(r, camera_id, summary)
            
        except Exception as e:
            logger.error(f"Failed to update Redis summary for {camera_id}: {e}")


class RedisPoller:
    """
    Single poller that drains Redis queue and distributes to workers
    Guarantees fair distribution
    """
    
    def __init__(self, shared_queue: asyncio.Queue):
        self.shared_queue = shared_queue
        self.running = False
        self._task = None
    
    async def start(self):
        """Start poller"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Redis poller started")
    
    async def stop(self):
        """Stop poller"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Redis poller stopped")
    
    async def _poll_loop(self):
        """Continuously drain Redis queue into internal queue"""
        r = await get_async_redis()
        queue_key = RedisKeys.queue_network()
        
        while self.running:
            try:
                # Block on Redis queue
                result = await r.brpop(queue_key, timeout=1)
                
                if result:
                    _, task_json = result
                    
                    if not task_json or not task_json.strip():
                        continue
                    
                    try:
                        task = json.loads(task_json)
                        
                        # Validate task
                        if not self._validate_task(task):
                            continue
                        
                        # Put in internal queue (workers will pull fairly)
                        await self.shared_queue.put(task)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Poller JSON decode error: {e}")
                        continue
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poller error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def _validate_task(self, task: dict) -> bool:
        """Validate task structure"""
        required_fields = ["camera_id", "check_type", "camera"]
        
        if not isinstance(task, dict):
            return False
        
        for field in required_fields:
            if field not in task:
                logger.error(f"Task missing field: {field}")
                return False
        
        if not isinstance(task["camera"], dict):
            logger.error("Task camera field is not a dict")
            return False
        
        return True


class NetworkWorkerPool:
    """
    Manages fair worker pool with single poller
    """
    
    def __init__(self, num_workers: int = 3, concurrency_per_worker: int = 10):
        self.num_workers = num_workers
        self.concurrency_per_worker = concurrency_per_worker
        
        # Shared internal queue (unbounded for now, can add max size)
        self.shared_queue = asyncio.Queue()
        
        self.poller = RedisPoller(self.shared_queue)
        self.workers = []
    
    async def start(self):
        """Start poller and all workers"""
        # Start poller first
        await self.poller.start()
        
        # Start workers
        for i in range(self.num_workers):
            worker = FairNetworkWorker(i, self.shared_queue, self.concurrency_per_worker)
            await worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started 1 poller + {self.num_workers} workers with fair distribution")
    
    async def stop(self):
        """Stop all components"""
        await self.poller.stop()
        await asyncio.gather(*[w.stop() for w in self.workers])
        logger.info("All network workers stopped")


# Global instance
network_worker_pool = NetworkWorkerPool()