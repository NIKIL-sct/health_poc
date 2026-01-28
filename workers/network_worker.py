"""
Network Worker - 
Matches your actual database schema exactly
"""

import asyncio
import json
import logging
from datetime import datetime
from storage.redis_client import get_async_redis, RedisKeys, RedisData
from core.ping_checker import PingChecker
from storage.db_config import get_sync_db

logger = logging.getLogger(__name__)


class NetworkWorker:
    """
    Network worker with DATABASE PERSISTENCE
    Uses correct field names from your schema
    """
    
    def __init__(self, worker_id: int, shared_queue: asyncio.Queue, concurrency: int = 10):
        self.worker_id = worker_id
        self.shared_queue = shared_queue
        self.concurrency = concurrency
        self.running = False
        self.ping_checker = PingChecker()
        self._tasks = []
        logger.info(f"Network worker {worker_id} - Database session initialized")
    
    async def start(self):
        """Start worker pool consuming from shared queue"""
        if self.running:
            return
        
        self.running = True
        
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
        """Consumer loop"""
        r = await get_async_redis()
        
        while self.running:
            try:
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
                await self._check_ip_enhanced(r, camera_id, camera)
            elif check_type == "port":
                await self._check_port_enhanced(r, camera_id, camera)
            elif check_type == "connectivity":
                await self._check_connectivity_full(r, camera_id, camera)
            else:
                logger.warning(f"Unknown check type: {check_type}")
        except Exception as e:
            logger.error(f"Error processing {check_type} for {camera_id}: {e}", exc_info=True)
    
    def _save_to_database_and_create_alert(
        self, 
        camera_id: str, 
        event_type: str, 
        status: str, 
        event_description: str,  # ✅ CORRECT
        meta_data: dict,
        should_alert: bool,
        alert_type: str = None,
        alert_description: str = None
    ):
        """
        **CRITICAL: Save to database with CORRECT field names**
        Also creates alert if needed (Alert requires event_id from HealthLog)
        """
        try:
            with get_sync_db() as db:
                from storage.models import HealthLog, Alert
                
                # 1. Create HealthLog first
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
                db.flush()  # Get event_id without committing
                
                logger.debug(
                    f"[{camera_id}] ✓ {event_type} saved to DB "
                    f"(event_id: {health_log.event_id}, Status: {status})"
                )
                
                # 2. Create Alert if needed (requires event_id)
                if should_alert and alert_type and alert_description:
                    # Check if alert already exists
                    from sqlalchemy import select, and_
                    existing_alert_query = select(Alert).where(
                        and_(
                            Alert.camera_id == camera_id,
                            Alert.alert_type == alert_type,
                            Alert.resolved == False
                        )
                    )
                    result = db.execute(existing_alert_query)
                    existing_alert = result.scalar_one_or_none()
                    
                    if not existing_alert:
                        alert = Alert(
                            camera_id=camera_id,
                            event_id=health_log.event_id,  # ✅ REQUIRED
                            alert_type=alert_type,
                            alert_description=alert_description,  # ✅ CORRECT
                            timestamp=datetime.utcnow(),  # ✅ CORRECT
                            resolved=False
                        )
                        db.add(alert)
                        db.flush()
                        
                        logger.info(
                            f"[{camera_id}]  Alert created: {alert_type} "
                            f"(alert_id: {alert.alert_id})"
                        )
                    else:
                        logger.debug(f"[{camera_id}] Alert {alert_type} already exists")
                
                # Commit both health_log and alert
                db.commit()
                return health_log
                
        except Exception as e:
            logger.error(f"[{camera_id}] ✗ Failed to save to DB: {e}")
            return None
    
    def _resolve_alerts(self, camera_id: str, alert_type: str):
        """Resolve alerts when issue is fixed"""
        try:
            with get_sync_db() as db:
                from storage.models import Alert
                from sqlalchemy import select, and_
                
                # Find active alerts of this type
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
                
                if count > 0:
                    db.commit()
                    logger.info(f"[{camera_id}] ✓ Resolved {count} {alert_type} alert(s)")
                    
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to resolve alerts: {e}")
    
    async def _check_ip_enhanced(self, r, camera_id: str, camera: dict):
        """Enhanced IP check with DATABASE STORAGE"""
        try:
            loop = asyncio.get_event_loop()
            
            # Run enhanced ping
            metrics = await loop.run_in_executor(
                None,
                self.ping_checker.ping_with_metrics,
                camera["ip"],
                5
            )
            
            ip_up = metrics["is_reachable"]
            
            # Determine status
            if metrics["packet_loss_percent"] == 0:
                status = "PASS"
            elif metrics["packet_loss_percent"] < 50:
                status = "WARNING"
            else:
                status = "FAIL"
            
            # Prepare metadata
            meta_data = {
                "packet_loss_percent": metrics["packet_loss_percent"],
                "rtt_min_ms": metrics["rtt_min_ms"],
                "rtt_avg_ms": metrics["rtt_avg_ms"],
                "rtt_max_ms": metrics["rtt_max_ms"],
                "ip_address": camera["ip"],
                "is_reachable": ip_up,
                "check_timestamp": datetime.utcnow().isoformat()
            }
            
            # Format description
            if ip_up:
                event_description = (
                    f"IP check: UP | Loss: {metrics['packet_loss_percent']:.1f}% | "
                    f"RTT: {metrics['rtt_avg_ms']:.2f}ms "
                    f"(min:{metrics['rtt_min_ms']:.2f}, max:{metrics['rtt_max_ms']:.2f})"
                )
            else:
                event_description = f"IP check: DOWN | Loss: {metrics['packet_loss_percent']:.1f}%"
            
            # Save to database and create alert if needed
            await loop.run_in_executor(
                None,
                self._save_to_database_and_create_alert,
                camera_id,
                "IP_CHECK",
                status,
                event_description,
                meta_data,
                status == "FAIL",  # should_alert
                "IP_UNREACHABLE" if status == "FAIL" else None,
                f"Camera IP {camera['ip']} is unreachable (packet loss: {metrics['packet_loss_percent']:.1f}%)" if status == "FAIL" else None
            )
            
            # Resolve alerts if back up
            if status != "FAIL":
                await loop.run_in_executor(
                    None,
                    self._resolve_alerts,
                    camera_id,
                    "IP_UNREACHABLE"
                )
            
            # Update Redis summary
            summary_update = {
                "Ip_status": "UP" if ip_up else "DOWN",
                "Last_ip_check": datetime.now().isoformat(),
                "Ip_metrics": {
                    "packet_loss_percent": metrics["packet_loss_percent"],
                    "rtt_avg_ms": metrics["rtt_avg_ms"],
                    "rtt_min_ms": metrics["rtt_min_ms"],
                    "rtt_max_ms": metrics["rtt_max_ms"]
                }
            }
            await self._update_redis_summary(r, camera_id, **summary_update)
            
            logger.info(
                f"[{camera_id}] IP check: {'UP' if ip_up else 'DOWN'} | "
                f"Loss: {metrics['packet_loss_percent']:.1f}% | "
                f"RTT: {metrics['rtt_avg_ms']:.2f}ms (min:{metrics['rtt_min_ms']:.2f}, max:{metrics['rtt_max_ms']:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Enhanced IP check failed for {camera_id}: {e}", exc_info=True)
            
            # Save error to database
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._save_to_database_and_create_alert,
                camera_id,
                "IP_CHECK",
                "FAIL",
                f"IP check error: {str(e)}",
                {"error": str(e), "ip_address": camera.get("ip", "unknown")},
                False,  # don't create alert for errors
                None,
                None
            )
    
    async def _check_port_enhanced(self, r, camera_id: str, camera: dict):
        """Enhanced port check with DATABASE STORAGE"""
        try:
            info = self.ping_checker.extract_rtsp_info(camera["rtsp_url"])
            loop = asyncio.get_event_loop()
            
            # Run enhanced port check
            metrics = await loop.run_in_executor(
                None,
                self.ping_checker.check_port_with_latency,
                info["ip"],
                info["port"],
                3
            )
            
            port_ok = metrics["is_accessible"]
            
            # Determine status
            if metrics["connection_success_rate"] >= 80:
                status = "PASS"
            elif metrics["connection_success_rate"] >= 50:
                status = "WARNING"
            else:
                status = "FAIL"
            
            # Prepare metadata
            meta_data = {
                "connection_success_rate": metrics["connection_success_rate"],
                "latency_min_ms": metrics["latency_min_ms"],
                "latency_avg_ms": metrics["latency_avg_ms"],
                "latency_max_ms": metrics["latency_max_ms"],
                "ip_address": info["ip"],
                "port": info["port"],
                "is_accessible": port_ok,
                "check_timestamp": datetime.utcnow().isoformat()
            }
            
            # Format description
            latency_avg = metrics.get('latency_avg_ms')
            if port_ok and latency_avg is not None:
                event_description = (
                    f"PORT check: UP | Success Rate: {metrics['connection_success_rate']:.1f}% | "
                    f"Latency: {latency_avg:.2f}ms"
                )
            else:
                event_description = (
                    f"PORT check: DOWN | Success Rate: {metrics['connection_success_rate']:.1f}% | "
                    f"Latency: None"
                )
            
            # Save to database
            await loop.run_in_executor(
                None,
                self._save_to_database_and_create_alert,
                camera_id,
                "PORT_CHECK",
                status,
                event_description,
                meta_data,
                status == "FAIL",
                "PORT_UNREACHABLE" if status == "FAIL" else None,
                f"Port {info['port']} on {info['ip']} is unreachable" if status == "FAIL" else None
            )
            
            # Resolve alerts if back up
            if status != "FAIL":
                await loop.run_in_executor(
                    None,
                    self._resolve_alerts,
                    camera_id,
                    "PORT_UNREACHABLE"
                )
            
            # Update Redis summary
            summary_update = {
                "Port_status": "UP" if port_ok else "DOWN",
                "Last_port_check": datetime.now().isoformat(),
                "Port_metrics": {
                    "connection_success_rate": metrics["connection_success_rate"],
                    "latency_avg_ms": metrics.get("latency_avg_ms", 0),
                    "latency_min_ms": metrics.get("latency_min_ms", 0),
                    "latency_max_ms": metrics.get("latency_max_ms", 0)
                }
            }
            await self._update_redis_summary(r, camera_id, **summary_update)
            
            logger.info(
                f"[{camera_id}] PORT check: {'UP' if port_ok else 'DOWN'} | "
                f"Success Rate: {metrics['connection_success_rate']:.1f}% | "
                f"Latency: {latency_avg:.2f}ms" if latency_avg else "Latency: None"
            )
            
        except Exception as e:
            logger.error(f"Enhanced PORT check failed for {camera_id}: {e}", exc_info=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._save_to_database_and_create_alert,
                camera_id,
                "PORT_CHECK",
                "FAIL",
                f"PORT check error: {str(e)}",
                {"error": str(e), "ip_address": camera.get("ip", "unknown")},
                False,
                None,
                None
            )
    
    async def _check_connectivity_full(self, r, camera_id: str, camera: dict):
        """Full connectivity check with DATABASE STORAGE"""
        try:
            info = self.ping_checker.extract_rtsp_info(camera["rtsp_url"])
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self.ping_checker.verify_connectivity,
                info["ip"],
                info["port"],
                5
            )
            
            # Determine status
            if result["overall_status"] == "HEALTHY":
                status = "PASS"
            elif result["overall_status"] in ["DEGRADED", "SERVICE_DOWN", "PING_BLOCKED"]:
                status = "WARNING"
            else:
                status = "FAIL"
            
            meta_data = {
                "overall_status": result["overall_status"],
                "network_reachable": result["network_reachable"],
                "service_accessible": result["service_accessible"],
                "ping_metrics": result["ping_metrics"],
                "port_metrics": result["port_metrics"],
                "ip_address": info["ip"],
                "port": info["port"],
                "check_timestamp": datetime.utcnow().isoformat()
            }
            
            event_description = (
                f"Connectivity: {result['overall_status']} | "
                f"Network: {result['network_reachable']} | "
                f"Service: {result['service_accessible']}"
            )
            
            await loop.run_in_executor(
                None,
                self._save_to_database_and_create_alert,
                camera_id,
                "CONNECTIVITY_CHECK",
                status,
                event_description,
                meta_data,
                status == "FAIL",
                "CONNECTIVITY_FAILED" if status == "FAIL" else None,
                f"Full connectivity check failed: {result['overall_status']}" if status == "FAIL" else None
            )
            
            if status != "FAIL":
                await loop.run_in_executor(
                    None,
                    self._resolve_alerts,
                    camera_id,
                    "CONNECTIVITY_FAILED"
                )
            
            summary_update = {
                "Overall_status": result["overall_status"],
                "Network_reachable": result["network_reachable"],
                "Service_accessible": result["service_accessible"],
                "Last_connectivity_check": datetime.now().isoformat(),
                "Connectivity_metrics": {
                    "ping": result["ping_metrics"],
                    "port": result["port_metrics"]
                }
            }
            await self._update_redis_summary(r, camera_id, **summary_update)
            
            logger.info(
                f"[{camera_id}] Connectivity: {result['overall_status']} | "
                f"Network: {result['network_reachable']} | "
                f"Service: {result['service_accessible']}"
            )
            
        except Exception as e:
            logger.error(f"Full connectivity check failed for {camera_id}: {e}", exc_info=True)
    
    async def _update_redis_summary(self, r, camera_id: str, **kwargs):
        """Update Redis summary"""
        try:
            summary = await RedisData.get_summary(r, camera_id)
            if not summary:
                summary = {
                    "Ip_status": "UNKNOWN",
                    "Port_status": "UNKNOWN",
                    "Vision_status": "UNKNOWN",
                    "Overall_status": "UNKNOWN",
                    "Network_reachable": False,
                    "Service_accessible": False,
                    "Alert_active": False,
                    "Last_ip_check": None,
                    "Last_port_check": None,
                    "Last_vision_check": None,
                    "Last_connectivity_check": None,
                    "Ip_metrics": {},
                    "Port_metrics": {},
                    "Connectivity_metrics": {}
                }
            
            summary.update(kwargs)
            await RedisData.update_summary(r, camera_id, summary)
            
        except Exception as e:
            logger.error(f"Failed to update Redis summary for {camera_id}: {e}")


class RedisPoller:
    """Single poller that drains Redis queue"""
    
    def __init__(self, shared_queue: asyncio.Queue):
        self.shared_queue = shared_queue
        self.running = False
        self._task = None
    
    async def start(self):
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Redis poller started")
    
    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Redis poller stopped")
    
    async def _poll_loop(self):
        r = await get_async_redis()
        queue_key = RedisKeys.queue_network()
        
        while self.running:
            try:
                result = await r.brpop(queue_key, timeout=1)
                
                if result:
                    _, task_json = result
                    
                    if not task_json or not task_json.strip():
                        continue
                    
                    try:
                        task = json.loads(task_json)
                        
                        if not self._validate_task(task):
                            continue
                        
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
    """Manages fair worker pool"""
    
    def __init__(self, num_workers: int = 3, concurrency_per_worker: int = 10):
        self.num_workers = num_workers
        self.concurrency_per_worker = concurrency_per_worker
        self.shared_queue = asyncio.Queue()
        self.poller = RedisPoller(self.shared_queue)
        self.workers = []
    
    async def start(self):
        await self.poller.start()
        
        for i in range(self.num_workers):
            worker = NetworkWorker(i, self.shared_queue, self.concurrency_per_worker)
            await worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started 1 poller + {self.num_workers} workers with fair distribution (Enhanced + DB)")
    
    async def stop(self):
        await self.poller.stop()
        await asyncio.gather(*[w.stop() for w in self.workers])
        logger.info("All network workers stopped")


# Global instance
network_worker_pool = NetworkWorkerPool()