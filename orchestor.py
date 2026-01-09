"""
Camera Health Monitoring Orchestrator
Manages the execution of health checks and persists results to comprehensive logs.
Features:
- Per-camera health logs with complete metrics
- Global snapshot with aggregated statistics
- Memory consumption tracking per camera and overall
- Professional-grade logging and error handling
"""

import asyncio  
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from ping_checker import CameraHealthService, StateManager

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
LOGS_DIR = Path("logs")
HEALTH_CHECK_INTERVAL = 30  # seconds between orchestrator cycles

# Camera Configuration
CAMERAS_CONFIG = [
(
        "cam1",
        "154.210.224.77",
        8011,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8011/unicaststream/1"
    ),
    (
        "cam2",
        "154.210.224.77",
        8012,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8012/unicaststream/1"
    ),
    (
        "cam3",
        "154.210.224.77",
        8013,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8013/unicaststream/1"
    ),
    (
        "cam4",
        "154.210.224.77",
        8014,
        "rtmp://media5.ambicam.com:1938/live/viprepaid"
    ),
    (
        "cam5",
        "154.210.224.77",
        8015,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8015/unicaststream/1"
    ),
    (
        "cam6",
        "154.210.224.77",
        8016,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8016/unicaststream/1"
    ),
    (
        "cam7",
        "154.210.224.77",
        8017,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8017/unicaststream/1"
    ),
    (
        "cam8",
        "154.210.224.77",
        8018,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8018/unicaststream/1"
    ),
    (
        "cam9",
        "154.210.224.77",
        8019,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8019/unicaststream/1"
    ),
    (
        "cam10",
        "154.210.224.77",
        8020,
        "rtsp://viewer:Matrix%40123@154.210.224.77:8020/unicaststream/1"
    ),
    (
        "cam11",
        "154.210.224.77",
        8021,
        "rtsp://viewer:Vigil_x1@154.210.224.77:8021/cam/realmonitor?channel=1&subtype=0"
    ),
    (
        "cam12",
        "154.210.224.77",
        8022,
        "rtsp://viewer:Vigil_x1@154.210.224.77:8022/cam/realmonitor?channel=1&subtype=0"
    ),
    (
        "cam13", 
        "192.168.1.202", 
        554,
        "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
    ), 
]

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "orchestrator.log")
    ]
)
logger = logging.getLogger("orchestrator")

# -------------------------------------------------------------------
# Health Snapshot Generator
# -------------------------------------------------------------------
class HealthSnapshotGenerator:
    """
    Generates comprehensive health snapshots from service state.
    Includes per-camera memory metrics and aggregated statistics.
    """
    
    @staticmethod
    async def generate(
        state_manager: StateManager, 
        metrics_collector
    ) -> Dict[str, Any]:
        """
        Generate a complete health snapshot with memory metrics.
        
        Args:
            state_manager: Camera state manager instance
            metrics_collector: Metrics collector instance
            
        Returns:
            Dictionary containing complete health status and metrics
        """
        cameras = await state_manager.get_all()
        stats = await metrics_collector.get_stats()
        
        # Calculate aggregated memory consumption
        total_camera_memory = sum(
            cam.total_memory_consumed_mb for cam in cameras
        )
        
        snapshot = {
            "summary": {
                "total_cameras": len(cameras),
                "cameras_up": sum(
                    1 for c in cameras 
                    if c.last_ip_status.value == "up"
                ),
                "cameras_down": sum(
                    1 for c in cameras 
                    if c.last_ip_status.value == "down"
                ),
                "cameras_alerted": sum(
                    1 for c in cameras 
                    if c.alert_active
                ),
                "cameras_unknown": sum(
                    1 for c in cameras 
                    if c.last_ip_status.value == "unknown"
                )
            },
            "performance_metrics": {
                "health_checks": {
                    "total_checks": stats.total_checks,
                    "ip_checks": stats.total_ip_checks,
                    "port_checks": stats.total_port_checks
                },
                "latency": {
                    "avg_overall_ms": round(stats.avg_overall_latency_ms, 2),
                    "avg_ip_ms": round(stats.avg_ip_latency_ms, 2),
                    "avg_port_ms": round(stats.avg_port_latency_ms, 2),
                    "min_ms": round(stats.min_latency_ms, 2) 
                        if stats.min_latency_ms != float('inf') else 0,
                    "max_ms": round(stats.max_latency_ms, 2),
                    "p50_ms": round(stats.p50_latency_ms, 2),
                    "p95_ms": round(stats.p95_latency_ms, 2),
                    "p99_ms": round(stats.p99_latency_ms, 2)
                },
                "memory_consumption": {
                    "total_health_checks_mb": round(
                        stats.total_memory_consumed_mb, 2
                    ),
                    "avg_per_health_check_mb": round(
                        stats.avg_memory_per_check_mb, 4
                    ),
                    "total_all_cameras_mb": round(
                        total_camera_memory, 2
                    ),
                    "overall_system_mb": round(
                        stats.total_memory_consumed_mb + total_camera_memory, 2
                    )
                },
                "concurrency": {
                    "max_concurrent": stats.max_concurrent_checks,
                    "avg_concurrent": round(stats.avg_concurrent_checks, 2)
                }
            },
            "per_camera_metrics": [
                {
                    "camera_id": cam.camera_id,
                    "memory_consumed_mb": round(
                        cam.total_memory_consumed_mb, 2
                    )
                }
                for cam in cameras
            ]
        }
        
        return snapshot

# -------------------------------------------------------------------
# Global Snapshot Manager
# -------------------------------------------------------------------
class GlobalSnapshotManager:
    """
    Manages global snapshot persistence.
    Stores aggregated metrics without individual camera logs.
    """
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.global_snapshot_file = logs_dir / "global_snapshot.json"
        self.service = None  # Will be set by orchestrator
        
    async def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Save global snapshot with system-wide metrics.
        
        Args:
            snapshot: Complete snapshot dictionary
            
        Raises:
            IOError: If file write fails
        """
        try:
            snapshot_data = {
                "timestamp": datetime.now().isoformat(),
                "orchestrator_version": "2.0.0",
                "summary": snapshot["summary"],
                "performance_metrics": snapshot["performance_metrics"],
                "per_camera_metrics": snapshot["per_camera_metrics"]
            }
            
            with open(self.global_snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"[SNAPSHOT] Global snapshot saved: "
                f"{self.global_snapshot_file.name}"
            )
            
            # Also update per-camera health check logs
            await self._update_camera_health_logs(snapshot)
            
        except Exception as e:
            logger.error(
                f"[SNAPSHOT] Failed to save global snapshot: {str(e)}",
                exc_info=True
            )
            raise
    
    async def _update_camera_health_logs(self, snapshot: Dict[str, Any]) -> None:
        """
        Update individual camera health logs with latest check data.
        
        Args:
            snapshot: Complete snapshot dictionary
        """
        try:
            cameras = await self.service.state.get_all()
            
            for cam in cameras:
                health_file = self.logs_dir / f"{cam.camera_id}_health.json"
                
                # Load or create camera log
                if health_file.exists():
                    try:
                        with open(health_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = self._create_camera_log_structure(cam)
                else:
                    data = self._create_camera_log_structure(cam)
                
                # Update camera info
                data["camera_info"] = {
                    "camera_id": cam.camera_id,
                    "ip": cam.ip,
                    "onvif_port": cam.onvif_port,
                    "rtsp_port": cam.rtsp_port,
                    "rtsp_url": cam.rtsp_url
                }
                
                # Update latest health check status
                data["latest_health_check"] = {
                    "timestamp": datetime.now().isoformat(),
                    "phase": cam.phase.value,
                    "ip_status": cam.last_ip_status.value,
                    "port_status": cam.last_port_status.value,
                    "alert_active": cam.alert_active,
                    "last_ip_check": cam.last_ip_check_at,
                    "last_port_check": cam.last_port_check_at,
                    "total_memory_consumed_mb": round(cam.total_memory_consumed_mb, 2)
                }
                
                # Append to health check history
                if "health_check_history" not in data:
                    data["health_check_history"] = []
                
                data["health_check_history"].append(data["latest_health_check"])
                
                # Keep only last 100 entries
                if len(data["health_check_history"]) > 100:
                    data["health_check_history"] = data["health_check_history"][-100:]
                
                # Write updated log
                with open(health_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"[SNAPSHOT] Updated {len(cameras)} camera health logs")
            
        except Exception as e:
            logger.error(
                f"[SNAPSHOT] Failed to update camera health logs: {str(e)}",
                exc_info=True
            )
    
    def _create_camera_log_structure(self, cam) -> Dict[str, Any]:
        """Create initial camera log structure"""
        return {
            "camera_id": cam.camera_id,
            "camera_info": {},
            "latest_health_check": None,
            "health_check_history": [],
            "vision_checks": [],
            "vision_statistics": {
                "total_checks": 0,
                "total_passed": 0,
                "total_failed": 0,
                "pass_rate": 0.0,
                "average_similarity": 0.0,
                "average_check_duration_ms": 0.0,
                "total_memory_consumed_mb": 0.0,
                "average_memory_per_check_mb": 0.0,
                "first_check": None,
                "last_check": None
            }
        }

# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------
class HealthMonitorOrchestrator:
    """
    Main orchestrator for camera health monitoring service.
    Coordinates service execution, metrics collection, and log persistence.
    
    Responsibilities:
    - Initialize and manage camera health service
    - Periodically generate and save health snapshots
    - Track system-wide memory consumption
    - Handle graceful shutdown
    """
    
    def __init__(self, cameras_config: List[tuple]):
        """
        Initialize orchestrator with camera configuration.
        
        Args:
            cameras_config: List of tuples (camera_id, ip, port, rtsp_url)
        """
        self.cameras_config = cameras_config
        self.service: Optional[CameraHealthService] = None
        self.snapshot_manager = GlobalSnapshotManager(LOGS_DIR)
        self.snapshot_generator = HealthSnapshotGenerator()
        self.running = False
    
    async def initialize_service(self) -> None:
        """
        Initialize the camera health service with configured cameras.
        
        Raises:
            RuntimeError: If service initialization fails
        """
        logger.info("[INIT] Initializing Camera Health Service...")
        
        try:
            self.service = CameraHealthService()
            
            # Link service to snapshot manager for camera log updates
            self.snapshot_manager.service = self.service
            
            # Register cameras from configuration
            for camera_id, ip, port, rtsp_url in self.cameras_config:
                await self.service.state.register_camera(
                    camera_id, ip, port, rtsp_url
                )
            
            logger.info(
                f"[INIT] Successfully registered {len(self.cameras_config)} cameras"
            )
            
        except Exception as e:
            logger.error(f"[INIT] Service initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize service: {str(e)}")
    
    async def snapshot_loop(self) -> None:
        """
        Periodically capture and save health snapshots.
        Runs continuously until service shutdown.
        """
        logger.info("[SNAPSHOT] Starting snapshot loop...")
        
        while self.running:
            try:
                # Generate comprehensive snapshot
                snapshot = await self.snapshot_generator.generate(
                    self.service.state,
                    self.service.metrics
                )
                
                # Save global snapshot and update per-camera logs
                await self.snapshot_manager.save_snapshot(snapshot)
                
            except Exception as e:
                logger.error(
                    f"[SNAPSHOT] Snapshot loop error: {str(e)}", 
                    exc_info=True
                )
            
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    async def start(self) -> None:
        """
        Start the orchestrator and monitoring service.
        Runs until interrupted or error occurs.
        """
        self._print_startup_banner()
        
        self.running = True
        
        try:
            # Initialize service
            await self.initialize_service()
            
            # Start both service and snapshot loop concurrently
            await asyncio.gather(
                self.service.start(),
                self.snapshot_loop()
            )
            
        except asyncio.CancelledError:
            logger.info("[SHUTDOWN] Orchestrator cancelled")
        except Exception as e:
            logger.error(
                f"[ERROR] Orchestrator error: {str(e)}", 
                exc_info=True
            )
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """
        Perform graceful shutdown with final snapshot.
        Ensures all data is persisted before exit.
        """
        logger.info("[SHUTDOWN] Shutting down orchestrator...")
        self.running = False
        
        # Save final snapshot
        if self.service:
            try:
                final_snapshot = await self.snapshot_generator.generate(
                    self.service.state,
                    self.service.metrics
                )
                
                await self.snapshot_manager.save_snapshot(final_snapshot)
                
                logger.info("[SHUTDOWN] Final snapshot saved")
            except Exception as e:
                logger.error(
                    f"[SHUTDOWN] Failed to save final snapshot: {str(e)}"
                )
        
        logger.info("[SHUTDOWN] Orchestrator shutdown complete")
    
    def _print_startup_banner(self) -> None:
        """Print startup banner with system information"""
        logger.info("=" * 70)
        logger.info("CAMERA HEALTH MONITORING ORCHESTRATOR v2.0")
        logger.info("=" * 70)
        logger.info(f"Cameras configured: {len(self.cameras_config)}")
        logger.info(f"Logs directory: {LOGS_DIR.absolute()}")
        logger.info(f"Snapshot interval: {HEALTH_CHECK_INTERVAL}s")
        logger.info("=" * 70)

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
async def main() -> None:
    """
    Main entry point for the orchestrator application.
    Handles initialization, execution, and cleanup.
    """
    orchestrator = HealthMonitorOrchestrator(CAMERAS_CONFIG)
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("[MAIN] Received keyboard interrupt")
    except Exception as e:
        logger.error(f"[MAIN] Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[MAIN] Program terminated by user")