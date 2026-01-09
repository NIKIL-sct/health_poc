"""
Camera Health Monitoring Service with Latency Tracking and Memory Profiling
Updated workflow:
- IP check every 60 seconds (CONTINUOUS)
- ONVIF port check every 20 seconds (CONTINUOUS)
- Two consecutive successful port checks are required
- After 2 successful port checks → continue IP/Port checks + schedule vision check for 2 minutes later
- Vision check runs at 2 minute mark while IP/Port checks continue
- If any port check fails → alert immediately (no further IP checks)
- Comprehensive memory tracking for all operations
"""

import asyncio
import logging
import time
import tracemalloc
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import statistics

from vision_checker import VisionChecker
from vision_storage import store_vision_result

# -------------------------------------------------------------------
# Timing Configuration (seconds)
# -------------------------------------------------------------------
IP_CHECK_INTERVAL = 60
PORT_CHECK_INTERVAL = 20
VISION_CHECK_DELAY = 60  # 2 minutes delay after successful port checks
CYCLE_DURATION = 900
SCHEDULER_TICK = 5

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("camera-health")

# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------
class HealthStatus(Enum):
    """Health status enumeration"""
    UNKNOWN = "unknown"
    UP = "up"
    DOWN = "down"

class Phase(Enum):
    """Monitoring phase enumeration"""
    IP_CHECK = "ip_check"
    PORT_CHECK_1 = "port_check_1"
    PORT_CHECK_2 = "port_check_2"
    CONTINUOUS_MONITORING = "continuous_monitoring"  # IP + Port checks continue
    ALERTED = "alerted"

# -------------------------------------------------------------------
# Metrics Data Classes
# -------------------------------------------------------------------
@dataclass
class CheckMetrics:
    """Metrics for a single health check"""
    camera_id: str
    check_type: str  # 'ip' or 'port'
    start_time: float
    end_time: float
    duration_ms: float
    status: HealthStatus
    concurrent_checks: int
    memory_consumed_mb: float

@dataclass
class LatencyStats:
    """Aggregated latency and performance statistics"""
    total_checks: int = 0
    total_ip_checks: int = 0
    total_port_checks: int = 0
    avg_ip_latency_ms: float = 0.0
    avg_port_latency_ms: float = 0.0
    avg_overall_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_concurrent_checks: int = 0
    avg_concurrent_checks: float = 0.0
    total_memory_consumed_mb: float = 0.0
    avg_memory_per_check_mb: float = 0.0

# -------------------------------------------------------------------
# Metrics Collector with Memory Tracking
# -------------------------------------------------------------------
class MetricsCollector:
    """Collects and analyzes latency and memory metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: deque[CheckMetrics] = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._concurrent_checks = 0

    async def record_check(self, metric: CheckMetrics):
        """Record a check metric with memory information"""
        async with self._lock:
            self.metrics.append(metric)

    async def increment_concurrent(self) -> int:
        """Increment concurrent check counter"""
        async with self._lock:
            self._concurrent_checks += 1
            return self._concurrent_checks

    async def decrement_concurrent(self) -> int:
        """Decrement concurrent check counter"""
        async with self._lock:
            self._concurrent_checks -= 1
            return self._concurrent_checks

    async def get_stats(self) -> LatencyStats:
        """Calculate and return aggregated statistics"""
        async with self._lock:
            if not self.metrics:
                return LatencyStats()

            stats = LatencyStats()
            stats.total_checks = len(self.metrics)

            ip_latencies, port_latencies, all_latencies = [], [], []
            concurrent_counts, memory_values = [], []

            for m in self.metrics:
                all_latencies.append(m.duration_ms)
                concurrent_counts.append(m.concurrent_checks)
                memory_values.append(m.memory_consumed_mb)
                
                if m.check_type == 'ip':
                    ip_latencies.append(m.duration_ms)
                    stats.total_ip_checks += 1
                else:
                    port_latencies.append(m.duration_ms)
                    stats.total_port_checks += 1

            # Calculate latency averages
            if ip_latencies:
                stats.avg_ip_latency_ms = statistics.mean(ip_latencies)
            if port_latencies:
                stats.avg_port_latency_ms = statistics.mean(port_latencies)
            if all_latencies:
                stats.avg_overall_latency_ms = statistics.mean(all_latencies)
                stats.min_latency_ms = min(all_latencies)
                stats.max_latency_ms = max(all_latencies)
                
                sorted_latencies = sorted(all_latencies)
                stats.p50_latency_ms = self._percentile(sorted_latencies, 50)
                stats.p95_latency_ms = self._percentile(sorted_latencies, 95)
                stats.p99_latency_ms = self._percentile(sorted_latencies, 99)

            # Calculate concurrency stats
            if concurrent_counts:
                stats.max_concurrent_checks = max(concurrent_counts)
                stats.avg_concurrent_checks = statistics.mean(concurrent_counts)

            # Calculate memory stats
            if memory_values:
                stats.total_memory_consumed_mb = sum(memory_values)
                stats.avg_memory_per_check_mb = statistics.mean(memory_values)

            return stats

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data"""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    async def print_stats(self):
        """Print formatted statistics to console"""
        stats = await self.get_stats()
        
        print("\n" + "="*70)
        print("HEALTH CHECK PERFORMANCE STATISTICS")
        print("="*70)
        print(f"Total Checks:        {stats.total_checks}")
        print(f"  - IP Checks:       {stats.total_ip_checks}")
        print(f"  - Port Checks:     {stats.total_port_checks}")
        print()
        print("LATENCY (milliseconds):")
        print(f"  Average Overall:   {stats.avg_overall_latency_ms:.2f} ms")
        print(f"  Average IP:        {stats.avg_ip_latency_ms:.2f} ms")
        print(f"  Average Port:      {stats.avg_port_latency_ms:.2f} ms")
        print(f"  P50 (median):      {stats.p50_latency_ms:.2f} ms")
        print(f"  P95:               {stats.p95_latency_ms:.2f} ms")
        print(f"  P99:               {stats.p99_latency_ms:.2f} ms")
        print()
        print("MEMORY CONSUMPTION:")
        print(f"  Total:             {stats.total_memory_consumed_mb:.2f} MB")
        print(f"  Average/Check:     {stats.avg_memory_per_check_mb:.4f} MB")
        print()
        print("CONCURRENCY:")
        print(f"  Max Concurrent:    {stats.max_concurrent_checks}")
        print(f"  Avg Concurrent:    {stats.avg_concurrent_checks:.2f}")
        print("="*70 + "\n")

# -------------------------------------------------------------------
# Camera State
# -------------------------------------------------------------------
@dataclass
class CameraState:
    """State container for a single camera"""
    camera_id: str
    ip: str
    onvif_port: int
    rtsp_port: int
    rtsp_url: str = ""
    phase: Phase = Phase.IP_CHECK
    cycle_started_at: float = 0.0
    last_ip_check_at: float = 0.0
    last_port_check_at: float = 0.0
    vision_ready_at: float = 0.0  # Time when vision check becomes ready
    vision_check_running: bool = False  # Flag to prevent duplicate vision checks
    port_check_success_count: int = 0  # Count consecutive successful port checks
    last_ip_status: HealthStatus = HealthStatus.UNKNOWN
    last_port_status: HealthStatus = HealthStatus.UNKNOWN
    alert_active: bool = False
    total_memory_consumed_mb: float = 0.0

    def __post_init__(self):
        if self.cycle_started_at == 0.0:
            self.cycle_started_at = time.time()

# -------------------------------------------------------------------
# State Manager
# -------------------------------------------------------------------
class StateManager:
    """Manages camera state in memory with thread-safe operations"""
    
    def __init__(self):
        self._cameras: Dict[str, CameraState] = {}
        self._lock = asyncio.Lock()

    async def register_camera(
        self, 
        camera_id: str, 
        ip: str, 
        port: int, 
        rtsp_url: str
    ):
        """Register a new camera for monitoring"""
        async with self._lock:
            self._cameras[camera_id] = CameraState(
                camera_id=camera_id,
                ip=ip,
                onvif_port=port,
                rtsp_port=port,
                rtsp_url=rtsp_url
            )
            logger.info(f"[REGISTER] Camera {camera_id} ({ip}:{port})")

    async def get_all(self) -> List[CameraState]:
        """Get all registered cameras"""
        async with self._lock:
            return list(self._cameras.values())

    async def mark_alerted(self, cam: CameraState):
        """Mark camera as alerted"""
        async with self._lock:
            cam.phase = Phase.ALERTED
            cam.alert_active = True

    async def update_memory_consumption(
        self, 
        cam: CameraState, 
        memory_mb: float
    ):
        """Update camera's total memory consumption"""
        async with self._lock:
            cam.total_memory_consumed_mb += memory_mb

# -------------------------------------------------------------------
# Health Checker
# -------------------------------------------------------------------
class HealthChecker:
    """Performs actual health checks on cameras"""
    
    @staticmethod
    async def check_ip(ip: str, timeout: int = 2) -> HealthStatus:
        """
        Check if IP is reachable using ping.
        
        Args:
            ip: IP address to check
            timeout: Ping timeout in seconds
            
        Returns:
            HealthStatus indicating reachability
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ping", "-c", "1", "-W", str(timeout), ip,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            returncode = await process.wait()
            return HealthStatus.UP if returncode == 0 else HealthStatus.DOWN
        except Exception as e:
            logger.debug(f"[IP CHECK] Error for {ip}: {str(e)}")
            return HealthStatus.DOWN

    @staticmethod
    async def check_port(ip: str, port: int, timeout: int = 3) -> HealthStatus:
        """
        Check if specific port is reachable.
        
        Args:
            ip: IP address
            port: Port number
            timeout: Connection timeout in seconds
            
        Returns:
            HealthStatus indicating port reachability
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), 
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return HealthStatus.UP
        except Exception as e:
            logger.debug(f"[PORT CHECK] Error for {ip}:{port}: {str(e)}")
            return HealthStatus.DOWN

# -------------------------------------------------------------------
# Alert Manager
# -------------------------------------------------------------------
class AlertManager:
    """Manages alert generation and logging"""
    
    @staticmethod
    async def raise_alert(cam: CameraState, reason: str):
        """
        Raise an alert for a camera issue.
        
        Args:
            cam: Camera state
            reason: Alert reason/description
        """
        logger.warning({
            "event": "ALERT",
            "camera_id": cam.camera_id,
            "ip": cam.ip,
            "onvif_port": cam.onvif_port,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

# -------------------------------------------------------------------
# Check Executor with Memory Profiling
# -------------------------------------------------------------------
class CheckExecutor:
    """Executes health checks with concurrency control and metrics"""
    
    def __init__(self, max_concurrent: int, metrics: MetricsCollector):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.metrics = metrics

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB"""
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        return 0.0

    async def run_ip_check(self, cam: CameraState):
        """Execute IP health check with metrics tracking"""
        start = time.time()
        concurrent = await self.metrics.increment_concurrent()
        
        # Start memory tracking
        tracemalloc.start()
        start_mem = self._measure_memory()
        
        try:
            async with self.sem:
                cam.last_ip_status = await HealthChecker.check_ip(cam.ip)
                cam.last_ip_check_at = time.time()
                
                end = time.time()
                duration_ms = (end - start) * 1000
                
                # Measure memory
                end_mem = self._measure_memory()
                memory_consumed = max(0, end_mem - start_mem)
                
                # Record metrics
                metric = CheckMetrics(
                    camera_id=cam.camera_id,
                    check_type='ip',
                    start_time=start,
                    end_time=end,
                    duration_ms=duration_ms,
                    status=cam.last_ip_status,
                    concurrent_checks=concurrent,
                    memory_consumed_mb=round(memory_consumed, 4)
                )
                await self.metrics.record_check(metric)
                
                logger.info(
                    f"[IP] {cam.camera_id} → {cam.last_ip_status.value} "
                    f"({duration_ms:.2f}ms)"
                )
        finally:
            tracemalloc.stop()
            await self.metrics.decrement_concurrent()

    async def run_port_check(self, cam: CameraState):
        """Execute port health check with metrics tracking"""
        start = time.time()
        concurrent = await self.metrics.increment_concurrent()
        
        # Start memory tracking
        tracemalloc.start()
        start_mem = self._measure_memory()
        
        try:
            async with self.sem:
                cam.last_port_status = await HealthChecker.check_port(
                    cam.ip, 
                    cam.onvif_port
                )
                cam.last_port_check_at = time.time()
                
                end = time.time()
                duration_ms = (end - start) * 1000
                
                # Measure memory
                end_mem = self._measure_memory()
                memory_consumed = max(0, end_mem - start_mem)
                
                # Record metrics
                metric = CheckMetrics(
                    camera_id=cam.camera_id,
                    check_type='port',
                    start_time=start,
                    end_time=end,
                    duration_ms=duration_ms,
                    status=cam.last_port_status,
                    concurrent_checks=concurrent,
                    memory_consumed_mb=round(memory_consumed, 4)
                )
                await self.metrics.record_check(metric)
                
                logger.info(
                    f"[PORT] {cam.camera_id} → {cam.last_port_status.value} "
                    f"({duration_ms:.2f}ms)"
                )
        finally:
            tracemalloc.stop()
            await self.metrics.decrement_concurrent()

# -------------------------------------------------------------------
# Global Scheduler
# -------------------------------------------------------------------
class GlobalScheduler:
    """Global scheduler for coordinating health checks across all cameras"""
    
    def __init__(
        self, 
        state: StateManager, 
        executor: CheckExecutor, 
        metrics: MetricsCollector
    ):
        self.state = state
        self.executor = executor
        self.metrics = metrics
        self.running = True
        self.vision = VisionChecker()

    async def start(self):
        """Start the scheduling loop"""
        while self.running:
            cams = await self.state.get_all()
            now = time.time()

            for cam in cams:
                if cam.phase == Phase.ALERTED:
                    continue

                # Phase: IP_CHECK
                if cam.phase == Phase.IP_CHECK:
                    if now - cam.last_ip_check_at >= IP_CHECK_INTERVAL:
                        await self.executor.run_ip_check(cam)
                        
                        if cam.last_ip_status == HealthStatus.DOWN:
                            await AlertManager.raise_alert(
                                cam, 
                                "IP unreachable"
                            )
                            await self.state.mark_alerted(cam)
                        else:
                            cam.phase = Phase.PORT_CHECK_1
                            cam.last_port_check_at = 0
                            cam.port_check_success_count = 0

                # Phase: PORT_CHECK_1
                elif cam.phase == Phase.PORT_CHECK_1:
                    if now - cam.last_port_check_at >= PORT_CHECK_INTERVAL:
                        await self.executor.run_port_check(cam)
                        
                        if cam.last_port_status == HealthStatus.DOWN:
                            await AlertManager.raise_alert(
                                cam, 
                                "ONVIF port unreachable"
                            )
                            await self.state.mark_alerted(cam)
                        else:
                            cam.port_check_success_count = 1
                            cam.phase = Phase.PORT_CHECK_2

                # Phase: PORT_CHECK_2
                elif cam.phase == Phase.PORT_CHECK_2:
                    if now - cam.last_port_check_at >= PORT_CHECK_INTERVAL:
                        await self.executor.run_port_check(cam)

                        if cam.last_port_status == HealthStatus.DOWN:
                            await AlertManager.raise_alert(
                                cam, 
                                "ONVIF port unreachable"
                            )
                            await self.state.mark_alerted(cam)
                        else:
                            # Two successful port checks completed
                            cam.port_check_success_count = 2
                            cam.phase = Phase.CONTINUOUS_MONITORING
                            cam.vision_ready_at = now + VISION_CHECK_DELAY
                            cam.last_ip_check_at = now  # Reset for continuous checks
                            logger.info(
                                f"[CONTINUOUS] {cam.camera_id} entering continuous monitoring. "
                                f"Vision check scheduled in {VISION_CHECK_DELAY}s"
                            )

                # Phase: CONTINUOUS_MONITORING (IP + Port checks continue, vision check at 2min mark)
                elif cam.phase == Phase.CONTINUOUS_MONITORING:
                    # Continue IP checks every 60 seconds
                    if now - cam.last_ip_check_at >= IP_CHECK_INTERVAL:
                        await self.executor.run_ip_check(cam)
                        
                        if cam.last_ip_status == HealthStatus.DOWN:
                            await AlertManager.raise_alert(
                                cam, 
                                "IP unreachable during continuous monitoring"
                            )
                            await self.state.mark_alerted(cam)
                            continue
                    
                    # Continue Port checks every 20 seconds
                    if now - cam.last_port_check_at >= PORT_CHECK_INTERVAL:
                        await self.executor.run_port_check(cam)
                        
                        if cam.last_port_status == HealthStatus.DOWN:
                            await AlertManager.raise_alert(
                                cam, 
                                "ONVIF port unreachable during continuous monitoring"
                            )
                            await self.state.mark_alerted(cam)
                            continue
                    
                    # Run vision check at 2 minute mark (only once)
                    if now >= cam.vision_ready_at and not cam.vision_check_running:
                        cam.vision_check_running = True
                        # Run vision check asynchronously so IP/Port checks continue
                        asyncio.create_task(self._run_vision_check(cam))

            await asyncio.sleep(SCHEDULER_TICK)

    async def _run_vision_check(self, cam: CameraState):
        """
        Execute vision check and handle results.
        Runs asynchronously while IP/Port checks continue.
        
        Args:
            cam: Camera state
        """
        rtsp_url = cam.rtsp_url

        try:
            logger.info(f"[VISION] Starting check for {cam.camera_id}")
            
            vision_result = self.vision.run(
                camera_id=cam.camera_id,
                rtsp_url=rtsp_url
            )

            # Store result (this handles ALL vision log storage)
            store_vision_result(cam.camera_id, vision_result)

            # Update camera memory consumption
            memory_consumed = vision_result["memory_metrics"]["memory_consumed_mb"]
            await self.state.update_memory_consumption(cam, memory_consumed)

            # Check status
            if vision_result["status"] == "FAIL":
                summary = vision_result["semantic_analysis"]["summary"]
                await AlertManager.raise_alert(
                    cam,
                    f"Vision check failed: {summary}"
                )
                await AlertManager.raise_alert(
                    cam,
                    f"Vision issue detected: {summary}"
                )

            else:
                # Success - restart IP cycle
                logger.info(
                    f"[VISION] {cam.camera_id} PASSED - "
                    f"restarting IP check cycle"
                )
                cam.phase = Phase.IP_CHECK
                cam.last_ip_check_at = 0
                cam.vision_ready_at = 0
                cam.vision_check_running = False
                cam.port_check_success_count = 0

        except Exception as e:
            logger.error(f"[VISION] Error for {cam.camera_id}: {str(e)}")
            await AlertManager.raise_alert(
                cam,
                f"Vision check error: {str(e)}"
            )
            
        
        finally:
            # ALWAYS restart cycle
            cam.phase = Phase.IP_CHECK
            cam.last_ip_check_at = 0
            cam.last_port_check_at = 0
            cam.vision_ready_at = 0
            cam.vision_check_running = False
            cam.port_check_success_count = 0

            logger.info(
                f"[VISION] {cam.camera_id} completed. "
                f"Restarting monitoring cycle."
            )


# -------------------------------------------------------------------
# Service
# -------------------------------------------------------------------
class CameraHealthService:
    """Main camera health monitoring service"""
    
    def __init__(self):
        self.state = StateManager()
        self.metrics = MetricsCollector(max_history=5000)
        self.executor = CheckExecutor(max_concurrent=100, metrics=self.metrics)
        self.scheduler = GlobalScheduler(self.state, self.executor, self.metrics)

    async def start(self):
        """Start the health monitoring service"""
        logger.info(
            "Camera Health Service started with comprehensive monitoring"
        )
        await self.scheduler.start()