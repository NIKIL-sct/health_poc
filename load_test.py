"""
Load Test for Vigil-X Redis-Based Camera Health Architecture

• Registers 500 cameras via FastAPI
• Relies on Redis + Scheduler + Worker pools
• Collects system + Redis metrics
• Generates final performance report
"""

import time
import json
import psutil
import requests
from threading import Thread, Lock
from datetime import datetime
from pathlib import Path
import redis

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
REGISTER_ENDPOINT = f"{API_BASE_URL}/health/camera/"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

TARGET_CAMERAS = 13
REGISTER_BATCH_SIZE = 25
REGISTER_DELAY = 0.3

METRIC_INTERVAL = 10  # seconds
REPORT_DIR = Path("logs/load_test")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# -------------------------------------------------------
# BASE CAMERA CONFIGS (13)
# -------------------------------------------------------

BASE_CAMERAS = [
    
    {
        "id":"cam13",
        "ip": "192.168.1.202",
        "port": 554,
        "rtsp_url": "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
    }
]

# -------------------------------------------------------
# METRICS TRACKER
# -------------------------------------------------------

class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.lock = Lock()
        self.data = {
            "test_info": {
                "target_cameras": TARGET_CAMERAS,
                "start_time": datetime.now().isoformat()
            },
            "system_metrics": [],
            "redis_metrics": [],
            "api_health": []
        }
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

    def collect(self):
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024 / 1024

        metric = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - self.start_time, 2),
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "memory_mb": round(mem, 2),
            "threads": proc.num_threads()
        }

        redis_metric = {
            "network_queue": self.redis.llen("queue:network"),
            "vision_queue": self.redis.llen("queue:vision"),
            "active_cameras": self.redis.scard("active:cameras")
        }

        try:
            health = requests.get(HEALTH_ENDPOINT, timeout=2).json()
        except Exception:
            health = {"status": "UNREACHABLE"}

        with self.lock:
            self.data["system_metrics"].append(metric)
            self.data["redis_metrics"].append(redis_metric)
            self.data["api_health"].append(health)

        print(
            f"[{metric['elapsed_sec']}s] "
            f"CPU {metric['cpu_percent']}% | "
            f"MEM {metric['memory_mb']}MB | "
            f"NET_Q {redis_metric['network_queue']} | "
            f"VIS_Q {redis_metric['vision_queue']} | "
            f"CAM {redis_metric['active_cameras']}"
        )

    def save(self):
        self.data["test_info"]["end_time"] = datetime.now().isoformat()
        self.data["test_info"]["duration_sec"] = round(time.time() - self.start_time, 2)

        path = REPORT_DIR / f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"\n✓ Load test report saved: {path}")


# -------------------------------------------------------
# CAMERA REGISTRATION
# -------------------------------------------------------

def generate_cameras():
    cameras = []
    for i in range(TARGET_CAMERAS):
        base = BASE_CAMERAS[i % len(BASE_CAMERAS)]
        cam = {
            "id": f"cam_{i+1:04d}",
            "ip": base["ip"],
            "port": base["port"],
            "rtsp_url": base["rtsp_url"]
        }
        cameras.append(cam)
    return cameras


def register_cameras(cameras):
    print(f"\nRegistering {len(cameras)} cameras...\n")
    for i in range(0, len(cameras), REGISTER_BATCH_SIZE):
        batch = cameras[i:i + REGISTER_BATCH_SIZE]
        for cam in batch:
            try:
                r = requests.post(REGISTER_ENDPOINT, json=cam, timeout=3)
                if r.status_code not in (200, 201, 409):
                    print(f"Failed {cam['id']} → {r.status_code}")
            except Exception as e:
                print(f"Error {cam['id']}: {e}")
        print(f"Registered {min(i + REGISTER_BATCH_SIZE, len(cameras))}/{len(cameras)}")
        time.sleep(REGISTER_DELAY)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def run_load_test():
    print("\n" + "=" * 70)
    print("VIGIL-X LOAD TEST (REDIS ARCHITECTURE)")
    print("=" * 70)

    cameras = generate_cameras()
    tracker = MetricsTracker()

    monitor_running = True

    def monitor():
        while monitor_running:
            tracker.collect()
            time.sleep(METRIC_INTERVAL)

    Thread(target=monitor, daemon=True).start()

    register_cameras(cameras)

    print("\nLoad test running. Press CTRL+C to stop...\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping load test...")
        monitor_running = False
        time.sleep(2)
        tracker.save()


if __name__ == "__main__":
    run_load_test()
