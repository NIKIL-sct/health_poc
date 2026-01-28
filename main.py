import time
from threading import Thread
from camera_worker import CameraWorker

# ------------------------------------------------------------------
# CAMERA CONFIGURATION
# ------------------------------------------------------------------
CAMERAS = [
    {
        "id": "cam13",
        "ip": "192.168.1.202",
        "rtsp_port": 554,
        "onvif_port": 554,
        "rtsp_url": "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
    }
]

# ------------------------------------------------------------------
# START CAMERA WORKERS
# ------------------------------------------------------------------
threads = []

for cam in CAMERAS:
    worker = CameraWorker(cam)
    t = Thread(target=worker.run, daemon=True)
    t.start()
    threads.append(t)

print(" Camera health monitoring started")

# ------------------------------------------------------------------
# KEEP MAIN PROCESS ALIVE
# ------------------------------------------------------------------
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
