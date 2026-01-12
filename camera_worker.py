import time
import json
import psutil
from datetime import datetime
from pathlib import Path

from ping_checker import PingChecker
from vision_checker import VisionChecker
from vision_storage import VisionStorage


class CameraWorker:
    def __init__(self, camera_config: dict):
        self.camera = camera_config
        self.cam_id = camera_config["id"]

        self.ping = PingChecker()
        self.vision = VisionChecker()
        self.storage = VisionStorage()

        self.state = {
            "ip_up": False,
            "port_up": False,
            "last_ip_check": 0.0,
            "last_port_check": 0.0,
            "last_vision_check": 0.0,
            "alert_active": False,
            "vision": None   
        }


        self.log_path = Path(f"logs/{self.cam_id}.json")
        self.log_path.parent.mkdir(exist_ok=True)

        self.history = []

    def _memory_mb(self):
        return psutil.Process().memory_info().rss / 1024 / 1024

    def _log_event(self, phase):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "ip_status": "up" if self.state["ip_up"] else "down",
            "port_status": "up" if self.state["port_up"] else "down",
            "alert_active": self.state["alert_active"],
            "last_ip_check": self.state["last_ip_check"],
            "last_port_check": self.state["last_port_check"],
            "total_memory_consumed_mb": round(self._memory_mb(), 2)
        }
        self.history.append(entry)

        full_log = {
            "camera_id": self.cam_id,
            "camera_info": self.camera,
            "latest_health_check": entry,
            "health_check_history": self.history[-100:],  # cap history
             "vision_checks": self.state["vision"],
            "last_updated": datetime.now().isoformat()
        }

        with open(self.log_path, "w") as f:
            json.dump(full_log, f, indent=2)

    def run(self):
        print(f"[{self.cam_id}] Worker started")

        while True:
            now = time.time()

            # IP CHECK – every 60 sec
            if now - self.state["last_ip_check"] >= 60:
                result = self.ping.ping_ip(self.camera["ip"])
                self.state["ip_up"] = result
                self.state["last_ip_check"] = now
                self._log_event("ip_check")

            # PORT CHECK – every 15 sec
            if now - self.state["last_port_check"] >= 15:
                info = self.ping.extract_rtsp_info(self.camera["rtsp_url"])
                port_ok = self.ping.check_port(info["ip"], info["port"])
                self.state["port_up"] = port_ok
                self.state["last_port_check"] = now
                self._log_event("port_check")

            # VISION CHECK – every 120 sec (only if net OK)
            if (
                self.state["ip_up"]
                and self.state["port_up"]
                and now - self.state["last_vision_check"] >= 120
            ):
                vision_result = self.vision.check_camera_vision(
                    self.cam_id, self.camera["rtsp_url"]
                )

                # store vision INSIDE worker state
                self.state["vision"] = vision_result

                # update stats only (no files)
                self.storage.store_vision_result(self.cam_id, vision_result)

                self.state["alert_active"] = vision_result["status"] != "PASS"
                self.state["last_vision_check"] = now
                self._log_event("vision_check")


            time.sleep(1)
