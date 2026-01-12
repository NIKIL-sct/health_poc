"""
Camera Health Check Orchestrator
Coordinates ping, port, and vision checks across multiple cameras
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

from ping_checker import PingChecker
from vision_checker import VisionChecker
from vision_storage import VisionStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraOrchestrator:
    """Orchestrates complete camera health check workflow"""
    
    def __init__(self, cameras: List[Dict], baseline_dir: str = "img/baseline"):
        self.cameras = cameras
        self.baseline_dir = baseline_dir
        self.ping_checker = PingChecker()
        self.vision_checker = VisionChecker(baseline_dir)
        self.storage = VisionStorage()
        
    def check_single_camera(self, camera: Dict) -> Dict:
        """
        Complete health check for a single camera
        Returns: Full health status with network and vision results
        """
        cam_id = camera.get('id', 'unknown')
        cam_ip = camera.get('ip', '')
        rtsp_url = camera.get('rtsp_url', '')
        
        result = {
            'camera_id': cam_id,
            'timestamp': datetime.now().isoformat(),
            'network_health': {},
            'vision_health': {},
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Step 1: Network health check
            logger.info(f"[{cam_id}] Starting network health check...")
            network_status = self.ping_checker.check_camera_network(cam_ip, rtsp_url)
            result['network_health'] = network_status
            
            # Step 2: Vision check only if network is healthy
            if network_status.get('status') == 'ONLINE':
                logger.info(f"[{cam_id}] Network healthy. Starting vision check...")
                vision_status = self.vision_checker.check_camera_vision(
                    cam_id, rtsp_url
                )
                result['vision_health'] = vision_status
                
                # Store vision results
                if vision_status:
                    self.storage.store_vision_result(cam_id, vision_status)
                
                # Determine overall status
                if vision_status.get('status') == 'PASS':
                    result['overall_status'] = 'HEALTHY'
                else:
                    result['overall_status'] = 'VISION_ISSUE'
            else:
                logger.warning(f"[{cam_id}] Network check failed. Skipping vision check.")
                result['overall_status'] = 'NETWORK_ISSUE'
                
        except Exception as e:
            logger.error(f"[{cam_id}] Error during health check: {e}")
            result['overall_status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def check_all_cameras(self, max_workers: int = 5) -> List[Dict]:
        """
        Run health checks on all cameras concurrently
        Returns: List of all camera health check results
        """
        logger.info(f"Starting health checks for {len(self.cameras)} cameras...")
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_camera = {
                executor.submit(self.check_single_camera, cam): cam 
                for cam in self.cameras
            }
            
            for future in as_completed(future_to_camera):
                camera = future_to_camera[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[{result['camera_id']}] Check complete: {result['overall_status']}"
                    )
                except Exception as e:
                    logger.error(f"Camera check failed: {e}")
                    results.append({
                        'camera_id': camera.get('id', 'unknown'),
                        'overall_status': 'ERROR',
                        'error': str(e)
                    })
        
        elapsed = time.time() - start_time
        logger.info(f"All health checks completed in {elapsed:.2f}s")
        
        return results
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from check results"""
        summary = {
            'total_cameras': len(results),
            'healthy': 0,
            'network_issues': 0,
            'vision_issues': 0,
            'errors': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        for result in results:
            status = result.get('overall_status')
            if status == 'HEALTHY':
                summary['healthy'] += 1
            elif status == 'NETWORK_ISSUE':
                summary['network_issues'] += 1
            elif status == 'VISION_ISSUE':
                summary['vision_issues'] += 1
            else:
                summary['errors'] += 1
        
        return summary


# Example usage
if __name__ == "__main__":
    # Sample camera configuration
    cameras = [
        {
            'id': 'CAM_001',
            'ip': '192.168.1.100',
            'rtsp_url': 'rtsp://192.168.1.100:554/stream1'
        },
        {
            'id': 'cam13',
            'ip': '192.168.1.202',
            'rtsp_url': 'rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/'
        }
    ]
    
    orchestrator = CameraOrchestrator(cameras)
    results = orchestrator.check_all_cameras()
    summary = orchestrator.generate_summary(results)
    
    print("\n=== Health Check Summary ===")
    print(f"Total Cameras: {summary['total_cameras']}")
    print(f"Healthy: {summary['healthy']}")
    print(f"Network Issues: {summary['network_issues']}")
    print(f"Vision Issues: {summary['vision_issues']}")
    print(f"Errors: {summary['errors']}")
