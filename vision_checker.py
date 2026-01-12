"""
Vision Health Checker for Cameras
Performs high-accuracy semantic anomaly detection and image quality analysis
"""

import cv2
import numpy as np
import logging
import time
import psutil
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionChecker:
    """High-accuracy vision health analysis using baseline comparison"""
    
    def __init__(self, baseline_dir: str = "img/baseline", capture_dir: str = "img/captures"):
        self.baseline_dir = baseline_dir
        self.capture_dir = capture_dir

        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Quality thresholds (tuned for real-world accuracy)
        self.BRIGHTNESS_MIN = 30
        self.BRIGHTNESS_MAX = 230
        self.BLUR_THRESHOLD = 80
        self.EDGE_COLLAPSE_RATIO = 0.25  # 75% edge loss = suspicious

        # SSIM thresholds - more lenient for real-world conditions
        self.SSIM_CRITICAL = 0.50   # Below this = severe issue
        self.SSIM_WARNING = 0.65    # Below this = potential issue
        self.SSIM_GOOD = 0.75       # Above this = healthy
        
        # Difference thresholds - adjusted for lighting/compression variations
        self.DIFF_SEVERE = 50       # >50% difference = severe obstruction
        self.DIFF_MODERATE = 35     # >35% difference = partial obstruction
        self.DIFF_DISPLACEMENT = 25 # >25% difference = possible displacement
        
    def capture_frame(self, rtsp_url: str, timeout: int = 10) -> Optional[np.ndarray]:
        """
        Capture a single frame from RTSP stream
        Returns: Frame as numpy array or None if failed
        """
        cap = None
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    return frame
                time.sleep(0.1)
            
            logger.error(f"Failed to capture frame within {timeout}s")
            return None
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
        finally:
            if cap:
                cap.release()
                
    def save_frame(self, frame: np.ndarray, camera_id: str, tag: str = "capture"):
        """Save captured frame for debugging/logging"""
        cam_dir = os.path.join(self.capture_dir, camera_id)
        os.makedirs(cam_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{timestamp}.jpg"
        path = os.path.join(cam_dir, filename)

        cv2.imwrite(path, frame)
        logger.info(f"[{camera_id}] Frame saved: {path}")

    def load_baseline(self, camera_id: str) -> Optional[np.ndarray]:
        """Load baseline image for comparison"""
        baseline_path = os.path.join(self.baseline_dir, f"{camera_id}.png")
        
        if not os.path.exists(baseline_path):
            logger.warning(f"No baseline found for {camera_id}")
            return None
        
        try:
            baseline = cv2.imread(baseline_path)
            return baseline
        except Exception as e:
            logger.error(f"Baseline load error: {e}")
            return None
    
    def analyze_brightness(self, frame: np.ndarray) -> Dict:
        """Analyze frame brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        status = "PASS"
        message = "Brightness normal"
        
        if mean_brightness < self.BRIGHTNESS_MIN:
            status = "FAIL"
            message = "Too dark - possible obstruction or lighting issue"
        elif mean_brightness > self.BRIGHTNESS_MAX:
            status = "FAIL"
            message = "Overexposed - possible lens damage or direct light"
        
        return {
            'status': status,
            'value': float(mean_brightness),
            'message': message
        }
    
    def analyze_blur(self, frame: np.ndarray) -> Dict:
        """Detect blur using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        status = "PASS" if blur_score > self.BLUR_THRESHOLD else "FAIL"
        message = "Image sharp" if status == "PASS" else "Blurry - possible focus issue or obstruction"
        
        return {
            'status': status,
            'value': float(blur_score),
            'message': message
        }
    
    def detect_edges(self, frame: np.ndarray) -> float:
        """
        Detect edge density to differentiate blur from obstruction
        Returns: Edge density score (0-100)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        return edge_density
    
    def compare_with_baseline(self, frame: np.ndarray, baseline: np.ndarray, 
                              blur_score: float, brightness: float) -> Dict:
        """
        Intelligent comparison using multiple factors:
        - SSIM for structural similarity
        - Pixel difference for changes
        - Edge detection for displacement vs obstruction
        - Blur score to identify focus issues
        - Brightness to identify lighting issues
        """
        # Resize frames to same size for comparison
        h, w = baseline.shape[:2]
        frame_resized = cv2.resize(frame, (w, h))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM (structural similarity)
        similarity, diff_img = ssim(gray_baseline, gray_frame, full=True)
        
        # Calculate pixel-wise difference
        diff = cv2.absdiff(gray_baseline, gray_frame)
        diff_percent = (np.sum(diff > 50) / diff.size) * 100
        
        # Edge detection for both frames
        frame_edges = self.detect_edges(frame_resized)
        baseline_edges = self.detect_edges(baseline)
        edge_diff = abs(frame_edges - baseline_edges)
        
        # INTELLIGENT ALERT CLASSIFICATION
        alerts = []
        status = "PASS"
        
        # 1. Check if it's a BLUR issue first
        if blur_score < self.BLUR_THRESHOLD:
            status = "FAIL"
            alerts.append(f"Camera out of focus or lens obstructed (blur score: {blur_score:.1f})")
            # Don't continue to other checks if blur is the issue
        
        # 2. Check if it's too DARK (always report this separately if true)
        if brightness < self.BRIGHTNESS_MIN and blur_score >= self.BLUR_THRESHOLD:
            status = "FAIL"
            alerts.append("Too dark - possible obstruction or lighting issue")
            # Continue to check what else might be wrong
        
        # 3. HIGH PRIORITY: Check for CAMERA DISPLACEMENT first (before obstruction)
        # Key indicator: Current frame has good edges (not blocked) but scene is very different
        if frame_edges > baseline_edges * 0.4 and diff_percent > self.DIFF_DISPLACEMENT and blur_score >= self.BLUR_THRESHOLD:
            # Camera can see details (has edges) but scene is different = camera moved
            status = "FAIL"
            if diff_percent > 60:
                alerts.append(f"Camera angle changed significantly - completely different view ({diff_percent:.1f}% difference)")
            elif edge_diff > 5:
                alerts.append(f"Camera position changed - viewing angle altered ({diff_percent:.1f}% scene difference)")
            else:
                alerts.append(f"Camera displacement detected ({diff_percent:.1f}% difference)")
        
        # 4. Check for SEVERE OBSTRUCTION (low edges + high difference)
        # If frame has very few edges compared to baseline = something blocking view
        elif (
            frame_edges < baseline_edges * self.EDGE_COLLAPSE_RATIO
            and diff_percent > self.DIFF_SEVERE
        ):
            status = "FAIL"

            if blur_score >= self.BLUR_THRESHOLD:
                alerts.append(
                    "Image clarity degraded - lens smear or camera-side blur detected"
                )
            else:
                alerts.append(
                    "Severe obstruction detected - camera view significantly blocked"
                )

        
        # 5. Check for PARTIAL OBSTRUCTION (moderate edge loss + moderate difference)
        elif frame_edges < baseline_edges * 0.6 and diff_percent > self.DIFF_MODERATE and blur_score >= self.BLUR_THRESHOLD:
            status = "FAIL"
            alerts.append("Partial obstruction detected - camera view partially blocked")
        
        # 6. SSIM-based checks for subtle issues
        elif similarity < self.SSIM_CRITICAL and blur_score >= self.BLUR_THRESHOLD:
            status = "FAIL"
            if frame_edges > baseline_edges * 0.5:
                # Has edges but very different = likely displacement
                alerts.append(f"Camera orientation changed ({similarity:.2f} similarity)")
            else:
                # Lost edges = likely obstruction
                alerts.append(f"Camera view degraded - possible lens damage or obstruction")
        
        # 7. LIGHTING VARIATION (SSIM low but diff acceptable)
        elif similarity < self.SSIM_WARNING and diff_percent <= self.DIFF_DISPLACEMENT and blur_score >= self.BLUR_THRESHOLD:
            status = "PASS"
            logger.info(f"Lighting variation detected: SSIM={similarity:.2f}, Diff={diff_percent:.1f}%")
        
        # 8. If no alerts yet but status is still PASS, everything is good
        if not alerts and status == "PASS":
            status = "PASS"
        
        message = "; ".join(alerts) if alerts else "Vision matches baseline"
        
        return {
            'status': status,
            'similarity': float(similarity),
            'difference_percent': float(diff_percent),
            'edge_density_current': float(frame_edges),
            'edge_density_baseline': float(baseline_edges),
            'message': message,
            'alerts': alerts
        }
    
    def check_camera_vision(self, camera_id: str, rtsp_url: str) -> Dict:
        """
        Complete vision health check with high accuracy
        Returns: Comprehensive vision status report
        """
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = {
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'UNKNOWN',
            'checks': {},
            'alerts': [],
            'performance': {}
        }
        
        try:
            # Step 1: Capture live frame
            logger.info(f"[{camera_id}] Capturing live frame...")
            frame = self.capture_frame(rtsp_url)
            
            if frame is None:
                result['status'] = 'FAIL'
                result['alerts'] = ['Failed to capture frame from camera']
                return result
            else:
                self.save_frame(frame, camera_id, tag="live")
            
            # Step 2: Quality checks
            logger.info(f"[{camera_id}] Analyzing image quality...")
            brightness = self.analyze_brightness(frame)
            blur = self.analyze_blur(frame)
            
            result['checks']['brightness'] = brightness
            result['checks']['blur'] = blur
            
            # Step 3: Baseline comparison (with context from quality checks)
            baseline = self.load_baseline(camera_id)
            if baseline is not None:
                logger.info(f"[{camera_id}] Comparing with baseline...")
                comparison = self.compare_with_baseline(
                    frame, 
                    baseline, 
                    blur['value'],
                    brightness['value']
                )
                result['checks']['baseline_comparison'] = comparison
                
                # Only add comparison alerts (brightness/blur already handled in comparison)
                if comparison['status'] == 'FAIL':
                    result['alerts'].extend(comparison['alerts'])
            else:
                # No baseline - rely on quality checks only
                result['checks']['baseline_comparison'] = {
                    'status': 'SKIPPED',
                    'message': 'No baseline available'
                }
                
                # Add quality alerts when no baseline
                if brightness['status'] == 'FAIL':
                    result['alerts'].append(brightness['message'])
                if blur['status'] == 'FAIL':
                    result['alerts'].append(blur['message'])
            
            # Determine final status
            if baseline is not None:
                # With baseline, use comparison result
                result['status'] = comparison['status']
            else:
                # Without baseline, use quality checks
                all_checks = [brightness['status'], blur['status']]
                if 'FAIL' in all_checks:
                    result['status'] = 'FAIL'
                else:
                    result['status'] = 'PASS'
            
            # Performance metrics
            elapsed = time.time() - start_time
            mem_after = process.memory_info().rss / 1024 / 1024
            
            result['performance'] = {
                'execution_time_sec': round(elapsed, 2),
                'memory_used_mb': round(mem_after - mem_before, 2)
            }
            
            logger.info(f"[{camera_id}] Vision check complete: {result['status']}")
            
        except Exception as e:
            logger.error(f"[{camera_id}] Vision check error: {e}")
            result['status'] = 'ERROR'
            result['alerts'] = [f"Vision check failed: {str(e)}"]
        
        return result


# Example usage
if __name__ == "__main__":
    checker = VisionChecker()
    
    # Test with sample camera
    camera_id = "CAM_001"
    rtsp_url = "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
    
    result = checker.check_camera_vision(camera_id, rtsp_url)
    
    print(f"\n=== Vision Check Results for {camera_id} ===")
    print(f"Status: {result['status']}")
    print(f"Alerts: {result['alerts']}")
    print(f"Execution Time: {result['performance']['execution_time_sec']}s")
    print(f"Memory Used: {result['performance']['memory_used_mb']} MB")
