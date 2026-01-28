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

        # SSIM thresholds - MORE LENIENT for day-to-day variations
        self.SSIM_CRITICAL = 0.40   # Below this = severe issue
        self.SSIM_WARNING = 0.55    # Below this = potential issue
        self.SSIM_GOOD = 0.65       # Above this = healthy
        
        # Difference thresholds - MORE LENIENT for lighting/compression variations
        self.DIFF_SEVERE = 55       # >55% difference = severe obstruction
        self.DIFF_MODERATE = 40     # >40% difference = partial obstruction
        self.DIFF_DISPLACEMENT = 35 # >35% difference = possible displacement
        
        # Brightness tolerance for day/night variations
        self.BRIGHTNESS_TOLERANCE = 40  # Allow 40 units of brightness change
        
    def capture_frame(self, rtsp_url: str, timeout: int = 10) -> Optional[np.ndarray]:
        """
        Capture a single frame from RTSP stream
        Returns: Frame as numpy array or None if failed
        """
        cap = None
        try:
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {rtsp_url}")
                return None
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Validate frame dimensions
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        return frame
                time.sleep(0.1)
            
            logger.error(f"Failed to capture valid frame within {timeout}s")
            return None
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}", exc_info=True)
            return None
        finally:
            if cap:
                cap.release()
                
    def save_frame(self, frame: np.ndarray, camera_id: str, tag: str = "capture"):
        """Save captured frame for debugging/logging"""
        try:
            cam_dir = os.path.join(self.capture_dir, camera_id)
            os.makedirs(cam_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{tag}_{timestamp}.jpg"
            path = os.path.join(cam_dir, filename)

            cv2.imwrite(path, frame)
            logger.debug(f"[{camera_id}] Frame saved: {path}")
        except Exception as e:
            logger.error(f"Failed to save frame for {camera_id}: {e}")

    def load_baseline(self, camera_id: str) -> Optional[np.ndarray]:
        """Load baseline image for comparison"""
        baseline_path = os.path.join(self.baseline_dir, f"{camera_id}.png")
        
        if not os.path.exists(baseline_path):
            logger.warning(f"No baseline found for {camera_id} at {baseline_path}")
            return None
        
        try:
            baseline = cv2.imread(baseline_path)
            if baseline is None or baseline.size == 0:
                logger.error(f"Failed to read baseline image for {camera_id}")
                return None
            return baseline
        except Exception as e:
            logger.error(f"Baseline load error: {e}")
            return None
    
    def analyze_brightness(self, frame: np.ndarray) -> Dict:
        """Analyze frame brightness"""
        try:
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
        except Exception as e:
            logger.error(f"Brightness analysis error: {e}")
            return {
                'status': 'ERROR',
                'value': 0.0,
                'message': f"Brightness analysis failed: {str(e)}"
            }
    
    def analyze_blur(self, frame: np.ndarray) -> Dict:
        """Detect blur using Laplacian variance"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            status = "PASS" if blur_score > self.BLUR_THRESHOLD else "FAIL"
            message = "Image sharp" if status == "PASS" else "Blurry - possible focus issue or obstruction"
            
            return {
                'status': status,
                'value': float(blur_score),
                'message': message
            }
        except Exception as e:
            logger.error(f"Blur analysis error: {e}")
            return {
                'status': 'ERROR',
                'value': 0.0,
                'message': f"Blur analysis failed: {str(e)}"
            }
    
    def detect_edges(self, frame: np.ndarray) -> float:
        """
        Detect edge density to differentiate blur from obstruction
        Returns: Edge density score (0-100)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = (np.sum(edges > 0) / edges.size) * 100
            return edge_density
        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return 0.0
    
    def compare_with_baseline(self, frame: np.ndarray, baseline: np.ndarray, 
                              blur_score: float, brightness: float) -> Dict:
        """
        Intelligent comparison using multiple factors with improved tolerance for natural variations
        """
        try:
            # Resize frames to same size for comparison
            h, w = baseline.shape[:2]
            frame_resized = cv2.resize(frame, (w, h))
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
            
            # Calculate baseline brightness for comparison
            baseline_brightness = np.mean(gray_baseline)
            brightness_diff = abs(brightness - baseline_brightness)
            
            # Calculate SSIM (structural similarity)
            similarity, diff_img = ssim(gray_baseline, gray_frame, full=True)
            
            # Calculate pixel-wise difference
            diff = cv2.absdiff(gray_baseline, gray_frame)
            diff_percent = (np.sum(diff > 50) / diff.size) * 100
            
            # Edge detection for both frames
            frame_edges = self.detect_edges(frame_resized)
            baseline_edges = self.detect_edges(baseline)
            edge_diff = abs(frame_edges - baseline_edges)
            edge_ratio = frame_edges / baseline_edges if baseline_edges > 0 else 1.0
            
            # INTELLIGENT ALERT CLASSIFICATION
            alerts = []
            status = "PASS"
            
            # 1. Check if it's a BLUR issue first
            if blur_score < self.BLUR_THRESHOLD:
                status = "FAIL"
                alerts.append(f"Camera out of focus or lens obstructed (blur score: {blur_score:.1f})")
            
            # 2. Check if it's too DARK (but account for day/night cycle)
            elif brightness < self.BRIGHTNESS_MIN and brightness_diff > self.BRIGHTNESS_TOLERANCE:
                status = "FAIL"
                alerts.append("Too dark - possible obstruction or lighting issue")
            
            # 3. CAMERA DISPLACEMENT - STRICTER CRITERIA to reduce false positives
            elif (
                edge_ratio > 0.7  # Frame has good edges
                and diff_percent > 45  # Very high difference
                and similarity < 0.50  # Low structural similarity
                and blur_score >= self.BLUR_THRESHOLD
                and brightness_diff < self.BRIGHTNESS_TOLERANCE  # Not just lighting change
            ):
                status = "FAIL"
                if diff_percent > 70:
                    alerts.append(f"Camera angle changed significantly - completely different view ({diff_percent:.1f}% difference)")
                elif diff_percent > 55:
                    alerts.append(f"Camera position changed - viewing angle altered ({diff_percent:.1f}% scene difference)")
                else:
                    alerts.append(f"Camera displacement detected ({diff_percent:.1f}% difference)")
            
            # 4. SEVERE OBSTRUCTION (low edges + high difference)
            elif (
                edge_ratio < self.EDGE_COLLAPSE_RATIO
                and diff_percent > self.DIFF_SEVERE
                and blur_score >= self.BLUR_THRESHOLD
            ):
                status = "FAIL"
                if blur_score >= self.BLUR_THRESHOLD:
                    alerts.append("Image clarity degraded - lens smear or camera-side blur detected")
                else:
                    alerts.append("Severe obstruction detected - camera view significantly blocked")
            
            # 5. PARTIAL OBSTRUCTION
            elif (
                edge_ratio < 0.6 
                and diff_percent > self.DIFF_MODERATE 
                and blur_score >= self.BLUR_THRESHOLD
                and brightness_diff < self.BRIGHTNESS_TOLERANCE
            ):
                status = "FAIL"
                alerts.append("Partial obstruction detected - camera view partially blocked")
            
            # 6. CRITICAL SSIM - only if not explained by lighting
            elif (
                similarity < self.SSIM_CRITICAL 
                and blur_score >= self.BLUR_THRESHOLD
                and brightness_diff < self.BRIGHTNESS_TOLERANCE
            ):
                status = "FAIL"
                if edge_ratio > 0.5:
                    alerts.append(f"Camera orientation changed ({similarity:.2f} similarity)")
                else:
                    alerts.append(f"Camera view degraded - possible lens damage or obstruction")
            
            # 7. LIGHTING VARIATION - now more tolerant
            elif (
                similarity < self.SSIM_WARNING 
                and diff_percent <= self.DIFF_DISPLACEMENT
                and brightness_diff >= self.BRIGHTNESS_TOLERANCE
            ):
                status = "PASS"
                logger.info(f"Natural lighting variation detected: SSIM={similarity:.2f}, Brightness diff={brightness_diff:.1f}")
            
            # 8. Normal day-to-day variation
            elif similarity >= self.SSIM_WARNING:
                status = "PASS"
            
            message = "; ".join(alerts) if alerts else "Vision matches baseline"
            
            return {
                'status': status,
                'similarity': float(similarity),
                'difference_percent': float(diff_percent),
                'edge_density_current': float(frame_edges),
                'edge_density_baseline': float(baseline_edges),
                'edge_ratio': float(edge_ratio),
                'brightness_diff': float(brightness_diff),
                'message': message,
                'alerts': alerts
            }
        except Exception as e:
            logger.error(f"Baseline comparison error: {e}", exc_info=True)
            return {
                'status': 'ERROR',
                'similarity': 0.0,
                'difference_percent': 0.0,
                'edge_density_current': 0.0,
                'edge_density_baseline': 0.0,
                'edge_ratio': 0.0,
                'brightness_diff': 0.0,
                'message': f"Comparison failed: {str(e)}",
                'alerts': [f"Comparison error: {str(e)}"]
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
            
            # Validate frame
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                result['status'] = 'FAIL'
                result['alerts'] = ['Captured frame is invalid or empty']
                return result
            
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
                
                # Only add comparison alerts
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
                if 'FAIL' in all_checks or 'ERROR' in all_checks:
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
            logger.error(f"[{camera_id}] Vision check error: {e}", exc_info=True)
            result['status'] = 'ERROR'
            result['alerts'] = [f"Vision check failed: {str(e)}"]
        
        return result


# Example usage
if __name__ == "__main__":
    checker = VisionChecker()
    
    # Test with sample camera
    camera_id = "cam13"
    rtsp_url = "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
    
    result = checker.check_camera_vision(camera_id, rtsp_url)
    
    print(f"\n=== Vision Check Results for {camera_id} ===")
    print(f"Status: {result['status']}")
    print(f"Alerts: {result['alerts']}")
    print(f"Execution Time: {result['performance']['execution_time_sec']}s")
    print(f"Memory Used: {result['performance']['memory_used_mb']} MB")