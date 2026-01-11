"""
Simple and Accurate Vision Checker
Detects only 4 things:
1. Camera blocked/half-blocked
2. Camera blurred
3. Camera position changed
4. Objects detected (humans/vehicles/animals) - REPORT ONLY, NO ALERT
"""

import time
import logging
import tracemalloc
from pathlib import Path
from typing import Dict
from datetime import datetime
from dataclasses import dataclass

import cv2
import torch
import clip
import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.metrics import structural_similarity as ssim

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
IMAGES_DIR = Path("img")
BASELINE_DIR = IMAGES_DIR / "baseline"
CAPTURE_DIR = IMAGES_DIR / "captures"

for d in [IMAGES_DIR, BASELINE_DIR, CAPTURE_DIR]:
    d.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Simple Detection Prompts
# -------------------------------------------------------------------
OBSTRUCTION_PROMPTS = [
    "camera lens completely blocked by object or hand",
    "camera partially covered blocking part of view",
    "camera lens covered by tape or material",
]

BLUR_PROMPTS = [
    "camera image blurred out of focus",
    "camera feed with motion blur",
]

POSITION_PROMPTS = [
    "camera rotated or displaced from original position",
    "camera angle changed significantly",
]

OBJECT_DETECTION_PROMPTS = [
    "person or human visible in camera view",
    "vehicle or car in camera view",
    "animal or pet in camera view",
]

NORMAL_PROMPTS = [
    "clear security camera view",
    "normal camera feed",
]

ALL_PROMPTS = (
    OBSTRUCTION_PROMPTS + BLUR_PROMPTS + POSITION_PROMPTS + 
    OBJECT_DETECTION_PROMPTS + NORMAL_PROMPTS
)

# -------------------------------------------------------------------
# Analysis Results Data Classes
# -------------------------------------------------------------------
@dataclass
class ObstructionAnalysis:
    has_full_obstruction: bool
    has_partial_obstruction: bool
    obstruction_percentage: float
    edge_blockage_score: float
    uniformity_score: float
    confidence: float
    details: str

@dataclass
class BlurAnalysis:
    is_severely_blurred: bool
    is_moderately_blurred: bool
    laplacian_variance: float
    fft_blur_score: float
    gradient_strength: float
    edge_density: float
    confidence: float
    details: str

@dataclass
class PositionAnalysis:
    is_severely_displaced: bool
    is_slightly_displaced: bool
    displacement_ratio: float
    rotation_angle: float
    keypoint_match_ratio: float
    homography_quality: float
    confidence: float
    details: str

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("vision_checker")

# -------------------------------------------------------------------
# CLIP Model Singleton
# -------------------------------------------------------------------
class CLIPModel:
    _model = None
    _preprocess = None

    @classmethod
    def load(cls):
        if cls._model is None:
            logger.info(f"Loading CLIP ViT-B/32 on {DEVICE}")
            cls._model, cls._preprocess = clip.load("ViT-B/32", device=DEVICE)
            cls._model.eval()
        return cls._model, cls._preprocess

# -------------------------------------------------------------------
# Vision Checker
# -------------------------------------------------------------------
class VisionChecker:
    """
    Simple vision checker - detects only what matters
    """
    
    def __init__(self):
        self.model, self.preprocess = CLIPModel.load()
        self._tokenize_prompts()
        logger.info("Vision Checker initialized")
        
    def _tokenize_prompts(self):
        logger.info(f"Tokenizing {len(ALL_PROMPTS)} prompts")
        self.tokenized_prompts = clip.tokenize(ALL_PROMPTS).to(DEVICE)

    # ==================== FRAME CAPTURE ====================
    def capture_frame(self, camera_id: str, rtsp_url: str) -> Path:
        """Capture frame from RTSP stream"""
        logger.info(f"[CAPTURE] {camera_id}")
        
        cap = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open RTSP: {rtsp_url}")
                
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                success, frame = cap.read()
                
                if not success or frame is None or frame.size == 0:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries}")
                        time.sleep(0.5)
                        continue
                    raise RuntimeError(f"Failed to read frame from {camera_id}")
                
                timestamp = int(time.time())
                output_path = CAPTURE_DIR / f"{camera_id}_{timestamp}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                logger.info(f"[CAPTURE] Saved: {output_path.name}")
                return output_path
                
            finally:
                if cap is not None:
                    cap.release()

    # ==================== SIMILARITY CHECK (MOST IMPORTANT) ====================
    def compute_clip_similarity(self, img1_path: Path, img2_path: Path) -> float:
        """Compute CLIP semantic similarity between baseline and current"""
        img1 = self.preprocess(Image.open(img1_path)).unsqueeze(0).to(DEVICE)
        img2 = self.preprocess(Image.open(img2_path)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            emb1 = self.model.encode_image(img1)
            emb2 = self.model.encode_image(img2)
            emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
            similarity = (emb1 @ emb2.T).item()
        
        return round(similarity, 4)

    # ==================== BLUR DETECTION ====================
    def detect_blur(self, img_path: Path, similarity: float) -> BlurAnalysis:
        """
        Simple blur detection using Laplacian and FFT
        Uses similarity to avoid false positives
        """
        # If similarity is very high, skip blur detection
        if similarity >= 0.85:
            return BlurAnalysis(
                is_severely_blurred=False,
                is_moderately_blurred=False,
                laplacian_variance=1000.0,
                fft_blur_score=1.0,
                gradient_strength=100.0,
                edge_density=1.0,
                confidence=0.0,
                details="Sharp - high similarity to baseline"
            )
        
        img = cv2.imread(str(img_path))
        if img is None:
            return BlurAnalysis(False, False, 0, 0, 0, 0, 0, "Image load error")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # FFT high frequency analysis
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = ((y - center_y)**2 + (x - center_x)**2) > radius**2
        high_freq_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)
        fft_blur_score = high_freq_energy / (total_energy + 1e-6)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient strength
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_strength = np.mean(gradient_magnitude)
        
        # VERY STRICT THRESHOLDS - Only flag OBVIOUS blur
        is_severely_blurred = (
            laplacian_var < 20 and 
            edge_density < 0.01 and
            fft_blur_score < 0.10
        )
        
        is_moderately_blurred = (
            laplacian_var < 50 and 
            edge_density < 0.025 and
            fft_blur_score < 0.15 and
            not is_severely_blurred
        )
        
        confidence = 0.0
        details = "Sharp image"
        
        if is_severely_blurred:
            confidence = 0.95
            details = f"Severe blur: laplacian={laplacian_var:.0f}, edges={edge_density:.3f}"
        elif is_moderately_blurred:
            confidence = 0.80
            details = f"Moderate blur: laplacian={laplacian_var:.0f}, edges={edge_density:.3f}"
        
        logger.info(
            f"[BLUR] Severe={is_severely_blurred}, Moderate={is_moderately_blurred}, "
            f"Laplacian={laplacian_var:.0f}, EdgeDensity={edge_density:.3f}"
        )
        
        return BlurAnalysis(
            is_severely_blurred=is_severely_blurred,
            is_moderately_blurred=is_moderately_blurred,
            laplacian_variance=laplacian_var,
            fft_blur_score=fft_blur_score,
            gradient_strength=gradient_strength,
            edge_density=edge_density,
            confidence=confidence,
            details=details
        )

    # ==================== OBSTRUCTION DETECTION ====================
    def detect_obstruction(
        self,
        img_path: Path,
        baseline_path: Path,
        similarity: float
    ) -> ObstructionAnalysis:
        """
        Detect if camera is blocked (fully or partially)
        Uses similarity as primary indicator
        """
        # If similarity is high, camera is NOT blocked
        if similarity >= 0.85:
            return ObstructionAnalysis(
                has_full_obstruction=False,
                has_partial_obstruction=False,
                obstruction_percentage=0.0,
                edge_blockage_score=0.0,
                uniformity_score=0.0,
                confidence=0.0,
                details=f"Camera clear - similarity {similarity:.3f}"
            )

        img = cv2.imread(str(img_path))
        baseline = cv2.imread(str(baseline_path))

        if img is None or baseline is None:
            return ObstructionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0,
                "Image load error"
            )

        if img.shape != baseline.shape:
            img = cv2.resize(img, (baseline.shape[1], baseline.shape[0]))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

        # Check if image is uniform (covered by solid object)
        std_dev = np.std(gray)
        variance = np.var(gray)
        uniformity_score = 1.0 - min(std_dev / 128.0, 1.0)

        # Edge comparison
        edges_current = cv2.Canny(gray, 50, 150)
        edges_baseline = cv2.Canny(baseline_gray, 50, 150)

        edge_density_current = np.sum(edges_current > 0) / edges_current.size
        edge_density_baseline = np.sum(edges_baseline > 0) / edges_baseline.size

        edge_reduction = max(
            0.0,
            1.0 - (edge_density_current / (edge_density_baseline + 1e-6))
        )

        # SSIM
        ssim_score = ssim(baseline_gray, gray)

        # Difference percentage
        diff = cv2.absdiff(baseline_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        obstruction_pct = (np.sum(thresh > 0) / thresh.size) * 100.0

        # Border blockage
        h, w = gray.shape
        border_size = min(h, w) // 10
        borders = [
            np.std(gray[:border_size, :]),
            np.std(gray[-border_size:, :]),
            np.std(gray[:, :border_size]),
            np.std(gray[:, -border_size:])
        ]
        border_uniformity = np.mean(borders)
        edge_blockage_score = 1.0 - min(border_uniformity / 50.0, 1.0)

        # VERY STRICT DECISION LOGIC - Only flag OBVIOUS obstruction
        # Full obstruction: Image is VERY uniform + VERY low edges + LOW similarity
        has_full_obstruction = (
            (uniformity_score > 0.92 and std_dev < 15) and
            (edge_density_current < 0.015) and
            (ssim_score < 0.40) and
            (similarity < 0.60)
        )

        # Partial obstruction: High edge loss + LOW similarity + significant difference
        has_partial_obstruction = (
            (edge_reduction > 0.70) and
            (ssim_score < 0.50) and
            (obstruction_pct > 40) and
            (similarity < 0.75) and
            not has_full_obstruction
        )

        confidence = 0.0
        details = "No obstruction"

        if has_full_obstruction:
            confidence = 0.95
            details = f"Full obstruction: uniformity={uniformity_score:.2f}, std={std_dev:.0f}, sim={similarity:.3f}"
        elif has_partial_obstruction:
            confidence = 0.85
            details = f"Partial obstruction: edge_loss={edge_reduction:.2f}, ssim={ssim_score:.2f}, sim={similarity:.3f}"

        logger.info(
            f"[OBSTRUCTION] Full={has_full_obstruction}, Partial={has_partial_obstruction}, "
            f"Uniformity={uniformity_score:.2f}, SSIM={ssim_score:.2f}, Similarity={similarity:.3f}"
        )

        return ObstructionAnalysis(
            has_full_obstruction=has_full_obstruction,
            has_partial_obstruction=has_partial_obstruction,
            obstruction_percentage=obstruction_pct,
            edge_blockage_score=edge_blockage_score,
            uniformity_score=uniformity_score,
            confidence=confidence,
            details=details
        )

    # ==================== POSITION DETECTION ====================
    def detect_position_change(
        self,
        baseline_path: Path,
        current_path: Path,
        similarity: float
    ) -> PositionAnalysis:
        """
        Detect camera position change
        CRITICAL: Only flags position change when similarity is VERY low
        """
        # CRITICAL FIX: If similarity >= 0.75, position is FINE
        # This covers normal operation with lighting/scene changes
        if similarity >= 0.75:
            return PositionAnalysis(
                is_severely_displaced=False,
                is_slightly_displaced=False,
                displacement_ratio=0.0,
                rotation_angle=0.0,
                keypoint_match_ratio=1.0,
                homography_quality=1.0,
                confidence=0.0,
                details=f"Position stable - similarity {similarity:.3f} (>= 0.75 threshold)"
            )

        # Even if similarity is low, check if it's due to obstruction/blur
        # Don't run expensive SIFT if similarity is in the 0.60-0.75 range
        # This range usually means scene content changed, NOT camera moved
        if 0.60 <= similarity < 0.75:
            return PositionAnalysis(
                is_severely_displaced=False,
                is_slightly_displaced=False,
                displacement_ratio=0.0,
                rotation_angle=0.0,
                keypoint_match_ratio=0.5,
                homography_quality=0.5,
                confidence=0.0,
                details=f"Position likely stable - similarity {similarity:.3f} in normal range (0.60-0.75)"
            )

        # Only check position if similarity is VERY low (< 0.60)
        # This means something major changed
        logger.info(f"[POSITION] Low similarity {similarity:.3f}, running keypoint analysis")

        img_base = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
        img_curr = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)

        if img_base is None or img_curr is None:
            return PositionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0, 0.0,
                "Image load error"
            )

        # Check if current image is uniform (obstruction, not position change)
        std_dev_curr = np.std(img_curr)
        if std_dev_curr < 25:
            return PositionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0, 0.0,
                f"Uniform scene (std={std_dev_curr:.0f}) - likely obstruction, not position change"
            )

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_base, None)
        kp2, des2 = sift.detectAndCompute(img_curr, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # Few keypoints could mean obstruction
            return PositionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0, 0.0,
                f"Insufficient keypoints (base={len(kp1) if des1 else 0}, curr={len(kp2) if des2 else 0}) - likely obstruction"
            )

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        match_ratio = len(good) / max(len(kp1), 1)

        # EXTREMELY STRICT: Only flag if match ratio is ULTRA low AND similarity is low
        is_severe = (match_ratio < 0.02 and similarity < 0.50)  # Only 2% matches + very low similarity
        is_slight = (0.02 <= match_ratio < 0.08 and similarity < 0.55)  # 2-8% matches + low similarity

        homography_quality = 0.0
        rotation_angle = 0.0

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                homography_quality = float(np.mean(mask))
                rotation_angle = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
                
                # Additional check: if rotation is small, not a position change
                if abs(rotation_angle) < 10 and match_ratio > 0.05:
                    is_severe = False
                    is_slight = False

        confidence = 0.0
        details = f"Position stable - {len(good)} matches ({match_ratio:.3f} ratio), sim={similarity:.3f}"

        if is_severe:
            confidence = 0.95
            details = f"Severe displacement - only {len(good)} matches ({match_ratio:.3f} ratio), sim={similarity:.3f}"
        elif is_slight:
            confidence = 0.80
            details = f"Slight displacement - {len(good)} matches ({match_ratio:.3f} ratio), sim={similarity:.3f}"

        logger.info(
            f"[POSITION] Severe={is_severe}, Slight={is_slight}, "
            f"Matches={len(good)}/{len(kp1)}, Ratio={match_ratio:.3f}, Sim={similarity:.3f}"
        )

        return PositionAnalysis(
            is_severely_displaced=is_severe,
            is_slightly_displaced=is_slight,
            displacement_ratio=1.0 - match_ratio,
            rotation_angle=rotation_angle,
            keypoint_match_ratio=match_ratio,
            homography_quality=homography_quality,
            confidence=confidence,
            details=details
        )

    # ==================== OBJECT DETECTION (REPORT ONLY) ====================
    def detect_objects(self, image_path: Path) -> Dict:
        """Detect humans/vehicles/animals - for REPORT only, NOT alert"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        
        object_prompts = OBJECT_DETECTION_PROMPTS
        tokenized = clip.tokenize(object_prompts).to(DEVICE)
        
        with torch.no_grad():
            img_features = self.model.encode_image(image)
            txt_features = self.model.encode_text(tokenized)
            
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            
            logits = (img_features @ txt_features.T) / 0.07
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
        
        detections = []
        for prompt, prob in zip(object_prompts, probs):
            if prob > 0.35:  # Slightly higher threshold
                detections.append({
                    "object": prompt,
                    "confidence": float(prob)
                })
        
        return {
            "objects_detected": detections,
            "has_objects": len(detections) > 0
        }

    # ==================== MAIN DECISION LOGIC ====================
    def run(self, camera_id: str, rtsp_url: str) -> Dict:
        """
        Main entry point - simple 4-step check
        """
        start_time = time.time()
        tracemalloc.start()

        baseline_path = BASELINE_DIR / f"{camera_id}.png"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline image missing for {camera_id}")

        # Step 1: Capture frame
        captured_path = self.capture_frame(camera_id, rtsp_url)

        # Step 2: Calculate similarity (MOST IMPORTANT METRIC)
        similarity = self.compute_clip_similarity(baseline_path, captured_path)
        logger.info(f"[SIMILARITY] {camera_id} = {similarity:.4f}")

        # CRITICAL DECISION POINT: If similarity >= 0.85, camera is FINE
        # This covers 99% of normal operations
        if similarity >= 0.85:
            logger.info(f"[DECISION] {camera_id} PASS - Similarity {similarity:.4f} >= 0.85 threshold")
            
            # Still check for objects (for report, not alert)
            object_detection = self.detect_objects(captured_path)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_consumed_mb = round(peak / (1024 * 1024), 3)
            
            return {
                "camera_id": camera_id,
                "status": "PASS",
                "severity": "NORMAL",
                "summary": "Camera functioning normally",
                "confidence_score": 0.0,
                "similarity": similarity,
                "obstruction": {"has_full_obstruction": False, "has_partial_obstruction": False, "confidence": 0.0, "details": "Camera clear"},
                "blur": {"is_severely_blurred": False, "is_moderately_blurred": False, "confidence": 0.0, "details": "Sharp image"},
                "position": {"is_severely_displaced": False, "is_slightly_displaced": False, "confidence": 0.0, "details": "Position stable"},
                "object_detection": object_detection,
                "memory_metrics": {
                    "memory_consumed_mb": memory_consumed_mb,
                    "ram_current_mb": round(current / (1024 * 1024), 3),
                    "ram_peak_mb": memory_consumed_mb,
                    "cpu_hint": "vision-light"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "check_duration_ms": round((time.time() - start_time) * 1000, 2),
            }

        # Step 3: Run detailed checks (only if similarity < 0.85)
        logger.info(f"[DECISION] {camera_id} - Low similarity {similarity:.4f}, running detailed checks")
        
        blur = self.detect_blur(captured_path, similarity)
        obstruction = self.detect_obstruction(captured_path, baseline_path, similarity)
        position = self.detect_position_change(baseline_path, captured_path, similarity)
        object_detection = self.detect_objects(captured_path)

        # Step 4: Make decision (STRICT PRIORITY ORDER)
        status = "PASS"
        severity = "NORMAL"
        summary = "Camera functioning normally"
        overall_confidence = 0.0

        # Priority 1: Blur (check confidence threshold)
        if blur.is_severely_blurred and blur.confidence > 0.7:
            status = "FAIL"
            severity = "MAJOR"
            summary = "Severe blur detected"
            overall_confidence = blur.confidence
        elif blur.is_moderately_blurred and blur.confidence > 0.6:
            status = "FAIL"
            severity = "MINOR"
            summary = "Moderate blur detected"
            overall_confidence = blur.confidence

        # Priority 2: Obstruction (check confidence threshold)
        elif obstruction.has_full_obstruction and obstruction.confidence > 0.8:
            status = "FAIL"
            severity = "CRITICAL"
            summary = "Camera fully obstructed"
            overall_confidence = obstruction.confidence
        elif obstruction.has_partial_obstruction and obstruction.confidence > 0.7:
            status = "FAIL"
            severity = "MAJOR"
            summary = "Camera partially obstructed"
            overall_confidence = obstruction.confidence

        # Priority 3: Position (check confidence threshold - HIGHEST bar)
        elif position.is_severely_displaced and position.confidence > 0.85:
            status = "FAIL"
            severity = "MAJOR"
            summary = "Camera position changed significantly"
            overall_confidence = position.confidence
        elif position.is_slightly_displaced and position.confidence > 0.75:
            status = "FAIL"
            severity = "MINOR"
            summary = "Camera position slightly changed"
            overall_confidence = position.confidence

        # Objects detected = REPORT only, not FAIL
        if object_detection["has_objects"] and status == "PASS":
            summary = f"Objects detected: {', '.join([d['object'].split()[0] for d in object_detection['objects_detected'][:2]])}"
            severity = "INFO"

        logger.info(f"[DECISION] {camera_id} {status} - {summary} (confidence={overall_confidence:.2f})")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_consumed_mb = round(peak / (1024 * 1024), 3)

        return {
            "camera_id": camera_id,
            "status": status,
            "severity": severity,
            "summary": summary,
            "confidence_score": overall_confidence,
            "similarity": similarity,
            "obstruction": obstruction.__dict__,
            "blur": blur.__dict__,
            "position": position.__dict__,
            "object_detection": object_detection,
            "semantic_analysis": {"summary": summary},
            "memory_metrics": {
                "memory_consumed_mb": memory_consumed_mb,
                "ram_current_mb": round(current / (1024 * 1024), 3),
                "ram_peak_mb": memory_consumed_mb,
                "cpu_hint": "vision-heavy"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
        }
