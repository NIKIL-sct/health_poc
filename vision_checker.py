"""
Enhanced Production-Grade Vision Checker
Multi-modal analysis with superior accuracy for security camera monitoring.

Key Improvements:
1. Multi-scale obstruction detection (full, partial, edge-based)
2. Advanced blur detection (Laplacian, FFT, gradient-based)
3. Robust position tracking (SIFT, homography, keypoint density)
4. Hierarchical decision making with confidence scoring
5. Ensemble analysis for higher precision
"""

import time
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
from dataclasses import dataclass

import cv2
import torch
import clip
import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.feature import local_binary_pattern
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
# Enhanced Detection Prompts - More Specific
# -------------------------------------------------------------------
FULL_OBSTRUCTION_PROMPTS = [
    "camera lens completely covered by hand blocking entire view",
    "security camera fully blocked by black tape over lens",
    "camera view entirely obscured by solid object showing nothing",
    "surveillance camera lens completely covered by opaque material",
    "camera blocked showing uniform dark or solid color only",
]

PARTIAL_OBSTRUCTION_PROMPTS = [
    "camera partially blocked with object covering part of view",
    "camera lens half covered by finger or hand at edge",
    "security camera with corner or edge blocked by obstacle",
    "camera view partially obscured by translucent material",
    "camera lens with partial coverage reducing visibility area",
    "camera view with object blocking portion of scene",
    "surveillance camera with vegetation partially blocking lens",
    "camera partially covered by spider web or debris",
]

ENVIRONMENTAL_OBSTRUCTION_PROMPTS = [
    "camera lens covered by water droplets or rain affecting clarity",
    "security camera view obscured by fog or heavy mist",
    "camera lens dirty with mud or dust covering surface",
    "camera view blocked by condensation or moisture on lens",
    "surveillance camera with ice or frost on lens surface",
]

SEVERE_POSITIONING_PROMPTS = [
    "camera rotated ninety degrees showing wrong orientation",
    "security camera pointing upward at ceiling instead of scene",
    "camera displaced facing wall or wrong direction entirely",
    "surveillance camera angle completely changed from original position",
    "camera tilted severely showing mostly floor or sky",
]

MINOR_POSITIONING_PROMPTS = [
    "camera slightly rotated from original mounting position",
    "security camera with minor angle shift from baseline",
    "camera position subtly changed showing different framing",
]

SEVERE_BLUR_PROMPTS = [
    "camera feed completely blurred unreadable out of focus",
    "security camera showing severe motion blur obscuring details",
    "camera image heavily defocused with no sharp features",
    "surveillance feed entirely blurred showing no clear objects",
]

MODERATE_BLUR_PROMPTS = [
    "camera image moderately blurred reducing detail clarity",
    "security camera feed with noticeable blur affecting recognition",
    "camera view with significant blur making details unclear",
]

LIGHTING_EXTREME_PROMPTS = [
    "camera showing completely black image no visible details",
    "security camera view entirely white washed out overexposed",
    "camera in total darkness showing pitch black screen",
    "camera with severe overexposure showing no scene details",
]

LIGHTING_MODERATE_PROMPTS = [
    "camera in very low light with heavily reduced visibility",
    "security camera with strong lens flare reducing clarity",
    "camera affected by glare or reflection obscuring parts",
    "surveillance camera in dim lighting with poor visibility",
]

NORMAL_PROMPTS = [
    "security camera with perfectly clear sharp outdoor view",
    "surveillance camera showing normal clear indoor scene",
    "camera feed sharp well-lit properly positioned monitoring area",
    "security camera functioning normally with excellent visibility",
]

ALL_PROMPTS = (
    FULL_OBSTRUCTION_PROMPTS + PARTIAL_OBSTRUCTION_PROMPTS + 
    ENVIRONMENTAL_OBSTRUCTION_PROMPTS + SEVERE_POSITIONING_PROMPTS +
    MINOR_POSITIONING_PROMPTS + SEVERE_BLUR_PROMPTS + 
    MODERATE_BLUR_PROMPTS + LIGHTING_EXTREME_PROMPTS + 
    LIGHTING_MODERATE_PROMPTS + NORMAL_PROMPTS
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
logger = logging.getLogger("vision_enhanced")

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
# Enhanced Vision Checker
# -------------------------------------------------------------------
class VisionChecker:
    """
    Production-grade vision checker with multi-modal analysis.
    Combines computer vision, deep learning, and heuristics for maximum accuracy.
    """
    
    def __init__(self):
        self.model, self.preprocess = CLIPModel.load()
        self._tokenize_prompts()
        logger.info("Enhanced Vision Checker initialized")
        
    def _tokenize_prompts(self):
        logger.info(f"Tokenizing {len(ALL_PROMPTS)} specialized prompts")
        self.tokenized_prompts = clip.tokenize(ALL_PROMPTS).to(DEVICE)

    # ==================== FRAME CAPTURE ====================
    def capture_frame(self, camera_id: str, rtsp_url: str) -> Path:
        """Capture frame from RTSP stream with retry logic"""
        logger.info(f"[CAPTURE] {camera_id}")
        
        cap = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open RTSP: {rtsp_url}")
                
                # Configure for real-time capture
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Read frame
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
                
                logger.info(f"[CAPTURE] Saved: {output_path.name} ({frame.shape})")
                return output_path
                
            finally:
                if cap is not None:
                    cap.release()

    # ==================== ADVANCED OBSTRUCTION DETECTION ====================
    def detect_obstruction(
        self,
        img_path: Path,
        baseline_path: Path,
        blur: BlurAnalysis
    ) -> ObstructionAnalysis:
        """
        Multi-method obstruction detection:
        1. Color uniformity analysis (full obstruction)
        2. Edge detection comparison (partial obstruction)
        3. Structural similarity (SSIM)
        4. Histogram analysis
        5. Variance-based detection
        """

        # --------------------------------------------------
        # 1️⃣ Load images
        # --------------------------------------------------
        img = cv2.imread(str(img_path))
        baseline = cv2.imread(str(baseline_path))

        if img is None or baseline is None:
            return ObstructionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0,
                "Image load error"
            )

        # --------------------------------------------------
        # 2️⃣ Resize to match baseline
        # --------------------------------------------------
        if img.shape != baseline.shape:
            img = cv2.resize(img, (baseline.shape[1], baseline.shape[0]))

        # --------------------------------------------------
        # 3️⃣ Convert to grayscale (MUST be early)
        # --------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------
        # 4️⃣ Blur guard (lens blur masks obstruction)
        # --------------------------------------------------
        if blur.is_severely_blurred or blur.is_moderately_blurred:
            return ObstructionAnalysis(
                has_full_obstruction=False,
                has_partial_obstruction=False,
                obstruction_percentage=0.0,
                edge_blockage_score=0.0,
                uniformity_score=0.0,
                confidence=0.0,
                details="Blur detected — obstruction skipped"
            )

        # --------------------------------------------------
        # 5️⃣ Scene complexity / position-change guard
        # --------------------------------------------------
        hist = np.histogram(gray, bins=256)[0].astype(np.float32)
        hist /= (hist.sum() + 1e-9)

        entropy = -np.sum(hist * np.log2(hist + 1e-9))

        if entropy > 4.5 and np.std(gray) > 30:
            return ObstructionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0,
                "Scene complexity high — obstruction suppressed (likely position change)"
            )

        # --------------------------------------------------
        # 6️⃣ Uniformity / variance analysis
        # --------------------------------------------------
        std_dev = np.std(gray)
        variance = np.var(gray)
        uniformity_score = 1.0 - min(std_dev / 128.0, 1.0)

        # --------------------------------------------------
        # 7️⃣ Edge density comparison
        # --------------------------------------------------
        edges_current = cv2.Canny(gray, 50, 150)
        edges_baseline = cv2.Canny(baseline_gray, 50, 150)

        edge_density_current = np.sum(edges_current > 0) / edges_current.size
        edge_density_baseline = np.sum(edges_baseline > 0) / edges_baseline.size

        edge_reduction = max(
            0.0,
            1.0 - (edge_density_current / (edge_density_baseline + 1e-6))
        )

        # --------------------------------------------------
        # 8️⃣ Structural similarity
        # --------------------------------------------------
        ssim_score = ssim(baseline_gray, gray)

        # --------------------------------------------------
        # 9️⃣ Border blockage detection
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 🔟 Histogram comparison
        # --------------------------------------------------
        hist_current = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_baseline = cv2.calcHist([baseline_gray], [0], None, [256], [0, 256])

        hist_correlation = cv2.compareHist(
            hist_current,
            hist_baseline,
            cv2.HISTCMP_CORREL
        )

        # --------------------------------------------------
        # 1️⃣1️⃣ Obstruction percentage
        # --------------------------------------------------
        diff = cv2.absdiff(baseline_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        obstruction_pct = (np.sum(thresh > 0) / thresh.size) * 100.0

        # --------------------------------------------------
        # 1️⃣2️⃣ Decision logic (rotation-safe)
        # --------------------------------------------------
        has_full_obstruction = (
            (uniformity_score > 0.88 and std_dev < 20) or
            (std_dev < 12 and variance < 80) or
            (edge_blockage_score > 0.85 and uniformity_score > 0.75)
        )

        has_partial_obstruction = (
            (edge_reduction > 0.55 and ssim_score < 0.55 and not has_full_obstruction) or
            (edge_blockage_score > 0.6 and obstruction_pct > 25) or
            (hist_correlation < 0.6 and obstruction_pct > 30)
        )

        # --------------------------------------------------
        # 1️⃣3️⃣ Confidence + details
        # --------------------------------------------------
        confidence = 0.0
        details = "No obstruction"

        if has_full_obstruction:
            confidence = min(uniformity_score + (1.0 - ssim_score), 1.0)
            details = (
                f"Full obstruction: uniformity={uniformity_score:.2f}, "
                f"variance={variance:.0f}"
            )

        elif has_partial_obstruction:
            confidence = min(edge_reduction + edge_blockage_score, 1.0) / 2.0
            details = (
                f"Partial obstruction: edge_loss={edge_reduction:.2f}, "
                f"blocked={obstruction_pct:.1f}%"
            )

        logger.info(
            f"[OBSTRUCTION] Full={has_full_obstruction}, "
            f"Partial={has_partial_obstruction}, "
            f"Uniformity={uniformity_score:.2f}, "
            f"EdgeLoss={edge_reduction:.2f}, "
            f"SSIM={ssim_score:.2f}"
        )

        # --------------------------------------------------
        # 1️⃣4️⃣ Return result
        # --------------------------------------------------
        return ObstructionAnalysis(
            has_full_obstruction=has_full_obstruction,
            has_partial_obstruction=has_partial_obstruction,
            obstruction_percentage=obstruction_pct,
            edge_blockage_score=edge_blockage_score,
            uniformity_score=uniformity_score,
            confidence=confidence,
            details=details
        )

    # ==================== ADVANCED BLUR DETECTION ====================
    def detect_blur(self, img_path: Path) -> BlurAnalysis:
        """
        Multi-method blur detection:
        1. Laplacian variance (standard method)
        2. FFT frequency analysis
        3. Gradient magnitude analysis
        4. Edge density and sharpness
        5. Local variance analysis
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return BlurAnalysis(False, False, 0, 0, 0, 0, 0, "Image load error")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (classic method)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 2. FFT Analysis - Sharp images have more high-frequency content
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        
        # Calculate high-frequency energy
        y, x = np.ogrid[:h, :w]
        mask = ((y - center_y)**2 + (x - center_x)**2) > radius**2
        high_freq_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)
        fft_blur_score = high_freq_energy / (total_energy + 1e-6)
        
        # 3. Gradient Magnitude
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_strength = np.mean(gradient_magnitude)
        
        # 4. Edge Density and Sharpness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. Local Variance (sharp images have high local variance)
        kernel_size = 9
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        local_var = sqr_mean - mean**2
        avg_local_var = np.mean(local_var)
        
        # Decision Logic
        is_severely_blurred = (
            (laplacian_var < 50) or
            (fft_blur_score < 0.15 and gradient_strength < 8) or
            (edge_density < 0.02 and laplacian_var < 100)
        )
        
        is_moderately_blurred = (
            (laplacian_var < 150 and fft_blur_score < 0.25) or
            (gradient_strength < 15 and edge_density < 0.05) or
            (avg_local_var < 200 and laplacian_var < 200)
        ) and not is_severely_blurred
        
        confidence = 0.0
        details = "Sharp image"
        
        if is_severely_blurred:
            confidence = max(1 - (laplacian_var / 50), 1 - (fft_blur_score / 0.15))
            details = f"Severe blur: laplacian={laplacian_var:.0f}, fft={fft_blur_score:.3f}"
        elif is_moderately_blurred:
            confidence = max(1 - (laplacian_var / 150), 1 - (fft_blur_score / 0.25))
            details = f"Moderate blur: laplacian={laplacian_var:.0f}, gradient={gradient_strength:.1f}"
        
        logger.info(
            f"[BLUR] Severe={is_severely_blurred}, Moderate={is_moderately_blurred}, "
            f"Laplacian={laplacian_var:.0f}, FFT={fft_blur_score:.3f}, Edges={edge_density:.3f}"
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

    # ==================== ADVANCED POSITION DETECTION ====================
    def detect_position_change(
        self,
        baseline_path: Path,
        current_path: Path
    ) -> PositionAnalysis:
        """
        Detect camera displacement using keypoint matching + homography.
        """

        img_base = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
        img_curr = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)

        if img_base is None or img_curr is None:
            return PositionAnalysis(
                False, False, 0.0, 0.0, 0.0, 0.0, 0.0,
                "Image load error"
            )

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img_base, None)
        kp2, des2 = sift.detectAndCompute(img_curr, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return PositionAnalysis(
                True, False, 1.0, 0.0, 0.0, 0.0, 0.9,
                "Insufficient keypoints — severe position change"
            )

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        match_ratio = len(good) / max(len(kp1), 1)

        is_severe = match_ratio < 0.15
        is_slight = 0.15 <= match_ratio < 0.35

        homography_quality = 0.0
        rotation_angle = 0.0

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                homography_quality = float(np.mean(mask))
                rotation_angle = float(
                    np.degrees(np.arctan2(H[1, 0], H[0, 0]))
                )

        confidence = min(1.0, 1.0 - match_ratio)

        details = (
            f"Keypoint match ratio={match_ratio:.2f}, "
            f"rotation={rotation_angle:.1f}°"
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

    # ==================== CLIP SEMANTIC ANALYSIS ====================
    def analyze_semantics(self, image_path: Path) -> Dict:
        """Enhanced CLIP analysis with temperature scaling"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            img_features = self.model.encode_image(image)
            txt_features = self.model.encode_text(self.tokenized_prompts)
            
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            
            # Temperature scaling for sharper predictions
            logits = (img_features @ txt_features.T) / 0.12
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probs)[-3:][::-1]
        top_probs = probs[top_indices]
        top_prompts = [ALL_PROMPTS[i] for i in top_indices]
        
        return {
            "top_predictions": list(zip(top_prompts, top_probs)),
            "primary_issue": top_prompts[0],
            "confidence": float(top_probs[0])
        }

    # ==================== SIMILARITY ====================
    def compute_clip_similarity(self, img1_path: Path, img2_path: Path) -> float:
        """Compute CLIP semantic similarity"""
        img1 = self.preprocess(Image.open(img1_path)).unsqueeze(0).to(DEVICE)
        img2 = self.preprocess(Image.open(img2_path)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            emb1 = self.model.encode_image(img1)
            emb2 = self.model.encode_image(img2)
            emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
            similarity = (emb1 @ emb2.T).item()
        
        return round(similarity, 4)
    # ==================== SIMILARITY-AWARE CORRECTION ====================
    def apply_similarity_corrections(
        self,
        similarity: float,
        obstruction: ObstructionAnalysis,
        blur: BlurAnalysis,
        position: PositionAnalysis
    ):
        """
        Uses CLIP similarity as a high-level sanity check
        """

        # ---------- POSITION GUARD ----------
        if similarity >= 0.97:
            position.is_severely_displaced = False
            position.is_slightly_displaced = False
            position.confidence = 0.0
            position.details += " | Overridden by high CLIP similarity"

        # ---------- OBSTRUCTION GUARD ----------
        if similarity >= 0.95 and obstruction.has_full_obstruction:
            obstruction.has_full_obstruction = False
            obstruction.confidence *= 0.4
            obstruction.details += " | Full obstruction downgraded by CLIP similarity"

        # ---------- BLUR GUARD ----------
        if similarity >= 0.96 and blur.is_severely_blurred:
            blur.is_severely_blurred = False
            blur.is_moderately_blurred = True
            blur.confidence *= 0.7
            blur.details += " | Blur severity reduced by CLIP similarity"

    def compute_overall_confidence(
        self,
        obstruction: ObstructionAnalysis,
        blur: BlurAnalysis,
        position: PositionAnalysis,
        semantic: Dict
    ) -> float:
        """
        Weighted confidence score (0–1) representing certainty of detected issue
        """
        scores = []

        if obstruction.confidence > 0:
            scores.append(obstruction.confidence * 1.2)

        if blur.confidence > 0:
            scores.append(blur.confidence)

        if position.confidence > 0:
            scores.append(position.confidence * 1.1)

        if semantic.get("confidence", 0) > 0.5:
            scores.append(semantic["confidence"] * 0.8)

        if not scores:
            return 0.0

        return round(min(sum(scores) / len(scores), 1.0), 3)


    # ==================== INTELLIGENT DECISION MAKING ====================
    def generate_comprehensive_summary(
        self,
        similarity,
        obstruction: ObstructionAnalysis,
        blur: BlurAnalysis,
        position: PositionAnalysis,
        semantic: Dict
    ):
        # Priority 1: Position
        if position.is_severely_displaced:
            return "Camera position changed significantly", "MAJOR"
        
        # Priority 2: Obstruction
        if obstruction.has_full_obstruction:
            return "Camera fully obstructed", "CRITICAL"

        if obstruction.has_partial_obstruction:
            return "Camera partially obstructed", "MAJOR"

        # Priority 3: Blur
        if blur.is_severely_blurred:
            return "Severe blur detected", "MAJOR"

        if blur.is_moderately_blurred:
            return "Moderate blur detected", "MINOR"

        

        # Priority 4: Semantic info (non-failure)
        if semantic.get("confidence", 0) > 0.55:
            return f"Scene likely contains: {semantic.get('primary_issue')}", "INFO"

        # Default safe state
        return "Camera functioning normally", "NORMAL"

    
    def run(self, camera_id: str, rtsp_url: str) -> Dict:
        """
        Main entry point called by orchestrator.
        Executes full vision health pipeline.
        """
        start_time = time.time()

        baseline_path = BASELINE_DIR / f"{camera_id}.png"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline image missing for {camera_id}")

        # 1. Capture frame
        captured_path = self.capture_frame(camera_id, rtsp_url)

        # 2. Core analyses
        blur = self.detect_blur(captured_path)

        position = self.detect_position_change(
            baseline_path,
            captured_path
        )

        obstruction = self.detect_obstruction(
            captured_path,
            baseline_path,
            blur
        )

        


        # 3. Semantic + similarity (non-blocking, informative)
        semantic = self.analyze_semantics(captured_path)
        similarity = self.compute_clip_similarity(baseline_path, captured_path)

        self.apply_similarity_corrections(
            similarity,
            obstruction,
            blur,
            position
        )

        if position.is_severely_displaced:
            obstruction.has_full_obstruction = False
            obstruction.has_partial_obstruction = False
            obstruction.confidence = 0.0
            obstruction.details += " | Suppressed due to severe position change"

        # 4. Decision
        summary, severity = self.generate_comprehensive_summary(
            similarity=similarity,
            position=position,
            obstruction=obstruction,
            blur=blur,
            semantic=semantic
        )

        overall_confidence = self.compute_overall_confidence(
            obstruction, blur, position, semantic
        )
        current, peak = tracemalloc.get_traced_memory()

        memory_consumed_mb = round(peak / (1024 * 1024), 3)

        return {
            "camera_id": camera_id,
            "status": "FAIL" if severity != "NORMAL" else "PASS",
            "severity": severity,
            "summary": summary,

            # REQUIRED BY STORAGE
            "confidence_score": overall_confidence,
            "similarity": similarity,

            # Detailed analysis
            "obstruction": obstruction.__dict__,
            "blur": blur.__dict__,
            "position": position.__dict__,
            "semantic_analysis": semantic,

            # REQUIRED BY STORAGE
            "memory_metrics": {
                "memory_consumed_mb": memory_consumed_mb,
                "ram_current_mb": round(current / (1024 * 1024), 3),
                "ram_peak_mb": memory_consumed_mb,
                "cpu_hint": "vision-heavy"
            },

            "timestamp": datetime.utcnow().isoformat(),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
        }

