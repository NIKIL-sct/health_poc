"""
Enhanced Vision Result Storage Module - Bulletproof Edition
Handles all vision check results with comprehensive error handling.
Features:
- Validates input before processing
- Never crashes on None or invalid input
- Provides detailed logging for debugging
- Atomic file writes to prevent corruption
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vision-storage-enhanced")

# -------------------------------------------------------------------
# Vision Result Validator
# -------------------------------------------------------------------
class VisionResultValidator:
    """Validates vision check results before storage"""
    
    @staticmethod
    def validate(vision_result: Any) -> tuple[bool, str]:
        """
        Validate vision result structure.
        
        Args:
            vision_result: Vision check result to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if None
        if vision_result is None:
            return False, "Vision result is None"
        
        # Check if dictionary
        if not isinstance(vision_result, dict):
            return False, f"Vision result is not a dict, got: {type(vision_result)}"
        
        # Check required fields
        required_fields = [
            "camera_id", "similarity", "status", "summary", 
            "confidence_score", "timestamp"
        ]
        
        missing_fields = [f for f in required_fields if f not in vision_result]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Validate field types
        if not isinstance(vision_result.get("camera_id"), str):
            return False, "camera_id must be string"
        
        if not isinstance(vision_result.get("similarity"), (int, float)):
            return False, "similarity must be numeric"
        
        if vision_result.get("status") not in ["PASS", "FAIL", "ERROR", "UNKNOWN"]:
            return False, f"Invalid status: {vision_result.get('status')}"
        
        return True, "Valid"

# -------------------------------------------------------------------
# Vision Result Storage
# -------------------------------------------------------------------
def store_vision_result(
    camera_id: str, 
    vision_result: Optional[Dict[str, Any]]
) -> bool:
    """
    Persist vision check result with bulletproof error handling.
    
    Args:
        camera_id: Unique camera identifier
        vision_result: Vision check result dictionary (can be None)
        
    Returns:
        True if storage succeeded, False otherwise
    """
    health_file = LOGS_DIR / f"{camera_id}_health.json"

    try:
        # Step 1: Validate input
        is_valid, error_msg = VisionResultValidator.validate(vision_result)
        
        if not is_valid:
            logger.error(
                f"[STORAGE] Invalid vision result for {camera_id}: {error_msg}"
            )
            # Store error entry instead of skipping
            vision_result = _create_error_entry(camera_id, error_msg)
        
        # Step 2: Load existing health log
        data = _load_health_log(camera_id, health_file)
        
        # Step 3: Ensure required sections exist
        _ensure_required_sections(data)
        
        # Step 4: Create vision check entry
        vision_entry = _create_vision_entry(vision_result)
        
        # Step 5: Append to history
        data["vision_checks"].append(vision_entry)
        
        # Keep last 100 entries
        if len(data["vision_checks"]) > 100:
            data["vision_checks"] = data["vision_checks"][-100:]
        
        # Step 6: Update statistics
        _update_vision_statistics(data["vision_statistics"], vision_entry)
        
        # Step 7: Update latest check
        data["latest_vision_check"] = vision_entry
        
        # Step 8: Write atomically
        _atomic_write(health_file, data)
        
        logger.info(
            f"[STORAGE] ✓ Stored vision result for {camera_id}: "
            f"Status={vision_entry['status']}, "
            f"Similarity={vision_entry['similarity']:.4f}, "
            f"Confidence={vision_entry['confidence_score']:.4f}"
        )
        
        return True
        
    except Exception as e:
        logger.error(
            f"[STORAGE] ✗ Failed to store vision result for {camera_id}: {str(e)}", 
            exc_info=True
        )
        return False

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def _create_error_entry(camera_id: str, error_msg: str) -> Dict[str, Any]:
    """Create error entry when vision result is invalid"""
    return {
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat(),
        "similarity": 0.0,
        "status": "ERROR",
        "summary": f"Storage error: {error_msg}",
        "confidence_score": 0.0,
        "check_duration_ms": 0.0,
        "baseline_image": "",
        "captured_image": "",
        "memory_metrics": {},
        "error": error_msg
    }

def _load_health_log(camera_id: str, health_file: Path) -> Dict[str, Any]:
    """Load existing health log or create new one"""
    if health_file.exists():
        try:
            with open(health_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"[STORAGE] Loaded existing log for {camera_id}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(
                f"[STORAGE] Corrupted log for {camera_id}, creating new. Error: {e}"
            )
            return _create_empty_health_log(camera_id)
        except Exception as e:
            logger.error(
                f"[STORAGE] Error loading log for {camera_id}: {e}, creating new"
            )
            return _create_empty_health_log(camera_id)
    else:
        logger.debug(f"[STORAGE] Creating new log for {camera_id}")
        return _create_empty_health_log(camera_id)

def _ensure_required_sections(data: Dict[str, Any]) -> None:
    """Ensure all required sections exist in data"""
    if "vision_checks" not in data:
        data["vision_checks"] = []
    if "vision_statistics" not in data:
        data["vision_statistics"] = _create_empty_vision_stats()
    if "latest_vision_check" not in data:
        data["latest_vision_check"] = None

def _create_vision_entry(vision_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized vision check entry"""
    return {
        "timestamp": vision_result.get("timestamp", datetime.now().isoformat()),
        "similarity": float(vision_result.get("similarity", 0.0)),
        "status": vision_result.get("status", "UNKNOWN"),
        "summary": vision_result.get("summary", "No summary available"),
        "confidence_score": float(vision_result.get("confidence_score", 0.0)),
        "check_duration_ms": float(vision_result.get("check_duration_ms", 0.0)),
        "baseline_image": vision_result.get("baseline_image", ""),
        "captured_image": vision_result.get("captured_image", ""),
        "memory_metrics": vision_result.get("memory_metrics", {}),
        "debug_metrics": vision_result.get("debug_metrics", {})
    }

def _atomic_write(health_file: Path, data: Dict[str, Any]) -> None:
    """Write data atomically to prevent corruption"""
    temp_file = health_file.with_suffix('.tmp')
    
    try:
        # Write to temp file
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic replace
        temp_file.replace(health_file)
        
    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise

def _create_empty_health_log(camera_id: str) -> Dict[str, Any]:
    """Create empty health log structure"""
    return {
        "camera_id": camera_id,
        "camera_info": {
            "camera_id": camera_id,
            "registered_at": datetime.now().isoformat()
        },
        "latest_health_check": None,
        "health_check_history": [],
        "vision_checks": [],
        "vision_statistics": _create_empty_vision_stats(),
        "latest_vision_check": None
    }

def _create_empty_vision_stats() -> Dict[str, Any]:
    """Create empty vision statistics"""
    return {
        "total_checks": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_errors": 0,
        "pass_rate": 0.0,
        "fail_rate": 0.0,
        "error_rate": 0.0,
        "average_similarity": 0.0,
        "min_similarity": 1.0,
        "max_similarity": 0.0,
        "average_confidence": 0.0,
        "min_confidence": 1.0,
        "max_confidence": 0.0,
        "average_check_duration_ms": 0.0,
        "total_memory_consumed_mb": 0.0,
        "average_memory_per_check_mb": 0.0,
        "first_check": None,
        "last_check": None
    }

def _update_vision_statistics(
    stats: Dict[str, Any], 
    vision_entry: Dict[str, Any]
) -> None:
    """Update statistics with new check result"""
    stats.setdefault("total_checks", 0)
    stats.setdefault("total_passed", 0)
    stats.setdefault("total_failed", 0)
    stats.setdefault("total_errors", 0)
    stats.setdefault("pass_rate", 0.0)
    stats.setdefault("fail_rate", 0.0)
    stats.setdefault("error_rate", 0.0)
    # Update counts
    stats["total_checks"] += 1
    
    status = vision_entry.get("status", "UNKNOWN")
    if status == "PASS":
        stats["total_passed"] += 1
    elif status == "FAIL":
        stats["total_failed"] += 1
    elif status == "ERROR":
        stats["total_errors"] += 1
    
    # Update rates
    total = stats["total_checks"]
    if total > 0:
        stats["pass_rate"] = round(stats["total_passed"] / total, 4)
        stats["fail_rate"] = round(stats["total_failed"] / total, 4)
        stats["error_rate"] = round(stats["total_errors"] / total, 4)
    
    # Update similarity statistics
    new_sim = vision_entry.get("similarity", 0.0)
    if new_sim > 0:  # Only update if valid similarity
        prev_avg_sim = stats.get("average_similarity", 0.0)
        stats["average_similarity"] = round(
            ((prev_avg_sim * (total - 1)) + new_sim) / total, 4
        )
        
        stats["min_similarity"] = round(
            min(stats.get("min_similarity", 1.0), new_sim), 4
        )
        stats["max_similarity"] = round(
            max(stats.get("max_similarity", 0.0), new_sim), 4
        )
    
    # Update confidence statistics
    new_conf = vision_entry.get("confidence_score", 0.0)
    if new_conf > 0:  # Only update if valid confidence
        prev_avg_conf = stats.get("average_confidence", 0.0)
        stats["average_confidence"] = round(
            ((prev_avg_conf * (total - 1)) + new_conf) / total, 4
        )
        
        stats["min_confidence"] = round(
            min(stats.get("min_confidence", 1.0), new_conf), 4
        )
        stats["max_confidence"] = round(
            max(stats.get("max_confidence", 0.0), new_conf), 4
        )
    
    # Update duration statistics
    prev_avg_duration = stats.get("average_check_duration_ms", 0.0)
    new_duration = vision_entry.get("check_duration_ms", 0.0)
    if new_duration > 0:
        stats["average_check_duration_ms"] = round(
            ((prev_avg_duration * (total - 1)) + new_duration) / total, 2
        )
    
    # Update memory statistics
    memory_metrics = vision_entry.get("memory_metrics", {})
    memory_consumed = memory_metrics.get("memory_consumed_mb", 0.0)
    
    if memory_consumed > 0:
        prev_total_memory = stats.get("total_memory_consumed_mb", 0.0)
        stats["total_memory_consumed_mb"] = round(
            prev_total_memory + memory_consumed, 2
        )
        
        if total > 0:
            stats["average_memory_per_check_mb"] = round(
                stats["total_memory_consumed_mb"] / total, 4
            )
    
    # Update timestamps
    current_time = vision_entry.get("timestamp", datetime.now().isoformat())
    if stats["first_check"] is None:
        stats["first_check"] = current_time
    stats["last_check"] = current_time

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def get_vision_statistics(camera_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve vision statistics for a camera.
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        Statistics dictionary or None if not found
    """
    health_file = LOGS_DIR / f"{camera_id}_health.json"
    
    if not health_file.exists():
        logger.warning(f"[STORAGE] No health log found for {camera_id}")
        return None
    
    try:
        with open(health_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("vision_statistics")
    except Exception as e:
        logger.error(f"[STORAGE] Error reading statistics for {camera_id}: {e}")
        return None

def get_latest_vision_check(camera_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve latest vision check for a camera.
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        Latest check dictionary or None if not found
    """
    health_file = LOGS_DIR / f"{camera_id}_health.json"
    
    if not health_file.exists():
        return None
    
    try:
        with open(health_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("latest_vision_check")
    except Exception as e:
        logger.error(f"[STORAGE] Error reading latest check for {camera_id}: {e}")
        return None

def get_vision_check_history(
    camera_id: str, 
    limit: int = 10
) -> list[Dict[str, Any]]:
    """
    Retrieve vision check history for a camera.
    
    Args:
        camera_id: Camera identifier
        limit: Maximum number of checks to return
        
    Returns:
        List of vision check entries (most recent first)
    """
    health_file = LOGS_DIR / f"{camera_id}_health.json"
    
    if not health_file.exists():
        return []
    
    try:
        with open(health_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        checks = data.get("vision_checks", [])
        return checks[-limit:][::-1]  # Return most recent first
        
    except Exception as e:
        logger.error(f"[STORAGE] Error reading history for {camera_id}: {e}")
        return []