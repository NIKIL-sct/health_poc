"""
Vision Storage Module
Reliable persistence layer for camera vision health check results
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import fcntl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionStorage:
    """Single source of truth for vision health check persistence"""
    
    def __init__(self, log_dir: str = "logs", max_history: int = 1000):
        self.log_dir = log_dir
        self.max_history = max_history
        self.stats_file = os.path.join(log_dir, "vision_stats.json")
        
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_camera_log_path(self, camera_id: str) -> str:
        """Get the log file path for a specific camera"""
        return os.path.join(self.log_dir, f"{camera_id}_vision.json")
    
    def _atomic_write(self, filepath: str, data: Dict):
        """
        Atomic write with file locking to prevent corruption
        """
        temp_file = f"{filepath}.tmp"
        
        try:
            # Write to temporary file
            with open(temp_file, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            os.replace(temp_file, filepath)
            
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
    
    def _load_json(self, filepath: str) -> Dict:
        """Safely load JSON file with error handling"""
        if not os.path.exists(filepath):
            return {}
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Corrupted JSON file: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    def store_vision_result(self, camera_id: str, result: Dict) -> bool:
        """
        Store vision check result for a camera
        Returns: True if successful, False otherwise
        """
        try:
            log_path = self._get_camera_log_path(camera_id)
            
            # Load existing history
            camera_log = self._load_json(log_path)
            if 'history' not in camera_log:
                camera_log = {
                    'camera_id': camera_id,
                    'history': [],
                    'last_updated': None
                }
            
            # Add new result
            camera_log['history'].append(result)
            camera_log['last_updated'] = datetime.now().isoformat()
            
            # Trim history if exceeds max
            if len(camera_log['history']) > self.max_history:
                camera_log['history'] = camera_log['history'][-self.max_history:]
            
            # Atomic write
            self._atomic_write(log_path, camera_log)
            
            # Update aggregated stats
            self._update_stats(camera_id, result)
            
            logger.info(f"[{camera_id}] Vision result stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to store result: {e}")
            return False
    
    def _update_stats(self, camera_id: str, result: Dict):
        """Update aggregated statistics"""
        try:
            stats = self._load_json(self.stats_file)
            
            if 'cameras' not in stats:
                stats = {
                    'cameras': {},
                    'last_updated': None
                }
            
            if camera_id not in stats['cameras']:
                stats['cameras'][camera_id] = {
                    'total_checks': 0,
                    'passed': 0,
                    'failed': 0,
                    'avg_similarity': 0,
                    'total_memory_mb': 0,
                    'avg_execution_time': 0
                }
            
            cam_stats = stats['cameras'][camera_id]
            
            # Update counts
            cam_stats['total_checks'] += 1
            if result.get('status') == 'PASS':
                cam_stats['passed'] += 1
            else:
                cam_stats['failed'] += 1
            
            # Update metrics (running average)
            baseline = result.get('checks', {}).get('baseline_comparison', {})
            if 'similarity' in baseline:
                old_avg = cam_stats.get('avg_similarity', 0)
                n = cam_stats['total_checks']
                cam_stats['avg_similarity'] = (old_avg * (n - 1) + baseline['similarity']) / n
            
            # Update performance metrics
            perf = result.get('performance', {})
            if 'memory_used_mb' in perf:
                cam_stats['total_memory_mb'] += perf['memory_used_mb']
            
            if 'execution_time_sec' in perf:
                old_avg = cam_stats.get('avg_execution_time', 0)
                n = cam_stats['total_checks']
                cam_stats['avg_execution_time'] = (old_avg * (n - 1) + perf['execution_time_sec']) / n
            
            # Calculate pass rate
            cam_stats['pass_rate'] = (cam_stats['passed'] / cam_stats['total_checks']) * 100
            
            stats['last_updated'] = datetime.now().isoformat()
            
            self._atomic_write(self.stats_file, stats)
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
    
    def get_camera_history(self, camera_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent vision check history for a camera
        Returns: List of recent check results
        """
        try:
            log_path = self._get_camera_log_path(camera_id)
            camera_log = self._load_json(log_path)
            
            history = camera_log.get('history', [])
            return history[-limit:] if history else []
            
        except Exception as e:
            logger.error(f"Error retrieving history for {camera_id}: {e}")
            return []
    
    def get_camera_stats(self, camera_id: str) -> Dict:
        """Get aggregated statistics for a camera"""
        try:
            stats = self._load_json(self.stats_file)
            return stats.get('cameras', {}).get(camera_id, {})
        except Exception as e:
            logger.error(f"Error retrieving stats for {camera_id}: {e}")
            return {}
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all cameras"""
        try:
            return self._load_json(self.stats_file)
        except Exception as e:
            logger.error(f"Error retrieving all stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    storage = VisionStorage()
    
    # Sample vision check result
    sample_result = {
        'camera_id': 'CAM_001',
        'timestamp': datetime.now().isoformat(),
        'status': 'PASS',
        'checks': {
            'brightness': {'status': 'PASS', 'value': 128.5},
            'blur': {'status': 'PASS', 'value': 245.8},
            'baseline_comparison': {
                'status': 'PASS',
                'similarity': 0.92,
                'difference_percent': 8.3
            }
        },
        'alerts': [],
        'performance': {
            'execution_time_sec': 2.45,
            'memory_used_mb': 45.2
        }
    }
    
    # Store result
    success = storage.store_vision_result('CAM_001', sample_result)
    print(f"Storage successful: {success}")
    
    # Retrieve history
    history = storage.get_camera_history('CAM_001', limit=5)
    print(f"\nRecent checks: {len(history)}")
    
    # Get stats
    stats = storage.get_camera_stats('CAM_001')
    print(f"\nCamera Stats:")
    print(f"  Total Checks: {stats.get('total_checks', 0)}")
    print(f"  Pass Rate: {stats.get('pass_rate', 0):.1f}%")
    print(f"  Avg Similarity: {stats.get('avg_similarity', 0):.3f}")
