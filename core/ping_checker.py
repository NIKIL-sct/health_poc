"""
Network Health Checker for Cameras
Validates IP reachability and RTSP port availability across platforms
"""

import logging
import platform
import socket
import subprocess
from typing import Dict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PingChecker:
    """Cross-platform network health validation for cameras"""
    
    def __init__(self, timeout: int = 3, port_timeout: int = 2):
        self.timeout = timeout
        self.port_timeout = port_timeout
        self.system = platform.system().lower()
        
    def ping_ip(self, ip: str) -> bool:
        """
        Cross-platform ping check
        Returns: True if IP is reachable, False otherwise
        """
        try:
            # Determine ping command based on OS
            if self.system == 'windows':
                cmd = ['ping', '-n', '1', '-w', str(self.timeout * 1000), ip]
            elif self.system == 'darwin':  # macOS
                cmd = ['ping', '-c', '1', '-W', str(self.timeout * 1000), ip]
            else:  # Linux and others
                cmd = ['ping', '-c', '1', '-W', str(self.timeout), ip]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout + 1
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Ping timeout for {ip}")
            return False
        except Exception as e:
            logger.error(f"Ping error for {ip}: {e}")
            return False
    
    def check_port(self, ip: str, port: int) -> bool:
        """
        Check if a specific port is open on the target IP
        Returns: True if port is open, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.port_timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"Port check error for {ip}:{port} - {e}")
            return False
    
    def extract_rtsp_info(self, rtsp_url: str) -> Dict:
        """
        Extract IP and port from RTSP URL
        Returns: Dict with 'ip' and 'port'
        """
        try:
            parsed = urlparse(rtsp_url)
            ip = parsed.hostname or ''
            port = parsed.port or 554  
            return {'ip': ip, 'port': port}
        except Exception as e:
            logger.error(f"RTSP URL parse error: {e}")
            return {'ip': '', 'port': 554}
    
    def check_camera_network(self, ip: str, rtsp_url: str = '') -> Dict:
        """
        Complete network health check for a camera
        Returns: Comprehensive network status report
        """
        result = {
            'ip': ip,
            'status': 'UNKNOWN',
            'ping_ok': False,
            'port_ok': False,
            'port': None,
            'message': ''
        }
        
        # Step 1: Ping check
        logger.info(f"Pinging {ip}...")
        result['ping_ok'] = self.ping_ip(ip)
        
        if not result['ping_ok']:
            result['status'] = 'OFFLINE'
            result['message'] = f"IP {ip} is not reachable"
            return result
        
        # Step 2: Port check (if RTSP URL provided)
        if rtsp_url:
            rtsp_info = self.extract_rtsp_info(rtsp_url)
            port = rtsp_info['port']
            result['port'] = port
            
            logger.info(f"Checking RTSP port {port} on {ip}...")
            result['port_ok'] = self.check_port(ip, port)
            
            if not result['port_ok']:
                result['status'] = 'OFFLINE'
                result['message'] = f"RTSP port {port} not accessible"
                return result
        
        # Both checks passed
        result['status'] = 'ONLINE'
        result['message'] = 'Camera network is healthy'
        return result


# Example usage
if __name__ == "__main__":
    checker = PingChecker()
    
    # Define test cameras (FIXED: was undefined)
    test_cameras = [
        {
            "ip": "192.168.1.202",
            "rtsp_url": "rtsp://Rohit:7995642622%40Ch@192.168.1.202:554/"
        },
        {
            "ip": "8.8.8.8",
            "rtsp_url": ""  # Just ping test
        }
    ]
    
    for cam in test_cameras:
        print(f"\n=== Testing {cam['ip']} ===")
        status = checker.check_camera_network(cam['ip'], cam['rtsp_url'])
        print(f"Status: {status['status']}")
        print(f"Ping OK: {status['ping_ok']}")
        print(f"Port OK: {status['port_ok']}")
        print(f"Message: {status['message']}")