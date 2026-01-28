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
import time
import statistics


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
            'rtsp_port': None,
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
            rtsp_port = rtsp_info['port']
            result['rtsp_port'] = rtsp_port
            
            logger.info(f"Checking RTSP port {rtsp_port} on {ip}...")
            result['port_ok'] = self.check_port(ip, rtsp_port)
            
            if not result['port_ok']:
                result['status'] = 'OFFLINE'
                result['message'] = f"RTSP port {rtsp_port} not accessible"
                return result
        
        # Both checks passed
        result['status'] = 'ONLINE'
        result['message'] = 'Camera network is healthy'
        return result


    def ping_with_metrics(self, ip: str, count: int = 5) -> Dict:
        """
        Ping IP multiple times and return latency + packet loss metrics
        """
        sent = count
        received = 0
        rtts = []

        for _ in range(count):
            start = time.time()
            success = self.ping_ip(ip)
            end = time.time()

            if success:
                received += 1
                rtts.append((end - start) * 1000)  # ms

            time.sleep(0.2)  # small gap between pings

        packet_loss_percent = round(((sent - received) / sent) * 100, 2)

        if rtts:
            rtt_avg = round(statistics.mean(rtts), 2)
            rtt_min = round(min(rtts), 2)
            rtt_max = round(max(rtts), 2)
        else:
            rtt_avg = rtt_min = rtt_max = None

        return {
            "is_reachable": received > 0,
            "packets_sent": sent,
            "packets_received": received,
            "packet_loss_percent": packet_loss_percent,
            "rtt_avg_ms": rtt_avg,
            "rtt_min_ms": rtt_min,
            "rtt_max_ms": rtt_max
        }


    def check_port_with_latency(self, ip: str, port: int, attempts: int = 3) -> Dict:
        """
        Check port multiple times and measure connection latency
        """
        success_count = 0
        latencies = []

        for _ in range(attempts):
            start = time.time()
            ok = self.check_port(ip, port)
            end = time.time()

            if ok:
                success_count += 1
                latencies.append((end - start) * 1000)  # ms

            time.sleep(0.2)

        success_rate = round((success_count / attempts) * 100, 2)

        if latencies:
            latency_avg = round(statistics.mean(latencies), 2)
            latency_min = round(min(latencies), 2)
            latency_max = round(max(latencies), 2)
        else:
            latency_avg = latency_min = latency_max = None

        return {
            "is_accessible": success_count > 0,
            "attempts": attempts,
            "success_count": success_count,
            "connection_success_rate": success_rate,
            "latency_avg_ms": latency_avg,
            "latency_min_ms": latency_min,
            "latency_max_ms": latency_max
        }


    def verify_connectivity(self, ip: str, port: int, ping_count: int = 5) -> Dict:
        """
        Full connectivity verification (IP + Port with metrics)
        """

        ping_metrics = self.ping_with_metrics(ip, ping_count)
        port_metrics = self.check_port_with_latency(ip, port, attempts=3)

        network_reachable = ping_metrics["is_reachable"]
        service_accessible = port_metrics["is_accessible"]

        if network_reachable and service_accessible:
            overall_status = "HEALTHY"
        elif network_reachable and not service_accessible:
            overall_status = "NETWORK_OK_SERVICE_DOWN"
        else:
            overall_status = "OFFLINE"

        return {
            "overall_status": overall_status,
            "network_reachable": network_reachable,
            "service_accessible": service_accessible,
            "ping_metrics": ping_metrics,
            "port_metrics": port_metrics
        }


# Example usage
if __name__ == "__main__":
    checker = PingChecker()
    
    