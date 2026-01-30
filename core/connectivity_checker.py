"""
Connectivity Checker - Network Verification, Latency & Packet Loss
New service for connectivity checks

File: core/connectivity_checker.py
"""

import asyncio
import re
import logging
import platform
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectivityChecker:
    """Service for performing connectivity checks on cameras"""
    
    @staticmethod
    async def check_api_alive(ip: str, port: int = 554, timeout: int = 5) -> Dict:
        """
        Task 1: Network Verification
        Check if the camera API/service is alive and responding
        
        Args:
            ip: Camera IP address
            port: Service port (default RTSP port 554)
            timeout: Timeout in seconds
        
        Returns:
            Dictionary with connectivity status
        """
        try:
            # Try to establish TCP connection
            future = asyncio.open_connection(ip, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            
            # Connection successful
            writer.close()
            await writer.wait_closed()
            
            return {
                "is_alive": True,
                "ip": ip,
                "port": port,
                "status": "ONLINE",
                "message": f"Service is reachable at {ip}:{port}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                "is_alive": False,
                "ip": ip,
                "port": port,
                "status": "TIMEOUT",
                "message": f"Connection timeout after {timeout}s",
                "timestamp": datetime.utcnow().isoformat()
            }
        except ConnectionRefusedError:
            return {
                "is_alive": False,
                "ip": ip,
                "port": port,
                "status": "REFUSED",
                "message": "Connection refused - service may not be running",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "is_alive": False,
                "ip": ip,
                "port": port,
                "status": "ERROR",
                "message": f"Connection error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def measure_latency(ip: str, count: int = 10, timeout: int = 5) -> Dict:
        """
        Task 2: Latency Measurement
        Measure Round Trip Time (RTT) using ICMP ping
        
        Args:
            ip: Target IP address
            count: Number of ping packets to send (default 10)
            timeout: Timeout for each ping in seconds
        
        Returns:
            Dictionary with latency metrics (avg, min, max RTT in ms)
        """
        try:
            # Determine the ping command based on OS
            system = platform.system().lower()
            
            if system == "windows":
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), ip]
            else:  # Linux/Unix
                cmd = ["ping", "-c", str(count), "-W", str(timeout), ip]
            
            # Execute ping command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            output = stdout.decode()
            
            # Parse ping output
            latency_data = ConnectivityChecker._parse_ping_output(output, system)
            
            if latency_data["success"]:
                return {
                    "success": True,
                    "ip": ip,
                    "rtt_avg": latency_data["rtt_avg"],
                    "rtt_min": latency_data["rtt_min"],
                    "rtt_max": latency_data["rtt_max"],
                    "packets_sent": count,
                    "packets_received": latency_data.get("packets_received", 0),
                    "is_reachable": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Latency measurement successful"
                }
            else:
                return {
                    "success": False,
                    "ip": ip,
                    "rtt_avg": None,
                    "rtt_min": None,
                    "rtt_max": None,
                    "packets_sent": count,
                    "packets_received": 0,
                    "is_reachable": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": latency_data.get("error", "Failed to measure latency")
                }
                
        except Exception as e:
            logger.error(f"Latency measurement failed for {ip}: {e}")
            return {
                "success": False,
                "ip": ip,
                "rtt_avg": None,
                "rtt_min": None,
                "rtt_max": None,
                "packets_sent": count,
                "packets_received": 0,
                "is_reachable": False,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Error: {str(e)}"
            }
    
    @staticmethod
    def _parse_ping_output(output: str, system: str) -> Dict:
        """
        Parse ping command output to extract RTT statistics
        
        Args:
            output: Raw ping command output
            system: Operating system ('windows', 'linux', etc.)
        
        Returns:
            Dictionary with parsed latency values
        """
        try:
            if system == "windows":
                # Windows ping output pattern
                match = re.search(
                    r"Minimum = (\d+)ms, Maximum = (\d+)ms, Average = (\d+)ms",
                    output
                )
                if match:
                    return {
                        "success": True,
                        "rtt_min": float(match.group(1)),
                        "rtt_max": float(match.group(2)),
                        "rtt_avg": float(match.group(3))
                    }
            else:
                # Linux/Unix ping output pattern
                match = re.search(
                    r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms",
                    output
                )
                if match:
                    return {
                        "success": True,
                        "rtt_min": float(match.group(1)),
                        "rtt_avg": float(match.group(2)),
                        "rtt_max": float(match.group(3))
                    }
                
                # Alternative pattern for some Linux versions
                match = re.search(
                    r"min/avg/max = ([\d.]+)/([\d.]+)/([\d.]+) ms",
                    output
                )
                if match:
                    return {
                        "success": True,
                        "rtt_min": float(match.group(1)),
                        "rtt_avg": float(match.group(2)),
                        "rtt_max": float(match.group(3))
                    }
            
            # If no pattern matched, check if host is unreachable
            if "unreachable" in output.lower() or "100% packet loss" in output.lower():
                return {
                    "success": False,
                    "error": "Host unreachable or 100% packet loss"
                }
            
            return {
                "success": False,
                "error": "Could not parse ping output"
            }
            
        except Exception as e:
            logger.error(f"Error parsing ping output: {e}")
            return {
                "success": False,
                "error": f"Parse error: {str(e)}"
            }
    
    @staticmethod
    async def calculate_packet_loss(ip: str, count: int = 100, timeout: int = 5) -> Dict:
        """
        Task 3: Packet Loss Calculation
        Calculate packet loss percentage
        
        Args:
            ip: Target IP address
            count: Number of packets to send (default 100 for accuracy)
            timeout: Timeout for each packet
        
        Returns:
            Dictionary with packet loss statistics
        """
        try:
            # Determine the ping command based on OS
            system = platform.system().lower()
            
            if system == "windows":
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), ip]
            else:  # Linux/Unix
                cmd = ["ping", "-c", str(count), "-W", str(timeout), ip]
            
            # Execute ping command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            output = stdout.decode()
            
            # Parse packet loss
            loss_data = ConnectivityChecker._parse_packet_loss(output, system, count)
            
            return {
                "success": loss_data["success"],
                "ip": ip,
                "packets_sent": count,
                "packets_received": loss_data.get("packets_received", 0),
                "packet_loss_percent": loss_data.get("packet_loss_percent", 100.0),
                "timestamp": datetime.utcnow().isoformat(),
                "message": loss_data.get("message", "Packet loss calculation completed")
            }
            
        except Exception as e:
            logger.error(f"Packet loss calculation failed for {ip}: {e}")
            return {
                "success": False,
                "ip": ip,
                "packets_sent": count,
                "packets_received": 0,
                "packet_loss_percent": 100.0,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Error: {str(e)}"
            }
    
    @staticmethod
    def _parse_packet_loss(output: str, system: str, sent: int) -> Dict:
        """
        Parse packet loss from ping output
        
        Args:
            output: Raw ping output
            system: Operating system
            sent: Number of packets sent
        
        Returns:
            Dictionary with packet loss data
        """
        try:
            if system == "windows":
                # Windows pattern
                match = re.search(
                    r"Packets: Sent = (\d+), Received = (\d+), Lost = (\d+) \((\d+)% loss\)",
                    output
                )
                if match:
                    received = int(match.group(2))
                    loss_percent = float(match.group(4))
                    return {
                        "success": True,
                        "packets_received": received,
                        "packet_loss_percent": loss_percent,
                        "message": f"{loss_percent}% packet loss"
                    }
            else:
                # Linux/Unix pattern
                match = re.search(
                    r"(\d+) packets transmitted, (\d+) received, ([\d.]+)% packet loss",
                    output
                )
                if match:
                    received = int(match.group(2))
                    loss_percent = float(match.group(3))
                    return {
                        "success": True,
                        "packets_received": received,
                        "packet_loss_percent": loss_percent,
                        "message": f"{loss_percent}% packet loss"
                    }
            
            # Fallback: assume 100% loss if can't parse
            return {
                "success": False,
                "packets_received": 0,
                "packet_loss_percent": 100.0,
                "message": "Could not parse packet loss - assuming 100%"
            }
            
        except Exception as e:
            logger.error(f"Error parsing packet loss: {e}")
            return {
                "success": False,
                "packets_received": 0,
                "packet_loss_percent": 100.0,
                "message": f"Parse error: {str(e)}"
            }
    
    