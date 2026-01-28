"""
API Endpoints for Enhanced Network Connectivity Features
Add these endpoints to your existing app.py

These endpoints provide:
1. On-demand connectivity checks with full metrics
2. Historical metrics retrieval
3. Network health reports
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from dto.dto_validator import (
    ConnectivityCheckRequestDTO,
    ConnectivityCheckResponseDTO,
    CameraSummaryDTO,
    MetricsHistoryDTO,
    NetworkHealthReportDTO
)
from storage.db_config import get_db_dependency
from storage.db_repositary import CameraRepository, HealthLogRepository
from storage.redis_client import get_async_redis, RedisData
from core.ping_checker import PingChecker

logger = logging.getLogger(__name__)

# Create router for connectivity endpoints
connectivity_router = APIRouter(prefix="/health/connectivity", tags=["connectivity"])


@connectivity_router.post("/check", response_model=ConnectivityCheckResponseDTO)
async def check_connectivity(
    request: ConnectivityCheckRequestDTO,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Perform on-demand connectivity check with full metrics
    
    This endpoint:
    - Verifies network reachability (ping with RTT measurement)
    - Checks service accessibility (port with latency measurement)
    - Calculates packet loss and connection success rates
    - Returns comprehensive connectivity report
    """
    camera_id = request.camera_id
    
    # Get camera from database
    camera = await CameraRepository.get_by_id_async(db, camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found"
        )
    
    if not camera.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' is disabled"
        )
    
    # Prepare camera dict for PingChecker
    camera_dict = {
        "id": camera.camera_id,
        "ip": camera.ip,
        "rtsp_port": camera.port,
        "rtsp_url": camera.rtsp_url
    }
    
    # Perform connectivity check
    ping_checker = PingChecker()
    
    try:
        # Extract RTSP info
        rtsp_info = ping_checker.extract_rtsp_info(camera.rtsp_url)
        rtsp_ip = rtsp_info["ip"] or camera.ip
        rtsp_port = rtsp_info["port"]
        
        # Run full connectivity verification
        result = ping_checker.verify_connectivity(
            rtsp_ip,
            rtsp_port,
            ping_count=request.ping_count
        )
        
        # Build response
        response = ConnectivityCheckResponseDTO(
            camera_id=camera_id,
            target_ip=camera.ip,
            rtsp_ip=rtsp_ip,
            rtsp_port=rtsp_port,
            connectivity=result,
            checked_at=datetime.now()
        )
        
        logger.info(
            f"[{camera_id}] On-demand connectivity check completed: {result['overall_status']}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Connectivity check failed for {camera_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Connectivity check failed: {str(e)}"
        )


@connectivity_router.get("/summary/{camera_id}", response_model=CameraSummaryDTO)
async def get_connectivity_summary(
    camera_id: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get latest connectivity summary with metrics for a camera
    
    Returns:
    - Current status of IP, port, and vision checks
    - Latest metrics (latency, packet loss, success rates)
    - Overall connectivity status
    - Alert status
    """
    # Verify camera exists
    camera = await CameraRepository.get_by_id_async(db, camera_id)
    
    if not camera:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found"
        )
    
    try:
        # Get summary from Redis (fastest)
        r = await get_async_redis()
        summary = await RedisData.get_summary(r, camera_id)
        
        if not summary:
            # Fallback to database
            from storage.db_repositary import SummaryRepository
            summary = await SummaryRepository.get_latest_summary_async(db, camera_id)
        
        # Build response
        response = CameraSummaryDTO(
            camera_id=camera_id,
            ip_status=summary.get("Ip_status", "UNKNOWN"),
            port_status=summary.get("Port_status", "UNKNOWN"),
            vision_status=summary.get("Vision_status", "UNKNOWN"),
            overall_status=summary.get("Overall_status", "UNKNOWN"),
            network_reachable=summary.get("Network_reachable", False),
            service_accessible=summary.get("Service_accessible", False),
            alert_active=summary.get("Alert_active", False),
            last_ip_check=summary.get("Last_ip_check"),
            last_port_check=summary.get("Last_port_check"),
            last_vision_check=summary.get("Last_vision_check"),
            last_connectivity_check=summary.get("Last_connectivity_check"),
            ip_metrics=summary.get("Ip_metrics", {}),
            port_metrics=summary.get("Port_metrics", {}),
            connectivity_metrics=summary.get("Connectivity_metrics", {})
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get summary for {camera_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve summary: {str(e)}"
        )



# ============================================
# Helper Functions
# ============================================

def _calculate_metrics_summary(data_points: List, metric_type: str) -> dict:
    """Calculate summary statistics from data points"""
    if not data_points:
        return {"message": "No data available"}
    
    summary = {
        "total_checks": len(data_points),
        "success_count": sum(1 for dp in data_points if dp["status"] == "PASS"),
        "failure_count": sum(1 for dp in data_points if dp["status"] == "FAIL"),
        "warning_count": sum(1 for dp in data_points if dp["status"] == "WARNING")
    }
    
    # Extract metrics based on type
    if metric_type == "ip_check":
        packet_losses = [dp["metrics"].get("packet_loss_percent", 0) for dp in data_points if dp["metrics"]]
        rtts = [dp["metrics"].get("rtt_avg_ms", 0) for dp in data_points if dp["metrics"] and dp["metrics"].get("rtt_avg_ms", 0) > 0]
        
        if packet_losses:
            summary["avg_packet_loss_percent"] = round(sum(packet_losses) / len(packet_losses), 2)
        
        if rtts:
            summary["avg_rtt_ms"] = round(sum(rtts) / len(rtts), 2)
            summary["min_rtt_ms"] = round(min(rtts), 2)
            summary["max_rtt_ms"] = round(max(rtts), 2)
    
    elif metric_type == "port_check":
        success_rates = [dp["metrics"].get("connection_success_rate", 0) for dp in data_points if dp["metrics"]]
        latencies = [dp["metrics"].get("latency_avg_ms", 0) for dp in data_points if dp["metrics"] and dp["metrics"].get("latency_avg_ms", 0) > 0]
        
        if success_rates:
            summary["avg_success_rate"] = round(sum(success_rates) / len(success_rates), 2)
        
        if latencies:
            summary["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
            summary["min_latency_ms"] = round(min(latencies), 2)
            summary["max_latency_ms"] = round(max(latencies), 2)
    
    return summary


def _analyze_ip_health(logs: List) -> dict:
    """Analyze IP health from logs"""
    if not logs:
        return {"status": "NO_DATA", "message": "No IP check data available"}
    
    total = len(logs)
    passed = sum(1 for log in logs if log.status == "PASS")
    success_rate = (passed / total) * 100
    
    # Extract metrics
    packet_losses = []
    rtts = []
    
    for log in logs:
        if log.meta_data:
            pl = log.meta_data.get("packet_loss_percent")
            if pl is not None:
                packet_losses.append(pl)
            
            rtt = log.meta_data.get("rtt_avg_ms")
            if rtt is not None and rtt > 0:
                rtts.append(rtt)
    
    avg_packet_loss = sum(packet_losses) / len(packet_losses) if packet_losses else 0
    avg_rtt = sum(rtts) / len(rtts) if rtts else 0
    
    # Determine health status
    if success_rate >= 95 and avg_packet_loss < 5:
        status = "EXCELLENT"
    elif success_rate >= 90 and avg_packet_loss < 10:
        status = "GOOD"
    elif success_rate >= 80 and avg_packet_loss < 20:
        status = "FAIR"
    else:
        status = "POOR"
    
    return {
        "status": status,
        "success_rate": round(success_rate, 2),
        "total_checks": total,
        "passed_checks": passed,
        "avg_packet_loss_percent": round(avg_packet_loss, 2),
        "avg_rtt_ms": round(avg_rtt, 2)
    }


def _analyze_port_health(logs: List) -> dict:
    """Analyze port health from logs"""
    if not logs:
        return {"status": "NO_DATA", "message": "No port check data available"}
    
    total = len(logs)
    passed = sum(1 for log in logs if log.status == "PASS")
    success_rate = (passed / total) * 100
    
    # Extract metrics
    conn_success_rates = []
    latencies = []
    
    for log in logs:
        if log.meta_data:
            csr = log.meta_data.get("connection_success_rate")
            if csr is not None:
                conn_success_rates.append(csr)
            
            lat = log.meta_data.get("latency_avg_ms")
            if lat is not None and lat > 0:
                latencies.append(lat)
    
    avg_conn_success = sum(conn_success_rates) / len(conn_success_rates) if conn_success_rates else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Determine health status
    if success_rate >= 95 and avg_conn_success >= 95:
        status = "EXCELLENT"
    elif success_rate >= 90 and avg_conn_success >= 90:
        status = "GOOD"
    elif success_rate >= 80 and avg_conn_success >= 80:
        status = "FAIR"
    else:
        status = "POOR"
    
    return {
        "status": status,
        "success_rate": round(success_rate, 2),
        "total_checks": total,
        "passed_checks": passed,
        "avg_connection_success_rate": round(avg_conn_success, 2),
        "avg_latency_ms": round(avg_latency, 2)
    }


def _determine_overall_health(ip_health: dict, port_health: dict) -> dict:
    """Determine overall health from IP and port health"""
    ip_status = ip_health.get("status", "NO_DATA")
    port_status = port_health.get("status", "NO_DATA")
    
    # Status priority: POOR > FAIR > GOOD > EXCELLENT
    status_priority = {"POOR": 0, "FAIR": 1, "GOOD": 2, "EXCELLENT": 3, "NO_DATA": -1}
    
    ip_priority = status_priority.get(ip_status, -1)
    port_priority = status_priority.get(port_status, -1)
    
    # Overall status is the worse of the two
    overall_priority = min(ip_priority, port_priority)
    
    overall_status = "UNKNOWN"
    for status, priority in status_priority.items():
        if priority == overall_priority:
            overall_status = status
            break
    
    return {
        "status": overall_status,
        "network_layer": ip_status,
        "service_layer": port_status
    }


def _generate_recommendations(ip_health: dict, port_health: dict, overall_health: dict) -> List[str]:
    """Generate recommendations based on health analysis"""
    recommendations = []
    
    ip_status = ip_health.get("status", "NO_DATA")
    port_status = port_health.get("status", "NO_DATA")
    
    # IP recommendations
    if ip_status in ["POOR", "FAIR"]:
        packet_loss = ip_health.get("avg_packet_loss_percent", 0)
        if packet_loss > 10:
            recommendations.append(f"High packet loss detected ({packet_loss}%). Check network infrastructure and cabling.")
        
        rtt = ip_health.get("avg_rtt_ms", 0)
        if rtt > 100:
            recommendations.append(f"High latency detected ({rtt}ms). Consider network optimization or reducing distance to camera.")
    
    # Port recommendations
    if port_status in ["POOR", "FAIR"]:
        conn_success = port_health.get("avg_connection_success_rate", 0)
        if conn_success < 90:
            recommendations.append(f"Low connection success rate ({conn_success}%). Check camera service status and firewall rules.")
        
        latency = port_health.get("avg_latency_ms", 0)
        if latency > 200:
            recommendations.append(f"High service latency ({latency}ms). Camera may be overloaded or network constrained.")
    
    # General recommendations
    if overall_health.get("status") == "POOR":
        recommendations.append("Critical: Network connectivity is severely degraded. Immediate investigation required.")
    
    if not recommendations:
        recommendations.append("Network health is good. No immediate action required.")
    
    return recommendations