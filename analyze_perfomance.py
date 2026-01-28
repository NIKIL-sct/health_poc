"""
Performance Analysis Script (Redis Architecture Compatible)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import statistics


class PerformanceAnalyzer:
    def __init__(self, report_path):
        self.report_path = Path(report_path)
        with open(self.report_path) as f:
            self.data = json.load(f)

        self.system = self.data.get("system_metrics", [])
        self.redis = self.data.get("redis_metrics", [])
        self.health = self.data.get("api_health", [])
        self.info = self.data.get("test_info", {})

    def analyze(self):
        print("=" * 80)
        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)

        self._test_info()
        self._system_performance()
        self._queue_analysis()
        self._stability()
        self._recommendations()

        print("=" * 80)

    # --------------------------------------------------

    def _test_info(self):
        duration = self.info.get("duration_sec", 0)
        print("\nTest Information:")
        print(f"  Target Cameras: {self.info.get('target_cameras')}")
        print(f"  Duration: {duration:.1f}s ({duration / 60:.1f} min)")
        print(f"  Start: {self.info.get('start_time')}")
        print(f"  End: {self.info.get('end_time')}")

    # --------------------------------------------------

    def _system_performance(self):
        print("\nSystem Performance:")

        cpu = [m["cpu_percent"] for m in self.system]
        mem = [m["memory_mb"] for m in self.system]
        threads = [m["threads"] for m in self.system]

        if not cpu:
            print("  No system metrics collected")
            return

        print(f"  CPU:")
        print(f"    Avg: {statistics.mean(cpu):.1f}%")
        print(f"    Max: {max(cpu):.1f}%")

        print(f"  Memory:")
        print(f"    Avg: {statistics.mean(mem):.2f} MB")
        print(f"    Peak: {max(mem):.2f} MB")
        print(f"    Growth: {mem[-1] - mem[0]:.2f} MB")

        print(f"  Threads:")
        print(f"    Min: {min(threads)}")
        print(f"    Max: {max(threads)}")

        per_cam = max(mem) / max(self.info.get("target_cameras", 1), 1)
        print(f"  Memory / Camera: {per_cam:.4f} MB")

    # --------------------------------------------------

    def _queue_analysis(self):
        print("\nRedis Queue Analysis:")

        net_q = [m["network_queue"] for m in self.redis]
        vis_q = [m["vision_queue"] for m in self.redis]

        if not net_q:
            print("  No Redis metrics collected")
            return

        print(f"  Network Queue:")
        print(f"    Avg: {statistics.mean(net_q):.0f}")
        print(f"    Peak: {max(net_q)}")

        print(f"  Vision Queue:")
        print(f"    Avg: {statistics.mean(vis_q):.0f}")
        print(f"    Peak: {max(vis_q)}")

        if max(net_q) > 1000:
            print("  ⚠ Network workers saturated")

        if max(vis_q) > 300:
            print("  ⚠ Vision workers saturated")

    # --------------------------------------------------

    def _stability(self):
        print("\nStability Analysis:")

        cpu = [m["cpu_percent"] for m in self.system]
        mem = [m["memory_mb"] for m in self.system]

        cpu_std = statistics.stdev(cpu) if len(cpu) > 1 else 0
        mem_growth = mem[-1] - mem[0]

        print(f"  CPU Std Dev: {cpu_std:.2f}% {'Stable' if cpu_std < 15 else 'Spiky'}")
        print(f"  Memory Growth: {mem_growth:.2f} MB {'OK' if mem_growth < 100 else '⚠ Possible Leak'}")

    # --------------------------------------------------

    def _recommendations(self):
        print("\nRecommendations:")

        net_peak = max(m["network_queue"] for m in self.redis)
        vis_peak = max(m["vision_queue"] for m in self.redis)

        recs = []

        if net_peak > 1000:
            recs.append("Increase network_worker_pool concurrency")
            recs.append("Increase IP/PORT check intervals")

        if vis_peak > 300:
            recs.append("Reduce vision frequency or add more vision workers")
            recs.append("Lower vision resolution / frame sampling")

        if max(m["cpu_percent"] for m in self.system) > 90:
            recs.append("CPU saturated – distribute workers across machines")

        if not recs:
            print("  ✓ System is stable under load")
        else:
            for i, r in enumerate(recs, 1):
                print(f"  {i}. {r}")


# --------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <report.json>")
        sys.exit(1)

    analyzer = PerformanceAnalyzer(sys.argv[1])
    analyzer.analyze()


if __name__ == "__main__":
    main()
