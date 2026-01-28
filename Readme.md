

```md
# HCS_VigilX


##  Project Structure

```

HCS_VigilX_1/
│
├── app/
│   ├── **init**.py
│   └── app.py                  # FastAPI entrypoint
│
├── core/
│   ├── **init**.py
│   ├── schedular.py             # Camera scheduling logic
│   ├── ping_checker.py          # Network health checker
│   └── vision_checker.py        # Vision analysis logic
│
├── workers/
│   ├── **init**.py
│   ├── network_worker.py        # Network worker pool
│   └── vision_worker.py         # Vision worker pool
│
├── storage/
│   ├── **init**.py
│   ├── redis_client.py          # Redis abstraction
│   ├── vision_storage.py        # Vision result persistence
│   └── camera_worker.py         # Per-camera logging utilities
│
├── img/
│   ├── baseline/                # Baseline reference frames
│   └── captures/                # Live captured frames
│
├── logs/                        # Runtime logs & JSON outputs
│
├── analyze_performance.py       # Load test performance analysis
├── load_test.py                 # Load testing script
├── demo.py                      # Demo / manual testing
├── architecture_code.docx       # Architecture documentation
├── requirements.txt
├── Readme.md
└── venv/

````

---

##  Architecture Overview

The system follows a **producer–consumer architecture**:

1. **Scheduler**
   - Periodically schedules health checks for cameras
   - Pushes tasks into Redis queues

2. **Network Workers**
   - Consume network health tasks
   - Validate IP/Port/RTSP availability
   - Update Redis summaries

3. **Vision Workers**
   - Consume vision health tasks
   - Perform image-based analysis
   - Store results and logs

4. **FastAPI Application**
   - Exposes APIs for manual health checks
   - Provides health summaries and responses

---

##  Running the Application

### 1️ Activate virtual environment

```bash
source venv/bin/activate
````

### 2️ Start Redis

Ensure Redis is running locally:

```bash
redis-server
```

---

### 3️ Start the FastAPI server

From the project root:

```bash
uvicorn app.app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## ⚙️ Background Workers

Workers are typically started as **separate processes**.

### Network workers

```bash
python workers/network_worker.py
```

### Vision workers

```bash
python workers/vision_worker.py
```

---

##  Load Testing

Run a load test simulating hundreds of cameras:

```bash
python load_test.py
```

---

##  Performance Analysis

Analyze load-test results:

```bash
python analyze_performance.py <path_to_log_file.json>
```

This provides:

* CPU usage
* Memory usage
* Queue depth
* Per-camera resource estimates
* Stability indicators

---

## 🧾 Logging Behavior

* Supports **per-camera JSON logging**
* Logging can be toggled via configuration flags
* Default flow:

  ```
  Worker → Redis → Response
  ```
* Optional:

  ```
  Worker → JSON logs (per camera)
  ```

---

