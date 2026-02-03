# Camera Health Check Microservice

## Purpose

This microservice is responsible for continuously monitoring the health of IP cameras. It exposes APIs to manage camera configurations, performs periodic health checks in the background, stores health results, raises alerts, and manages automatic cleanup of historical health logs based on retention policies.

The service is designed to run continuously after application startup and does not require manual triggering of health checks.

---

## High-Level Workflow

1. Application starts using `uvicorn app.app:app --reload`
2. Background schedulers and workers are initialized on startup
3. Cameras are registered and cached in Redis
4. Cameras are added to Redis-based schedulers with configured intervals
5. Scheduler continuously evaluates which cameras are due for health checks
6. Health check tasks are pushed to Redis queues
7. Workers consume tasks and execute health checks
8. Health results are stored in the database
9. Alerts are generated for unhealthy conditions
10. Old logs are cleaned automatically based on retention settings

---

## Health Checks Performed

* IP Reachability Check
* Port Availability Check
* Vision Checks

  * Blur detection
  * Obstruction detection
  * Position / displacement checks

Each check runs at its own configured interval per camera.

---

## API Endpoints

### Camera Management

* `POST /health/camera`

  * Register a new camera
  * Stores configuration in DB and Redis
  * Registers camera into the scheduler

* `GET /health/camera/{camera_id}`

  * Fetch camera configuration

* `GET /health/camera`

  * List all registered cameras

---

### Health Data APIs

* `GET /health/camera/{camera_id}/logs`

  * Retrieve health check logs for a camera

* `GET /health/camera/{camera_id}/alerts`

  * Retrieve alerts generated for a camera

* `GET /health/summary`

  * Fetch aggregated health summary

---

### Log Retention APIs

* `GET /health/camera/{camera_id}/log-retention`

  * Retrieve log retention configuration for a camera

* `POST /health/camera/{camera_id}/log-retention`

  * Configure health log retention interval

---

## Background Components

### Scheduler

* Redis ZSET-based time-wheel scheduler
* Maintains separate schedules for:

  * IP checks
  * Port checks
  * Vision checks
* Continuously runs after app startup

### Workers

* Network Workers

  * Async workers for IP and port checks

* Vision Workers

  * Multiprocessing workers for CPU-intensive vision analysis

* Log Cleanup Worker

  * Deletes old health logs and alerts based on retention policy

---

## Database Schema

### cameras

Stores camera configuration and scheduling intervals.

Key fields:

* id
* ip_address
* port
* rtsp_url
* interval_ip
* interval_port
* interval_vision

---

### health_logs

Stores results of all health checks.

Key fields:

* id
* camera_id
* check_type
* status
* metadata
* created_at

---

### alerts

Stores alerts generated when health checks fail or cross thresholds.

Key fields:

* id
* camera_id
* alert_type
* severity
* created_at

---

### camera_latency

Stores network latency and connectivity metrics.

Key fields:

* id
* camera_id
* latency
* created_at

---

## Redis Usage

* Camera runtime cache (`camera:{id}`)
* Scheduler ZSETs for time-based execution
* Task queues for worker consumption
* Aggregated health summaries

Redis acts as the runtime backbone, while PostgreSQL is the source of truth.

---

## Startup Behavior

On application startup:

* Database connections are initialized
* Redis connections are established
* Scheduler loop starts automatically
* Worker pools are started
* Log retention and cleanup schedulers are activated

No manual triggering is required for health checks once cameras are registered.

---

## Notes

* Health checks will not run unless cameras are registered into the scheduler
* Redis data is considered ephemeral and can be rebuilt from the database
* Log retention policies control automatic deletion of historical records

---

## Scope

This README documents the functionality, APIs, workflow, and data model of the microservice. It does not cover deployment, security hardening, or production infrastructure details.
