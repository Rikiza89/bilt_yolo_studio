# CLAUDE.md — Developer Guide for Claude Code

## Project Overview

BILT+YOLO Studio is a multi-process computer vision application. Understanding the **process boundary** is the most important architectural fact: AGPL-licensed services run as subprocesses and communicate with the LGPL UI exclusively over HTTP.

## Running the Application

```bash
python launcher.py
```

There is no separate build step. All three services start from the single launcher.

## Key Architecture Rules

### License boundary — do not cross it
- `services/bilt_service/` and `services/yolo_service/` are **AGPL**. Keep all new detection/training logic inside these directories.
- `ui/` is **LGPL** (PySide6 + Flask). It must never import from `bilt/` or `services/` directly.
- `shared/` is the only code shared across the boundary — it must remain dependency-light (stdlib only, no PySide6, no torch, no ultralytics).

### HTTP is the only IPC
All communication between the UI and a service must go through the HTTP API using `shared/service_client.py`. Do not add direct function calls or shared memory.

### Port assignments (see `shared/contracts.py`)
| Service | Default port | Env var |
|---------|-------------|---------|
| UI / Flask | 5100 | `STUDIO_UI_PORT` |
| BILT service | 5101 | `BILT_SERVICE_PORT` |
| YOLO service | 5102 | `YOLO_SERVICE_PORT` |

## Module Map

| Path | Purpose |
|------|---------|
| `launcher.py` | Starts all subprocesses; manage lifecycle here |
| `bilt/` | Pure BILT engine (PyTorch SSDLite320 MobileNetV3) |
| `services/bilt_service/app.py` | Flask wrapper around `bilt/` |
| `services/yolo_service/app.py` | Flask wrapper around Ultralytics YOLO |
| `shared/contracts.py` | Canonical dataclasses: `Detection`, `TrainingConfig`, `DetectionSettings`, enums |
| `shared/service_client.py` | HTTP client used by the UI to call services |
| `ui/web/` | Flask server + HTML/JS frontend |
| `ui/pyside/` | PySide6 `QMainWindow` that embeds the web UI |

## Shared Contracts

When adding a new field to an inter-process payload, update `shared/contracts.py` and both the service endpoint and the UI client in the same commit. The `to_dict` / `from_dict` methods are the serialization contract — keep them in sync.

## Common Tasks

### Add a new detection parameter
1. Add the field to `DetectionSettings` in `shared/contracts.py` (include it in `to_dict`).
2. Read it in `services/bilt_service/app.py` and/or `services/yolo_service/app.py`.
3. Update `ui/web/` or `ui/pyside/` to send the new parameter.

### Add a new service endpoint
1. Define the route in `services/<engine>_service/app.py`.
2. Add a corresponding method to `shared/service_client.py`.
3. Call it from the UI — never import the service module directly.

### Add a new BILT model architecture
1. Add the model class to `bilt/model.py`.
2. Update `bilt/core.py` (`DetectionModel`) to support the new architecture string.
3. Update `bilt/trainer.py` and `bilt/inferencer.py` as needed.

## Style Notes

- Python 3.10+. Use `from __future__ import annotations` at the top of every file.
- Use `shared/utils.py`'s `get_logger(__name__)` for all logging — do not use `print`.
- Dataclasses over Pydantic in `shared/` (keep the dependency footprint minimal for service processes).
- No type: ignore comments without an explanation.

## Testing

There is currently no automated test suite. When adding tests, place them in a top-level `tests/` directory and mirror the source tree structure (e.g., `tests/bilt/test_trainer.py`).

## License

All files in this repository are licensed under **AGPL-3.0** unless otherwise noted in the file header. PySide6 components are dynamically linked and covered by the LGPL; see `launcher.py` for the rationale.
