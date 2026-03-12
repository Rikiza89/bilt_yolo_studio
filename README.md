# BILT+YOLO Studio

A desktop/web computer vision studio for training and running real-time object detection using two interchangeable backends: **BILT** (a custom SSD-MobileNetV3 engine) and **YOLO** (via Ultralytics).

## Architecture

The application runs as three separate processes managed by a single launcher:

```
launcher.py
├── services/bilt_service/   — BILT detection backend  (AGPL-3.0, subprocess)
├── services/yolo_service/   — YOLO detection backend  (AGPL-3.0, subprocess)
├── ui/web/                  — Flask web UI             (LGPL, background thread)
└── ui/pyside/               — PySide6 desktop shell   (LGPL, main process)
```

The AGPL services are launched as subprocesses and communicate with the UI exclusively over HTTP. This isolates the AGPL license boundary and prevents static-linking obligations from propagating to the LGPL PySide6 UI layer.

## Features

- **Dual engine support** — switch between BILT (PyTorch SSD) and YOLO at runtime
- **Real-time detection** — live camera feed with configurable confidence, IoU, and class filters
- **Training pipeline** — dataset management, training configuration, and evaluation in one UI
- **Chain mode** — multi-step detection sequences with configurable timeouts and auto-advance
- **Dataset capture** — save frames directly from the live feed for labelling
- **Counter mode** — object counting overlay
- **PySide6 + web dual UI** — native desktop window backed by an embedded Flask server

## Supported Task Types

| Task | BILT | YOLO |
|------|------|------|
| Detection (`detect`) | Yes | Yes |
| Segmentation (`segment`) | No | Yes |
| OBB (`obb`) | No | Yes |
| Pose (`pose`) | No | Yes |

## Requirements

```
PySide6
Flask
Werkzeug
requests
ultralytics
opencv-python
numpy
Pillow
torch          # required by the BILT service
torchvision    # required by the BILT service
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the full application (launcher manages all services)
python launcher.py

# Environment variable overrides (optional)
STUDIO_UI_PORT=5100    # Flask web UI port (default: 5100)
BILT_SERVICE_PORT=5101 # BILT service port (default: 5101)
YOLO_SERVICE_PORT=5102 # YOLO service port (default: 5102)
```

## Directory Layout

```
bilt_yolo_studio/
├── launcher.py           # Application entry point
├── requirements.txt
├── bilt/                 # BILT engine library
│   ├── core.py           # DetectionModel (SSDLite320 MobileNetV3)
│   ├── trainer.py
│   ├── inferencer.py
│   ├── evaluator.py
│   ├── dataset.py
│   ├── model.py
│   ├── config.py
│   └── utils.py
├── services/
│   ├── bilt_service/     # Flask app wrapping the BILT engine
│   └── yolo_service/     # Flask app wrapping Ultralytics YOLO
├── shared/
│   ├── contracts.py      # Shared dataclasses & enums (Detection, TrainingConfig, …)
│   ├── camera_utils.py
│   ├── detection_common.py
│   └── service_client.py
└── ui/
    ├── pyside/           # PySide6 main window
    └── web/              # Flask web server & templates
```

## License

BILT+YOLO Studio is released under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE) for details.
