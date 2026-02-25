"""
shared/detection_common.py — Common detection state, chain logic, and utilities

Shared between BILT and YOLO backend services to eliminate code duplication.
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Allowed settings keys (security: whitelist)
# ══════════════════════════════════════════════

_ALLOWED_DETECTION_SETTINGS_KEYS = frozenset({
    "conf", "iou", "max_det", "classes", "counter_mode",
    "save_images", "dataset_capture", "project_folder",
    "chain_mode", "chain_steps", "chain_timeout",
    "chain_pause_time", "chain_auto_advance", "task",
})


def sanitize_detection_settings(update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a settings update dict to only allowed keys.

    Prevents clients from injecting arbitrary state into the
    detection settings dictionary.
    """
    return {k: v for k, v in update.items() if k in _ALLOWED_DETECTION_SETTINGS_KEYS}


# ══════════════════════════════════════════════
# Default state factories
# ══════════════════════════════════════════════

def make_default_detection_settings(
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a fresh copy of default detection settings."""
    settings: Dict[str, Any] = {
        "conf": 0.60, "iou": 0.10, "max_det": 10,
        "classes": None, "counter_mode": False,
        "save_images": False, "dataset_capture": False,
        "project_folder": "",
        "chain_mode": False, "chain_steps": [],
        "chain_timeout": 50.0, "chain_pause_time": 10.0,
        "chain_auto_advance": True,
    }
    if extra:
        settings.update(extra)
    return settings


def make_default_chain_state() -> Dict[str, Any]:
    """Create a fresh copy of default chain state."""
    return {
        "active": False, "current_step": 0,
        "step_start_time": None, "completed_cycles": 0,
        "failed_steps": 0, "step_history": [],
        "current_detections": {}, "last_step_result": None,
        "cycle_pause": False, "cycle_pause_start": None,
        "waiting_for_ack": False, "error_message": None,
        "wrong_object": None, "error_step": None,
    }


def make_default_detection_stats() -> Dict[str, Any]:
    """Create a fresh copy of default detection stats."""
    return {
        "total_detections": 0, "fps": 0.0,
        "last_detection_time": None,
        "detections_per_frame": 0,
        "current_classes": [],
    }


# ══════════════════════════════════════════════
# Chain detection logic
# ══════════════════════════════════════════════

def process_chain(
    dets: List[Dict[str, Any]],
    detection_settings: Dict[str, Any],
    chain_state: Dict[str, Any],
) -> None:
    """
    Advance the chain detection state machine.

    Shared between BILT and YOLO services.
    Thread-safety is the caller's responsibility.
    """
    steps = detection_settings["chain_steps"]
    if not steps or not chain_state["active"]:
        return

    now = time.time()

    # Cycle pause
    if chain_state["cycle_pause"]:
        elapsed = now - chain_state["cycle_pause_start"]
        if elapsed >= detection_settings["chain_pause_time"]:
            chain_state.update({
                "cycle_pause": False, "cycle_pause_start": None,
                "current_step": 0, "step_start_time": now,
            })
        return

    # Waiting for error acknowledgement
    if chain_state["waiting_for_ack"]:
        return

    step_idx = chain_state["current_step"]
    if step_idx >= len(steps):
        chain_state.update({
            "completed_cycles": chain_state["completed_cycles"] + 1,
            "cycle_pause": True, "cycle_pause_start": now,
        })
        return

    step = steps[step_idx]

    # Count detected classes
    present: Dict[str, int] = {}
    for d in dets:
        name = d["class_name"]
        present[name] = present.get(name, 0) + 1

    required: Dict[str, int] = step.get("classes", {})

    # Check for unexpected objects
    for cls in present:
        if cls not in required:
            chain_state.update({
                "waiting_for_ack": True,
                "error_step":      step_idx,
                "error_message":   f"ステップ '{step['name']}' で不正なオブジェクト '{cls}' を検出",
                "wrong_object":    cls,
                "failed_steps":    chain_state["failed_steps"] + 1,
            })
            return

    # Check if all required objects are present
    if all(present.get(cls, 0) >= cnt for cls, cnt in required.items()):
        chain_state["current_step"]     = step_idx + 1
        chain_state["step_start_time"]  = now
        chain_state["last_step_result"] = "success"


def start_chain(
    detection_settings: Dict[str, Any],
    chain_state: Dict[str, Any],
) -> None:
    """Initialize chain detection mode."""
    detection_settings["chain_mode"] = True
    chain_state.update({
        "active": True, "current_step": 0,
        "step_start_time": time.time(), "completed_cycles": 0,
        "failed_steps": 0, "step_history": [],
        "current_detections": {}, "last_step_result": None,
        "cycle_pause": False, "cycle_pause_start": None,
        "waiting_for_ack": False, "error_message": None,
        "wrong_object": None, "error_step": None,
    })


def acknowledge_chain_error(chain_state: Dict[str, Any]) -> bool:
    """
    Acknowledge a chain error and resume from the failed step.

    Returns True if there was an error to acknowledge.
    """
    if not chain_state.get("waiting_for_ack"):
        return False
    chain_state.update({
        "waiting_for_ack": False, "error_message": None,
        "wrong_object": None, "last_step_result": None,
        "step_start_time": time.time(),
        "current_step": chain_state.get("error_step", chain_state["current_step"]),
        "error_step": None,
    })
    return True


# ══════════════════════════════════════════════
# Project backup
# ══════════════════════════════════════════════

def backup_project(project_path: str) -> None:
    """Backup project images and labels with a timestamp suffix."""
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk_dir = Path(project_path) / f"backup_{ts}"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            src = Path(project_path) / split / sub
            if src.exists():
                shutil.copytree(str(src), str(bk_dir / split / sub))
    logger.info("バックアップ作成: %s", bk_dir)
