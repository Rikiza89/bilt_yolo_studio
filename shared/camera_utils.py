"""
shared/camera_utils.py — Thread-safe camera management

Shared between BILT and YOLO backend services.
Not imported by the UI process.

Eliminates duplication of camera code between service processes.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Thread-safe camera device manager.

    Encapsulates camera initialization, frame grabbing, and
    resolution management behind a single lock.
    """

    def __init__(self) -> None:
        self._camera: Optional[cv2.VideoCapture] = None
        self._camera_index: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def camera_index(self) -> Optional[int]:
        return self._camera_index

    # ── Device scanning ──────────────────────────

    @staticmethod
    def get_available_cameras(max_index: int = 4) -> List[Dict[str, Any]]:
        """Scan for available camera devices."""
        cameras: List[Dict[str, Any]] = []
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cameras.append({
                    "index":  idx,
                    "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps":    cap.get(cv2.CAP_PROP_FPS),
                    "name":   f"Camera {idx}",
                })
                cap.release()
        return cameras

    # ── Initialization ───────────────────────────

    def init_camera(self, index: int) -> bool:
        """
        Initialize camera at *index*.

        Releases any previously opened camera first.
        Tries the platform-preferred backend, then the default.
        """
        with self._lock:
            if self._camera is not None:
                self._camera.release()
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                logger.error("Cannot open camera %d", index)
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self._camera       = cap
            self._camera_index = index
            logger.info("Camera %d initialized", index)
            return True

    # ── Frame capture ────────────────────────────

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grab the latest frame. Returns ``None`` if camera is not ready."""
        with self._lock:
            if self._camera is None or not self._camera.isOpened():
                return None
            ret, frame = self._camera.read()
            return frame if ret else None

    def grab_frame_with_retry(self) -> Optional[np.ndarray]:
        """
        Grab a frame, reinitializing the camera on failure.

        Useful for endpoints that need a one-shot capture
        (e.g. preview, snapshot) where the camera may have been
        released after a device scan on Windows CAP_DSHOW.
        """
        frame = self.grab_frame()
        if frame is None and self._camera_index is not None:
            logger.warning(
                "Frame grab failed — reinitializing camera %d",
                self._camera_index,
            )
            if self.init_camera(self._camera_index):
                time.sleep(0.15)
                frame = self.grab_frame()
        return frame

    # ── Resolution / info ────────────────────────

    def set_resolution(self, width: int, height: int) -> None:
        """Set camera resolution."""
        with self._lock:
            if self._camera and self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Return current camera info dict, or ``None`` if not open."""
        with self._lock:
            if self._camera and self._camera.isOpened():
                return {
                    "index":  self._camera_index,
                    "width":  int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps":    self._camera.get(cv2.CAP_PROP_FPS),
                }
        return None

    # ── Cleanup ──────────────────────────────────

    def release(self) -> None:
        """Release the camera device."""
        with self._lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
