"""
services/yolo_service/app.py — YOLOバックエンドサービス (Ultralytics AGPL-3.0)

このプロセスはUIとは完全に分離された独立プロセスとして動作する。
BILTサービスと同一のHTTP API契約に準拠するため、
UIクライアントはエンジン種別のみを切り替えて同じコードで
両エンジンを使用できる。

サポートするタスク:
  - detect   : バウンディングボックス検出
  - segment  : セグメンテーション (ポリゴンマスク)
  - obb      : 方向付きバウンディングボックス
  - pose     : 姿勢推定 (キーポイント)

モデルフォーマット:
  - .pt  (PyTorch / Ultralytics ネイティブ)
  - .onnx, .engine 等は Ultralytics が自動処理
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────
# パス設定
# ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ultralytics のインポート (AGPL-3.0)
# このインポートはこのプロセス内にのみ存在する
from ultralytics import YOLO  # noqa: E402 (AGPL)

from shared.contracts import AppDirectories, Ports, TaskType  # noqa: E402

# ──────────────────────────────────────────────
# ログ設定
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [YOLO-SVC] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_ROOT / "yolo_service.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Ultralytics の過剰なログを抑制する
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ──────────────────────────────────────────────
# アプリケーション初期化
# ──────────────────────────────────────────────
app  = Flask(__name__)
dirs = AppDirectories(base=str(_ROOT))
dirs.ensure_all()

# ──────────────────────────────────────────────
# グローバル状態
# ──────────────────────────────────────────────
current_model: Optional[YOLO] = None
model_info: Dict[str, Any]    = {
    "name": None, "classes": [], "loaded": False, "task": "detect"
}

_camera: Optional[cv2.VideoCapture] = None
_camera_index: Optional[int]        = None
_camera_lock = threading.Lock()

detection_active  = False
detection_thread: Optional[threading.Thread] = None
latest_frame:  Optional[np.ndarray] = None
frame_lock     = threading.Lock()
counter_triggered: Dict[str, bool] = {}

training_active  = False
autotrain_active = False

detection_settings: Dict[str, Any] = {
    "conf": 0.60, "iou": 0.10, "max_det": 10,
    "classes": None, "counter_mode": False,
    "save_images": False, "dataset_capture": False,
    "project_folder": "",
    "chain_mode": False, "chain_steps": [],
    "chain_timeout": 50.0, "chain_pause_time": 10.0,
    "chain_auto_advance": True,
    "task": "detect",  # YOLOサービス固有: タスク種別
}

object_counters: Dict[str, int] = {}
detection_stats: Dict[str, Any] = {
    "total_detections": 0, "fps": 0.0, "last_detection_time": None
}

chain_state: Dict[str, Any] = {
    "active": False, "current_step": 0,
    "step_start_time": None, "completed_cycles": 0,
    "failed_steps": 0, "step_history": [],
    "current_detections": {}, "last_step_result": None,
    "cycle_pause": False, "cycle_pause_start": None,
    "waiting_for_ack": False, "error_message": None,
    "wrong_object": None, "error_step": None,
}


# ══════════════════════════════════════════════
# カメラユーティリティ (BILTサービスと同一実装)
# ══════════════════════════════════════════════

def _get_available_cameras() -> List[Dict[str, Any]]:
    cameras = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
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


def _init_camera(index: int) -> bool:
    global _camera, _camera_index
    with _camera_lock:
        if _camera:
            _camera.release()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        _camera       = cap
        _camera_index = index
        logger.info("カメラ %d を初期化しました", index)
        return True


def _grab_frame() -> Optional[np.ndarray]:
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            return None
        ret, frame = _camera.read()
        return frame if ret else None


# ══════════════════════════════════════════════
# YOLO推論ユーティリティ
# ══════════════════════════════════════════════

def _yolo_predict(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    フレームに対してYOLO推論を実行し、統一フォーマットのリストを返す。
    タスク種別 (detect/segment/obb/pose) に応じて結果を変換する。

    Returns:
        検出結果のリスト。各要素は以下のキーを持つ:
          class_id, class_name, score, bbox,
          points (segment/obb), keypoints (pose)
    """
    if current_model is None:
        return []

    results_list = current_model.predict(
        frame,
        conf    = detection_settings["conf"],
        iou     = detection_settings["iou"],
        max_det = detection_settings["max_det"],
        classes = detection_settings["classes"],
        verbose = False,
    )

    detections: List[Dict[str, Any]] = []
    if not results_list:
        return detections

    results  = results_list[0]
    names    = current_model.names
    task     = detection_settings.get("task", "detect")

    # ── バウンディングボックス (全タスク共通) ──
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cid   = int(box.cls[0])
            score = float(box.conf[0])
            det: Dict[str, Any] = {
                "class_id":   cid,
                "class_name": names.get(cid, f"class_{cid}"),
                "score":      score,
                "bbox":       [x1, y1, x2, y2],
                "task_type":  task,
            }
            detections.append(det)

    # ── セグメンテーション: ポリゴンマスクを付加 ──
    if task == "segment" and results.masks is not None:
        for i, mask in enumerate(results.masks.xy):
            if i < len(detections):
                # mask は (N, 2) の numpy 配列
                detections[i]["points"] = [
                    {"x": float(p[0]), "y": float(p[1])} for p in mask
                ]

    # ── OBB: 方向付きボックスのコーナー座標を付加 ──
    if task == "obb" and results.obb is not None:
        for i, obb in enumerate(results.obb.xyxyxyxy):
            if i < len(detections):
                pts = obb.cpu().numpy().reshape(-1, 2)
                detections[i]["points"] = [
                    {"x": float(p[0]), "y": float(p[1])} for p in pts
                ]

    # ── ポーズ推定: キーポイントを付加 ──
    if task == "pose" and results.keypoints is not None:
        for i, kpts in enumerate(results.keypoints.xy):
            if i < len(detections):
                detections[i]["keypoints"] = [
                    {"x": float(p[0]), "y": float(p[1])} for p in kpts
                ]

    return detections


def _draw_detections(frame: np.ndarray, dets: List[Dict]) -> np.ndarray:
    """検出結果をフレームに描画して返す。タスクに応じて描画内容を変える。"""
    colors = [
        (0, 200, 255), (0, 255, 128), (255, 80, 0),  (255, 255, 0),
        (200, 0, 255), (0, 128, 255), (255, 0, 128), (128, 255, 0),
    ]
    task = detection_settings.get("task", "detect")
    h, w = frame.shape[:2]

    for det in dets:
        cid   = det.get("class_id", 0)
        color = colors[cid % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # セグメント/OBBはポリゴン描画
        if task in ("segment", "obb") and det.get("points"):
            pts = np.array(
                [[int(p["x"]), int(p["y"])] for p in det["points"]], dtype=np.int32
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ポーズ: キーポイント描画
        if task == "pose" and det.get("keypoints"):
            for kp in det["keypoints"]:
                cx, cy = int(kp["x"]), int(kp["y"])
                if cx > 0 and cy > 0:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"{det.get('class_name','')} {det.get('score', 0):.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


# ══════════════════════════════════════════════
# チェーン検出 (BILTサービスと共通ロジック)
# ══════════════════════════════════════════════

def _process_chain(dets: List[Dict]) -> None:
    steps = detection_settings["chain_steps"]
    if not steps or not chain_state["active"]:
        return

    now = time.time()

    if chain_state["cycle_pause"]:
        if now - chain_state["cycle_pause_start"] >= detection_settings["chain_pause_time"]:
            chain_state.update({
                "cycle_pause": False, "cycle_pause_start": None,
                "current_step": 0, "step_start_time": now,
            })
        return

    if chain_state["waiting_for_ack"]:
        return

    step_idx = chain_state["current_step"]
    if step_idx >= len(steps):
        chain_state.update({
            "completed_cycles": chain_state["completed_cycles"] + 1,
            "cycle_pause": True, "cycle_pause_start": now,
        })
        return

    step    = steps[step_idx]
    present: Dict[str, int] = {}
    for d in dets:
        present[d["class_name"]] = present.get(d["class_name"], 0) + 1

    required: Dict[str, int] = step.get("classes", {})

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

    if all(present.get(cls, 0) >= cnt for cls, cnt in required.items()):
        chain_state["current_step"]     = step_idx + 1
        chain_state["step_start_time"]  = now
        chain_state["last_step_result"] = "success"


# ══════════════════════════════════════════════
# 検出ループ
# ══════════════════════════════════════════════

def _detection_loop() -> None:
    """カメラフレームを連続取得し、YOLOで推論するメインループ。"""
    global latest_frame, object_counters, counter_triggered

    fps_counter  = 0
    fps_time     = time.time()
    frame_period = 1.0 / 30

    while detection_active:
        t0 = time.time()
        try:
            frame = _grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = frame.copy()

            if current_model and model_info["loaded"]:
                yolo_dets = _yolo_predict(frame)

                if detection_settings["counter_mode"] and not detection_settings["chain_mode"]:
                    for det in yolo_dets:
                        name = det["class_name"]
                        if name not in counter_triggered:
                            object_counters[name] = object_counters.get(name, 0) + 1
                            counter_triggered[name] = True

                if detection_settings["chain_mode"]:
                    _process_chain(yolo_dets)

                annotated = _draw_detections(frame.copy(), yolo_dets)
                detection_stats["total_detections"] += len(yolo_dets)
                detection_stats["last_detection_time"] = datetime.now().isoformat()

            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                detection_stats["fps"] = round(fps_counter / (time.time() - fps_time), 1)
                fps_counter = 0
                fps_time    = time.time()

            with frame_lock:
                latest_frame = annotated

        except Exception:
            logger.error("検出ループエラー:\n%s", traceback.format_exc())

        elapsed = time.time() - t0
        time.sleep(max(0, frame_period - elapsed))


# ══════════════════════════════════════════════
# トレーニングワーカー
# ══════════════════════════════════════════════

def _training_worker(config: Dict[str, Any]) -> None:
    """バックグラウンドスレッドでYOLOトレーニングを実行する。"""
    global training_active
    try:
        training_active = True
        model_name = config.get("model_name") or config.get("model", "yolov8n.pt")
        task       = config.get("task_type", "detect")

        # scratch の場合は Ultralytics のプリセットを使用
        if model_name == "scratch":
            model_name = {
                "detect":  "yolov8n.pt",
                "segment": "yolov8n-seg.pt",
                "obb":     "yolov8n-obb.pt",
                "pose":    "yolov8n-pose.pt",
            }.get(task, "yolov8n.pt")

        model_path = Path(dirs.models) / model_name
        yolo = YOLO(str(model_path) if model_path.exists() else model_name)

        save_dir = Path(config.get("project_path", str(_ROOT / "runs")))
        run_name = f"yolo_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results = yolo.train(
            data       = config["data_yaml"],
            epochs     = int(config.get("epochs", 100)),
            batch      = int(config.get("batch_size", config.get("batch", 8))),
            imgsz      = int(config.get("img_size", config.get("imgsz", 640))),
            lr0        = float(config.get("learning_rate", config.get("lr0", 0.001))),
            device     = config.get("device", "cpu"),
            project    = str(save_dir),
            name       = run_name,
            task       = task,
            verbose    = True,
        )

        # カスタム名で保存
        custom = config.get("custom_save_name")
        if custom:
            if not custom.endswith(".pt"):
                custom += ".pt"
            best = save_dir / run_name / "weights" / "best.pt"
            if best.exists():
                shutil.copy2(str(best), str(Path(dirs.models) / custom))
                logger.info("モデルを %s として保存しました", custom)

        logger.info("YOLOトレーニング完了")
    except Exception:
        logger.error("YOLOトレーニングエラー:\n%s", traceback.format_exc())
    finally:
        training_active = False


def _autotrain_worker(config: Dict[str, Any]) -> None:
    global autotrain_active
    try:
        autotrain_active = True
        model_path = config["model_path"]
        task       = config.get("task_type", "detect")

        yolo = YOLO(model_path)

        if config.get("backup_enabled", True):
            _backup_project(config["project_path"])

        # YOLO でオートラベリング
        _auto_label_yolo(yolo, config, task)

        # 再トレーニング
        yolo.train(
            data    = config["data_yaml"],
            epochs  = int(config.get("epochs", 50)),
            batch   = int(config.get("batch", 8)),
            imgsz   = int(config.get("imgsz", 640)),
            lr0     = float(config.get("lr0", 0.001)),
            device  = config.get("device", "cpu"),
            project = config.get("project_path"),
            name    = f"yolo_autotrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task    = task,
            verbose = False,
        )
        logger.info("YOLOオートトレーニング完了")
    except Exception:
        logger.error("YOLOオートトレーニングエラー:\n%s", traceback.format_exc())
    finally:
        autotrain_active = False


def _auto_label_yolo(
    yolo: YOLO, config: Dict[str, Any], task: str
) -> None:
    """
    YOLOモデルで全画像に自動ラベルを付与する。
    タスク種別に応じたラベルフォーマットで書き出す。
    """
    project_path = config["project_path"]
    conf = float(config.get("conf_threshold", 0.25))
    iou  = float(config.get("iou_threshold",  0.45))

    for split in ("train", "val"):
        imgs_dir   = Path(project_path) / split / "images"
        labels_dir = Path(project_path) / split / "labels"
        if not imgs_dir.exists():
            continue
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_file in imgs_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            try:
                results_list = yolo.predict(
                    str(img_file), conf=conf, iou=iou, verbose=False
                )
                if not results_list:
                    continue
                res    = results_list[0]
                img_h, img_w = res.orig_shape
                lbl    = labels_dir / (img_file.stem + ".txt")
                lines: List[str] = []

                if task == "detect" and res.boxes is not None:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        xc = ((x1 + x2) / 2) / img_w
                        yc = ((y1 + y2) / 2) / img_h
                        bw = (x2 - x1) / img_w
                        bh = (y2 - y1) / img_h
                        lines.append(f"{int(box.cls[0])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                elif task == "segment" and res.masks is not None:
                    for box, mask in zip(res.boxes, res.masks.xy):
                        pts_str = " ".join(
                            f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in mask
                        )
                        lines.append(f"{int(box.cls[0])} {pts_str}")

                elif task == "obb" and res.obb is not None:
                    for obb in res.obb.xyxyxyxy:
                        pts = obb.cpu().numpy().reshape(-1, 2)
                        cls  = int(res.boxes[0].cls[0]) if res.boxes else 0
                        pts_str = " ".join(
                            f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in pts
                        )
                        lines.append(f"{cls} {pts_str}")

                lbl.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                logger.warning("YOLOオートラベリング失敗: %s", img_file)


def _backup_project(project_path: str) -> None:
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk_dir = Path(project_path) / f"backup_{ts}"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            src = Path(project_path) / split / sub
            if src.exists():
                shutil.copytree(str(src), str(bk_dir / split / sub))
    logger.info("バックアップ作成: %s", bk_dir)


# ══════════════════════════════════════════════
# Flask ルート定義 (BILTサービスと同一APIサーフェス)
# ══════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "yolo-service", "version": "2.0.0"})


# ── モデル管理 ─────────────────────────────────

@app.route("/models/available", methods=["POST"])
def get_available_models():
    task_type = (request.json or {}).get("task_type", "detect")

    # タスク別モデルサフィックス
    suffixes: Dict[str, List[str]] = {
        "detect":  [".pt"],
        "segment": ["-seg.pt", ".pt"],
        "obb":     ["-obb.pt", ".pt"],
        "pose":    ["-pose.pt", ".pt"],
    }
    compatible_sfx = suffixes.get(task_type, [".pt"])

    # スクラッチ (Ultralytics プリセット自動DL)
    preset = {
        "detect":  "yolov8n.pt",
        "segment": "yolov8n-seg.pt",
        "obb":     "yolov8n-obb.pt",
        "pose":    "yolov8n-pose.pt",
    }.get(task_type, "yolov8n.pt")

    models = [{
        "name":        "scratch",
        "compatible":  True,
        "description": f"新規モデル ({preset}) からスクラッチ学習",
    }]

    models_dir = Path(dirs.models)
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.suffix not in (".pt", ".onnx", ".engine"):
                continue
            # タスク互換性チェック (サフィックスベース)
            compatible = any(f.name.endswith(sfx.lstrip(".") if sfx.startswith(".") else sfx)
                             for sfx in compatible_sfx)
            models.append({
                "name":        f.name,
                "compatible":  compatible,
                "description": f"ローカルモデル: {f.name}",
            })

    return jsonify({"models": models})


@app.route("/api/model/load", methods=["POST"])
def load_model():
    global current_model, model_info
    name = (request.json or {}).get("model_name", "")
    path = Path(dirs.models) / name
    try:
        yolo = YOLO(str(path) if path.exists() else name)
        current_model = yolo
        # タスク自動検出
        task = getattr(yolo, "task", "detect") or "detect"
        detection_settings["task"] = task
        model_info = {
            "name":        name,
            "classes":     list(yolo.names.values()) if yolo.names else [],
            "loaded":      True,
            "class_count": len(yolo.names) if yolo.names else 0,
            "task":        task,
        }
        logger.info("YOLOモデルをロードしました: %s (タスク: %s)", name, task)
        return jsonify({"success": True, "model_info": model_info})
    except Exception as exc:
        logger.error("モデルロードエラー: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/model/info")
def model_info_route():
    return jsonify({"success": True, "model_info": model_info})


# ── カメラ管理 ─────────────────────────────────

@app.route("/api/cameras")
def get_cameras():
    return jsonify({"success": True, "cameras": _get_available_cameras()})


@app.route("/api/camera/select", methods=["POST"])
def select_camera():
    idx = int((request.json or {}).get("camera_index", 0))
    return jsonify({"success": _init_camera(idx)})


@app.route("/api/camera/resolution", methods=["POST"])
def set_camera_resolution():
    data = request.json or {}
    w, h = int(data.get("width", 1280)), int(data.get("height", 960))
    with _camera_lock:
        if _camera and _camera.isOpened():
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return jsonify({"success": True, "width": w, "height": h})


@app.route("/api/camera/info")
def camera_info():
    with _camera_lock:
        if _camera and _camera.isOpened():
            return jsonify({"success": True, "info": {
                "index":  _camera_index,
                "width":  int(_camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":    _camera.get(cv2.CAP_PROP_FPS),
            }})
    return jsonify({"success": False, "error": "カメラが初期化されていません"})


# ── 検出制御 ──────────────────────────────────

@app.route("/api/detection/settings", methods=["GET", "POST"])
def det_settings():
    if request.method == "POST":
        detection_settings.update(request.json or {})
        return jsonify({"success": True, "settings": detection_settings})
    return jsonify({"success": True, "settings": detection_settings})


@app.route("/api/detection/start", methods=["POST"])
def start_detection():
    global detection_active, detection_thread
    if not detection_active:
        detection_active = True
        if detection_thread is None or not detection_thread.is_alive():
            detection_thread = threading.Thread(
                target=_detection_loop, daemon=True, name="yolo-detect"
            )
            detection_thread.start()
    return jsonify({"success": True})


@app.route("/api/detection/stop", methods=["POST"])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"success": True})


@app.route("/api/detection/stats")
def detection_stats_route():
    return jsonify({"success": True, "stats": detection_stats})


@app.route("/api/frame/latest")
def latest_frame_route():
    with frame_lock:
        f = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
    ret, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ret:
        return Response(buf.tobytes(), mimetype="image/jpeg")
    return jsonify({"error": "エンコード失敗"}), 500


# ── カウンター ────────────────────────────────

@app.route("/api/counters")
def get_counters():
    return jsonify({"success": True, "counters": object_counters})


@app.route("/api/counters/reset", methods=["POST"])
def reset_counters():
    global object_counters, counter_triggered
    object_counters   = {}
    counter_triggered = {}
    detection_stats["total_detections"] = 0
    return jsonify({"success": True})


# ── チェーン検出 ──────────────────────────────

@app.route("/api/chain/status")
def chain_status_route():
    steps    = detection_settings["chain_steps"]
    step_idx = chain_state["current_step"]
    step_cfg = steps[step_idx] if steps and step_idx < len(steps) else None
    now      = time.time()
    remaining = max(0.0, detection_settings["chain_timeout"] -
                    (now - (chain_state["step_start_time"] or now)))
    pause_rem = 0.0
    if chain_state["cycle_pause"] and chain_state["cycle_pause_start"]:
        pause_rem = max(0.0, detection_settings["chain_pause_time"] -
                        (now - chain_state["cycle_pause_start"]))
    return jsonify({"success": True, "status": {
        **chain_state,
        "total_steps":         len(steps),
        "current_step_config": step_cfg,
        "remaining_time":      remaining,
        "pause_remaining":     pause_rem,
    }})


@app.route("/api/chain/control", methods=["POST"])
def chain_control():
    action = (request.json or {}).get("action", "")
    if action == "start":
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
    elif action == "stop":
        detection_settings["chain_mode"] = False
        chain_state["active"] = False
    elif action == "reset":
        chain_state.update({
            "current_step": 0, "step_start_time": time.time(),
            "step_history": [], "last_step_result": None,
        })
    return jsonify({"success": True})


@app.route("/api/chain/config", methods=["GET", "POST"])
def chain_config():
    keys = ["chain_steps", "chain_timeout", "chain_auto_advance", "chain_pause_time"]
    if request.method == "POST":
        for k in keys:
            if k in (request.json or {}):
                detection_settings[k] = request.json[k]
    return jsonify({"success": True, "config": {k: detection_settings[k] for k in keys}})


@app.route("/api/chain/acknowledge_error", methods=["POST"])
def ack_error():
    if chain_state.get("waiting_for_ack"):
        chain_state.update({
            "waiting_for_ack": False, "error_message": None,
            "wrong_object": None, "last_step_result": None,
            "step_start_time": time.time(),
            "current_step": chain_state.get("error_step", chain_state["current_step"]),
            "error_step": None,
        })
        return jsonify({"success": True})
    return jsonify({"success": False})


def _chains_dir() -> Path:
    p = Path(dirs.base) / "chains"
    p.mkdir(exist_ok=True)
    return p


@app.route("/api/chains/saved")
def saved_chains():
    chains = []
    for f in _chains_dir().glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            chains.append({
                "name":       data.get("name", f.stem),
                "model_name": data.get("model_name", ""),
                "steps":      len(data.get("chain_steps", [])),
                "created":    data.get("created", ""),
                "filename":   f.stem,
            })
        except Exception:
            pass
    return jsonify({"success": True, "chains": chains})


@app.route("/api/chains/save", methods=["POST"])
def save_chain():
    data = request.json or {}
    name = secure_filename(data.get("chain_name", "").strip())
    if not name:
        return jsonify({"success": False, "error": "チェーン名が必要です"}), 400
    payload = {
        "name": name, "model_name": data.get("model_name", ""),
        "created": datetime.now().isoformat(),
        "chain_steps": detection_settings["chain_steps"],
        "chain_timeout": detection_settings["chain_timeout"],
        "chain_auto_advance": detection_settings["chain_auto_advance"],
        "chain_pause_time": detection_settings["chain_pause_time"],
    }
    (_chains_dir() / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"success": True})


@app.route("/api/chains/load", methods=["POST"])
def load_chain():
    name = secure_filename((request.json or {}).get("chain_name", ""))
    path = _chains_dir() / f"{name}.json"
    if not path.exists():
        return jsonify({"success": False, "error": "チェーンが見つかりません"}), 404
    data = json.loads(path.read_text(encoding="utf-8"))
    detection_settings["chain_steps"]        = data["chain_steps"]
    detection_settings["chain_timeout"]      = data.get("chain_timeout", 50.0)
    detection_settings["chain_auto_advance"] = data.get("chain_auto_advance", True)
    detection_settings["chain_pause_time"]   = data.get("chain_pause_time", 10.0)
    return jsonify({"success": True, "chain_data": data})


@app.route("/api/chains/delete", methods=["POST"])
def delete_chain():
    name = secure_filename((request.json or {}).get("chain_name", ""))
    path = _chains_dir() / f"{name}.json"
    if path.exists():
        path.unlink()
    return jsonify({"success": True})


# ── トレーニング ──────────────────────────────

@app.route("/train/start", methods=["POST"])
def train_start():
    global training_active
    if training_active:
        return jsonify({"error": "トレーニングはすでに実行中です"}), 400
    config = request.json or {}
    threading.Thread(target=_training_worker, args=(config,), daemon=True).start()
    return jsonify({"success": True, "message": "YOLOトレーニングを開始しました"})


@app.route("/train/status")
def train_status():
    return jsonify({"active": training_active,
                    "status": "running" if training_active else "idle"})


@app.route("/autotrain/start", methods=["POST"])
def autotrain_start():
    global autotrain_active
    if autotrain_active:
        return jsonify({"error": "オートトレーニングはすでに実行中です"}), 400
    config = request.json or {}
    if not config.get("model_path"):
        return jsonify({"error": "model_path が必要です"}), 400
    threading.Thread(target=_autotrain_worker, args=(config,), daemon=True).start()
    return jsonify({"success": True})


@app.route("/autotrain/status")
def autotrain_status():
    return jsonify({"running": autotrain_active})


@app.route("/relabel/start", methods=["POST"])
def relabel_start():
    config = request.json or {}
    model_path = config.get("model_path", "")
    task       = config.get("task_type", "detect")
    try:
        yolo = YOLO(model_path)
        if config.get("backup_enabled", True):
            _backup_project(config["project_path"])
        _auto_label_yolo(yolo, config, task)
        return jsonify({"success": True, "message": "YOLOリラベリングが完了しました"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── プロジェクト管理 ────────────────────────────

@app.route("/api/projects")
def list_projects():
    pdir  = Path(dirs.projects)
    projs = [d.name for d in pdir.iterdir() if d.is_dir()] if pdir.exists() else []
    return jsonify({"success": True, "projects": projs})


@app.route("/api/projects/create", methods=["POST"])
def create_project():
    data = request.json or {}
    name = secure_filename(data.get("project_name", "").strip())
    if not name:
        return jsonify({"success": False, "error": "プロジェクト名が必要です"}), 400
    path = Path(dirs.projects) / name
    if path.exists():
        return jsonify({"success": False, "error": "プロジェクトは既に存在します"}), 400
    for sub in ("images", "labels"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    info = {
        "name":        name,
        "created":     datetime.now().isoformat(),
        "description": data.get("description", ""),
        "classes":     data.get("classes", []),
    }
    (path / "project_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"success": True, "project": info})


# ══════════════════════════════════════════════
# シャットダウン
# ══════════════════════════════════════════════

@atexit.register
def _cleanup():
    global detection_active
    detection_active = False
    with _camera_lock:
        if _camera:
            _camera.release()
    logger.info("YOLOサービス: クリーンアップ完了")


if __name__ == "__main__":
    port = Ports.YOLO_SERVICE
    logger.info("=" * 60)
    logger.info("YOLO Detection Service (Ultralytics AGPL-3.0)")
    logger.info("ポート %d で起動中...", port)
    logger.info("サポートタスク: detect / segment / obb / pose")
    logger.info("=" * 60)

    detection_thread = threading.Thread(
        target=_detection_loop, daemon=True, name="yolo-detect"
    )
    detection_thread.start()

    app.run(host="127.0.0.1", port=port, debug=False,
            threaded=True, use_reloader=False)
