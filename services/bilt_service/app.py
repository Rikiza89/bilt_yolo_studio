"""
services/bilt_service/app.py — BILTバックエンドサービス (AGPL-3.0)

このプロセスはUIとは完全に分離された独立プロセスとして動作する。
HTTP APIを通じてのみUIと通信し、静的リンクを行わない。

ライセンス分離の根拠:
  BILTライブラリはAGPL-3.0ライセンスである。
  UIコード (PySide6: LGPL) がこのサービスをインポートすると
  UIコード全体がAGPLの影響を受ける可能性がある。
  プロセス分離により、HTTP境界がライセンス境界となる。
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

# BILTライブラリのインポート (AGPL-3.0)
from bilt import BILT  # noqa: E402  (AGPL)

from shared.camera_utils import CameraManager             # noqa: E402
from shared.contracts import AppDirectories, Ports, TaskType  # noqa: E402
from shared.detection_common import (                      # noqa: E402
    acknowledge_chain_error,
    backup_project,
    make_default_chain_state,
    make_default_detection_settings,
    make_default_detection_stats,
    process_chain,
    sanitize_detection_settings,
    start_chain,
)

# ──────────────────────────────────────────────
# アプリケーション初期化
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BILT-SVC] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_ROOT / "bilt_service.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
dirs = AppDirectories(base=str(_ROOT))
dirs.ensure_all()

# ──────────────────────────────────────────────
# グローバル状態
# ──────────────────────────────────────────────

# Thread-safe camera manager (DRY: shared with YOLO service)
camera = CameraManager()

# Model state — protected by _model_lock
_model_lock = threading.Lock()
current_model: Optional[BILT] = None
model_info: Dict[str, Any] = {"name": None, "classes": [], "loaded": False}

# Detection loop control — threading.Event for safe cross-thread signaling
_detection_event = threading.Event()
detection_thread: Optional[threading.Thread] = None
latest_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
counter_triggered: Dict[str, bool] = {}

# Training state — protected by _training_lock
_training_lock = threading.Lock()
training_active  = False
autotrain_active = False

# Detection settings / stats / chain state (DRY: shared factories)
detection_settings: Dict[str, Any] = make_default_detection_settings()
object_counters: Dict[str, int] = {}
detection_stats: Dict[str, Any] = make_default_detection_stats()
chain_state: Dict[str, Any] = make_default_chain_state()


# ══════════════════════════════════════════════
# 検出ループ
# ══════════════════════════════════════════════

def _detection_loop() -> None:
    """
    カメラフレームを連続取得し、BILTで推論するメインループ。
    _detection_event が clear されるまで動作し続ける。
    """
    global latest_frame, object_counters, counter_triggered

    fps_counter  = 0
    fps_time     = time.time()
    frame_period = 1.0 / 30

    while _detection_event.is_set():
        t0 = time.time()
        try:
            frame = camera.grab_frame()
            if frame is None:
                if camera.camera_index is not None:
                    logger.warning(
                        "検出ループ: フレーム取得失敗。カメラを再初期化します (index=%d)",
                        camera.camera_index,
                    )
                    camera.init_camera(camera.camera_index)
                time.sleep(0.1)
                continue

            annotated = frame.copy()

            with _model_lock:
                model = current_model
                loaded = model_info["loaded"]

            if model and loaded:
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                bilt_dets = model.predict(
                    pil,
                    conf=detection_settings["conf"],
                    iou=detection_settings["iou"],
                )

                # カウンターモード処理
                if detection_settings["counter_mode"] and not detection_settings["chain_mode"]:
                    for det in bilt_dets:
                        name = det["class_name"]
                        if name not in counter_triggered:
                            object_counters[name] = object_counters.get(name, 0) + 1
                            counter_triggered[name] = True

                # チェーンモード処理 (DRY: shared logic)
                if detection_settings["chain_mode"]:
                    process_chain(bilt_dets, detection_settings, chain_state)

                annotated = _draw_detections(frame.copy(), bilt_dets)
                detection_stats["total_detections"]     += len(bilt_dets)
                detection_stats["detections_per_frame"]  = len(bilt_dets)
                detection_stats["current_classes"]       = list({d["class_name"] for d in bilt_dets})
                detection_stats["last_detection_time"]   = datetime.now().isoformat()

            # FPS計算
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


def _draw_detections(frame: np.ndarray, dets: List[Dict]) -> np.ndarray:
    """検出結果をフレームに描画して返す。"""
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 165, 0), (128, 0, 128), (0, 128, 255),
    ]
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid   = det.get("class_id", 0)
        color = colors[cid % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('class_name', '')} {det.get('score', 0):.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


# ══════════════════════════════════════════════
# トレーニングワーカー
# ══════════════════════════════════════════════

def _training_worker(config: Dict[str, Any]) -> None:
    """バックグラウンドスレッドでBILTトレーニングを実行する。"""
    global training_active
    try:
        with _training_lock:
            training_active = True
        model_name = config.get("model_name") or config.get("model")

        if not model_name or model_name == "scratch":
            model = BILT()
        else:
            mp = Path(dirs.models) / model_name
            model = BILT(str(mp)) if mp.exists() else BILT()

        results = model.train(
            dataset      = os.path.dirname(config["data_yaml"]),
            epochs       = int(config.get("epochs", 100)),
            batch_size   = int(config.get("batch_size", config.get("batch", 8))),
            img_size     = int(config.get("img_size", config.get("imgsz", 640))),
            learning_rate= float(config.get("learning_rate", config.get("lr0", 0.001))),
            device       = config.get("device", "cpu"),
            save_dir     = config.get("project_path", str(_ROOT / "runs")),
            name         = f"bilt_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        custom = config.get("custom_save_name")
        if custom:
            if not custom.endswith(".pth"):
                custom += ".pth"
            run_dir = Path(config.get("project_path", str(_ROOT))) / results.get("name", "")
            src = run_dir / "weights" / "best.pth"
            if src.exists():
                shutil.copy2(str(src), str(Path(dirs.models) / custom))
                logger.info("モデルを %s として保存しました", custom)

        logger.info("BILTトレーニング完了: %s", results)
    except Exception:
        logger.error("BILTトレーニングエラー:\n%s", traceback.format_exc())
    finally:
        with _training_lock:
            training_active = False


def _autotrain_worker(config: Dict[str, Any]) -> None:
    """オートラベリング → トレーニングを実行するワーカー。"""
    global autotrain_active
    try:
        with _training_lock:
            autotrain_active = True
        model_path = config["model_path"]
        model = BILT(model_path)

        if config.get("backup_enabled", True):
            backup_project(config["project_path"])

        _auto_label(model, config)

        model.train(
            dataset      = os.path.dirname(config["data_yaml"]),
            epochs       = int(config.get("epochs", 50)),
            batch_size   = int(config.get("batch", 8)),
            img_size     = int(config.get("imgsz", 640)),
            learning_rate= float(config.get("lr0", 0.001)),
            device       = config.get("device", "cpu"),
            save_dir     = config.get("project_path"),
            name         = f"autotrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        logger.info("オートトレーニング完了")
    except Exception:
        logger.error("オートトレーニングエラー:\n%s", traceback.format_exc())
    finally:
        with _training_lock:
            autotrain_active = False


def _auto_label(model: BILT, config: Dict[str, Any]) -> None:
    """BILTモデルで全画像に自動ラベルを付与する。"""
    from PIL import Image as PILImage
    project_path = config["project_path"]
    conf = float(config.get("conf_threshold", 0.25))
    iou  = float(config.get("iou_threshold", 0.45))

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
                img  = PILImage.open(img_file)
                w, h = img.size
                dets = model.predict(img, conf=conf, iou=iou)
                lbl  = labels_dir / (img_file.stem + ".txt")
                with open(lbl, "w", encoding="utf-8") as f:
                    for d in dets:
                        x1, y1, x2, y2 = d["bbox"]
                        xc = ((x1 + x2) / 2) / w
                        yc = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{d['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            except Exception:
                logger.warning("自動ラベリング失敗: %s", img_file)


# ══════════════════════════════════════════════
# Flask ルート定義
# ══════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "bilt-service", "version": "2.0.0"})


# ── モデル管理 ─────────────────────────────────

@app.route("/models/available", methods=["POST"])
def get_available_models():
    task_type = (request.json or {}).get("task_type", "detect")
    models = [{"name": "scratch", "compatible": True, "description": "新規モデルをスクラッチから学習"}]
    models_dir = Path(dirs.models)
    if models_dir.exists():
        for f in models_dir.glob("*.pth"):
            models.append({
                "name":        f.name,
                "compatible":  task_type == "detect",  # BILTは検出のみ
                "description": f"ローカルモデル: {f.name}",
            })
    return jsonify({"models": models})


@app.route("/api/model/load", methods=["POST"])
def load_model():
    global current_model, model_info
    name = (request.json or {}).get("model_name", "")
    path = Path(dirs.models) / name
    if not path.exists():
        return jsonify({"success": False, "error": "モデルが見つかりません"}), 400
    try:
        loaded = BILT(str(path))
        with _model_lock:
            current_model = loaded
            model_info = {
                "name":        name,
                "classes":     loaded.names,
                "loaded":      True,
                "class_count": loaded.num_classes,
            }
        logger.info("モデルをロードしました: %s (%d クラス)", name, loaded.num_classes)
        return jsonify({"success": True, "model_info": model_info})
    except Exception as exc:
        logger.error("モデルロードエラー: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/model/info")
def model_info_route():
    return jsonify({"success": True, "model_info": model_info})


# ── カメラ管理 (DRY: CameraManager) ──────────────

@app.route("/api/cameras")
def get_cameras():
    return jsonify({"success": True, "cameras": camera.get_available_cameras()})


@app.route("/api/camera/select", methods=["POST"])
def select_camera():
    try:
        idx = int((request.json or {}).get("camera_index", 0))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "camera_index が無効です"}), 400
    return jsonify({"success": camera.init_camera(idx)})


@app.route("/api/camera/resolution", methods=["POST"])
def set_camera_resolution():
    data = request.json or {}
    try:
        w, h = int(data.get("width", 1280)), int(data.get("height", 960))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "解像度パラメータが無効です"}), 400
    camera.set_resolution(w, h)
    return jsonify({"success": True, "width": w, "height": h})


@app.route("/api/camera/info")
def camera_info():
    info = camera.get_info()
    if info:
        return jsonify({"success": True, "info": info})
    return jsonify({"success": False, "error": "カメラが初期化されていません"})


@app.route("/api/camera/preview")
def camera_preview():
    """検出オーバーレイなしの生フレームをJPEGで返す。"""
    frame = camera.grab_frame_with_retry()
    if frame is None:
        frame = np.zeros((480, 640, 3), np.uint8)
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ret:
        return Response(buf.tobytes(), mimetype="image/jpeg")
    return jsonify({"error": "エンコード失敗"}), 500


@app.route("/api/camera/snapshot", methods=["POST"])
def camera_snapshot():
    """現在のカメラフレームをプロジェクトの images フォルダに保存する。"""
    data         = request.json or {}
    project_name = data.get("project_name", "").strip()
    filename     = data.get("filename", "").strip()

    if not project_name:
        return jsonify({"success": False, "error": "project_name が指定されていません"}), 400

    frame = camera.grab_frame_with_retry()

    if frame is None:
        return jsonify({"success": False, "error": "カメラが初期化されていません。サイドバーで再接続してください"}), 400

    if not filename:
        filename = datetime.now().strftime("cap_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"

    img_dir = Path(dirs.projects) / project_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dest = img_dir / filename

    ret = cv2.imwrite(str(dest), frame)
    if not ret:
        return jsonify({"success": False, "error": "画像の書き込みに失敗しました"}), 500

    logger.info("スナップショット保存: %s", dest)
    return jsonify({"success": True, "filename": filename, "path": str(dest)})


# ── 検出制御 ──────────────────────────────────

@app.route("/api/detection/settings", methods=["GET", "POST"])
def det_settings():
    if request.method == "POST":
        # Security: whitelist allowed keys to prevent state injection
        safe = sanitize_detection_settings(request.json or {})
        detection_settings.update(safe)
        return jsonify({"success": True, "settings": detection_settings})
    return jsonify({"success": True, "settings": detection_settings})


@app.route("/api/detection/start", methods=["POST"])
def start_detection():
    """検出ループを開始する。モデル未ロードならエラーを返す。"""
    global detection_thread
    with _model_lock:
        if not model_info["loaded"] or current_model is None:
            return jsonify({
                "success": False,
                "error":   "モデルが未ロードです。先にモデルをロードしてください。"
            }), 400
    if not _detection_event.is_set():
        _detection_event.set()
        if detection_thread is None or not detection_thread.is_alive():
            detection_thread = threading.Thread(
                target=_detection_loop, daemon=True, name="bilt-detect"
            )
            detection_thread.start()
    return jsonify({"success": True})


@app.route("/api/detection/stop", methods=["POST"])
def stop_detection():
    _detection_event.clear()
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


# ── チェーン検出 (DRY: shared logic) ──────────

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
        start_chain(detection_settings, chain_state)
    elif action == "stop":
        detection_settings["chain_mode"] = False
        chain_state["active"] = False
    elif action == "reset":
        chain_state.update({
            "current_step": 0, "step_start_time": time.time(),
            "step_history": [], "current_detections": {},
            "last_step_result": None,
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
    if acknowledge_chain_error(chain_state):
        return jsonify({"success": True, "message": "エラーを確認しました。ステップを再試行します"})
    return jsonify({"success": False, "message": "確認すべきエラーがありません"})


# ── チェーン保存・読み込み ──────────────────────

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
    data       = request.json or {}
    chain_name = secure_filename(data.get("chain_name", "").strip())
    if not chain_name:
        return jsonify({"success": False, "error": "チェーン名が必要です"}), 400
    payload = {
        "name":               chain_name,
        "model_name":         data.get("model_name", ""),
        "created":            datetime.now().isoformat(),
        "chain_steps":        detection_settings["chain_steps"],
        "chain_timeout":      detection_settings["chain_timeout"],
        "chain_auto_advance": detection_settings["chain_auto_advance"],
        "chain_pause_time":   detection_settings["chain_pause_time"],
    }
    (_chains_dir() / f"{chain_name}.json").write_text(
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
    with _training_lock:
        if training_active:
            return jsonify({"error": "トレーニングはすでに実行中です"}), 400
    config = request.json or {}
    threading.Thread(target=_training_worker, args=(config,), daemon=True).start()
    return jsonify({"success": True, "message": "BILTトレーニングを開始しました"})


@app.route("/train/status")
def train_status():
    return jsonify({"active": training_active,
                    "status": "running" if training_active else "idle"})


@app.route("/autotrain/start", methods=["POST"])
def autotrain_start():
    with _training_lock:
        if autotrain_active:
            return jsonify({"error": "オートトレーニングはすでに実行中です"}), 400
    config = request.json or {}
    if not config.get("model_path") or not Path(config["model_path"]).exists():
        return jsonify({"error": "モデルパスが無効です"}), 400
    threading.Thread(target=_autotrain_worker, args=(config,), daemon=True).start()
    return jsonify({"success": True, "message": "オートトレーニングを開始しました"})


@app.route("/autotrain/status")
def autotrain_status():
    return jsonify({"running": autotrain_active})


@app.route("/relabel/start", methods=["POST"])
def relabel_start():
    config = request.json or {}
    model_path = config.get("model_path", "")
    if not Path(model_path).exists():
        return jsonify({"error": "モデルパスが無効です"}), 400
    try:
        model = BILT(model_path)
        if config.get("backup_enabled", True):
            backup_project(config["project_path"])
        _auto_label(model, config)
        return jsonify({"success": True, "message": "リラベリングが完了しました"})
    except Exception as exc:
        logger.error("リラベリングエラー: %s", exc)
        return jsonify({"error": str(exc)}), 400


# ── プロジェクト管理 ────────────────────────────

@app.route("/api/projects")
def list_projects():
    pdir = Path(dirs.projects)
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
# シャットダウンフック
# ══════════════════════════════════════════════

@atexit.register
def _cleanup():
    _detection_event.clear()
    camera.release()
    logger.info("BILTサービス: クリーンアップ完了")


# ══════════════════════════════════════════════
# エントリーポイント
# ══════════════════════════════════════════════

if __name__ == "__main__":
    port = Ports.BILT_SERVICE
    logger.info("=" * 60)
    logger.info("BILT Detection Service (AGPL-3.0)")
    logger.info("ポート %d で起動中...", port)
    logger.info("モデルディレクトリ: %s", dirs.models)
    logger.info("=" * 60)

    # NOTE: Detection thread is started on-demand via /api/detection/start
    # to avoid wasted CPU when no model is loaded.

    app.run(host="127.0.0.1", port=port, debug=False,
            threaded=True, use_reloader=False)
