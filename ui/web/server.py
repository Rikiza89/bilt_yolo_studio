"""
ui/web/server.py — BILT+YOLO Studio UIサーバー

このモジュールはUIプロセス内で動作するFlaskサーバーを定義する。
QWebEngineViewから接続されるローカルHTTPサーバーとして機能し、
BILT/YOLOバックエンドサービスへのプロキシ兼オーケストレーターとなる。

ライセンス上の位置づけ:
  このファイルはBILT(AGPL)・Ultralytics(AGPL)を一切importしない。
  バックエンドとの通信はHTTPのみ(service_clientを通じて)。
  PySide6(LGPL)のUI層に属する。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# shared パッケージへのパスを解決する
# このファイルは ui/web/server.py に配置されるため、
# プロジェクトルートへの参照が必要
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from shared.contracts import AppDirectories, EngineType, Ports, TaskType
from shared.service_client import ServiceClient

# ──────────────────────────────────────────────
# ロガー設定
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Flaskアプリケーション初期化
# templatesディレクトリを明示的に指定する
# ──────────────────────────────────────────────
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_STATIC_DIR   = Path(__file__).parent / "static"

app = Flask(
    __name__,
    template_folder=str(_TEMPLATE_DIR),
    static_folder=str(_STATIC_DIR),
)

# ──────────────────────────────────────────────
# グローバルサービスクライアント
# UIサーバーの起動時に一度だけ初期化する
# ──────────────────────────────────────────────
_client: ServiceClient | None = None
_dirs:   AppDirectories | None = None

# DRY: 1x1 transparent PNG for empty-frame fallback (used by preview & frame endpoints)
_EMPTY_1X1_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def init_server(dirs: AppDirectories) -> None:
    """
    サーバーを初期化する。

    アプリ起動時にランチャーから呼び出される。
    サービスクライアントとディレクトリ参照を設定する。

    Args:
        dirs: アプリケーションディレクトリ設定。
    """
    global _client, _dirs
    _client = ServiceClient(
        bilt_url=f"http://127.0.0.1:{Ports.BILT_SERVICE}",
        yolo_url=f"http://127.0.0.1:{Ports.YOLO_SERVICE}",
    )
    _dirs = dirs
    dirs.ensure_all()
    logger.info("UIサーバー初期化完了")


def _svc() -> ServiceClient:
    """
    サービスクライアントを取得する。

    初期化前に呼び出された場合は RuntimeError を送出する。
    """
    if _client is None:
        raise RuntimeError("init_server() が呼ばれていない")
    return _client


# ═══════════════════════════════════════════════
# ページルーティング
# ═══════════════════════════════════════════════

@app.route("/")
def index() -> Response:
    """メインページ — リアルタイム検出タブにリダイレクト。"""
    return redirect(url_for("detection_page"))


@app.route("/detection")
def detection_page() -> str:
    """リアルタイム検出ページ。"""
    return render_template("detection.html")


@app.route("/annotation")
def annotation_page() -> str:
    """アノテーションページ。"""
    projects = _list_projects()
    return render_template("annotation.html", projects=projects)


@app.route("/training")
def training_page() -> str:
    """トレーニングページ。"""
    projects  = _list_projects()
    return render_template("training.html", projects=projects)


@app.route("/chain")
def chain_page() -> str:
    """チェーン検出ページ。"""
    return render_template("chain.html")


@app.route("/settings")
def settings_page() -> str:
    """設定ページ。"""
    return render_template("settings.html")


# ═══════════════════════════════════════════════
# APIプロキシエンドポイント
# バックエンドサービスへのリクエストをUIサーバー経由で中継する
# UIのシングルオリジン制約を満たしつつ、
# バックエンドURLをフロントエンドから隠蔽する
# ═══════════════════════════════════════════════

@app.route("/api/health")
def api_health() -> Response:
    """
    全サービスのヘルスチェックを集約して返す。

    両サービスへのリクエストをスレッドで並列実行し、
    直列実行時に比べてレスポンス時間を半減する。
    タイムアウトは短め (3秒) に設定し、UIのブロッキングを防ぐ。
    """
    import concurrent.futures

    def check(engine: EngineType) -> Dict[str, Any]:
        return _svc()._get(engine, "/health", timeout=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_bilt = ex.submit(check, EngineType.BILT)
        f_yolo = ex.submit(check, EngineType.YOLO)
        bilt_result = f_bilt.result()
        yolo_result = f_yolo.result()

    return jsonify({
        "bilt": bilt_result,
        "yolo": yolo_result,
        "ui":   True,
    })


# ── モデル ────────────────────────────────────

@app.route("/api/models/available", methods=["POST"])
def api_models_available() -> Response:
    """指定エンジンの利用可能モデル一覧を返す。"""
    engine = _engine_from_request()
    result = _svc().post(engine, "/models/available", request.json or {})
    return jsonify(result)


@app.route("/api/model/load", methods=["POST"])
def api_model_load() -> Response:
    """モデルをロードする。"""
    engine = _engine_from_request()
    result = _svc().post(engine, "/api/model/load", request.json or {})
    return jsonify(result)


@app.route("/api/model/info")
def api_model_info() -> Response:
    """現在ロード中のモデル情報を返す。"""
    engine = _engine_from_query()
    result = _svc().get(engine, "/api/model/info")
    return jsonify(result)


# ── カメラ ───────────────────────────────────

@app.route("/api/cameras")
def api_cameras() -> Response:
    """利用可能なカメラ一覧を返す。"""
    engine = _engine_from_query()
    result = _svc().get(engine, "/api/cameras")
    return jsonify(result)


@app.route("/api/camera/select", methods=["POST"])
def api_camera_select() -> Response:
    """カメラを選択する。"""
    engine = _engine_from_request()
    result = _svc().post(engine, "/api/camera/select", request.json or {})
    return jsonify(result)


@app.route("/api/camera/preview")
def api_camera_preview() -> Response:
    """
    アノテーション用のクリーンなカメラフレームを返す。
    検出オーバーレイは含まない生フレーム。
    """
    engine = _engine_from_query()
    jpeg   = _svc().get_raw(engine, "/api/camera/preview")
    if jpeg is None:
        return Response(_EMPTY_1X1_PNG, mimetype="image/png")
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/camera/snapshot", methods=["POST"])
def api_camera_snapshot() -> Response:
    """カメラフレームをプロジェクトに保存する。"""
    engine = _engine_from_request()
    result = _svc().post(engine, "/api/camera/snapshot", request.json or {})
    return jsonify(result)


@app.route("/api/camera/resolution", methods=["POST"])
def api_camera_resolution() -> Response:
    """解像度を設定する。"""
    engine = _engine_from_request()
    result = _svc().post(engine, "/api/camera/resolution", request.json or {})
    return jsonify(result)


# ── 検出制御 ─────────────────────────────────

@app.route("/api/detection/settings", methods=["GET", "POST"])
def api_detection_settings() -> Response:
    """検出設定を取得/更新する。"""
    engine = _engine_from_query() if request.method == "GET" else _engine_from_request()
    if request.method == "GET":
        return jsonify(_svc().get(engine, "/api/detection/settings"))
    return jsonify(_svc().post(engine, "/api/detection/settings", request.json or {}))


@app.route("/api/detection/start", methods=["POST"])
def api_detection_start() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/detection/start", request.json or {}))


@app.route("/api/detection/stop", methods=["POST"])
def api_detection_stop() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/detection/stop", {}))


@app.route("/api/detection/stats")
def api_detection_stats() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/api/detection/stats"))


# ── フレームプロキシ ─────────────────────────
# バックエンドから取得したJPEGフレームを中継する
# MIME typeを維持しQWebEngineViewで正常表示させる

@app.route("/api/frame/latest")
def api_frame_latest() -> Response:
    """最新の検出フレームをJPEGとして返す。"""
    engine = _engine_from_query()
    jpeg = _svc().get_raw(engine, "/api/frame/latest")
    if jpeg is None:
        return Response(_EMPTY_1X1_PNG, mimetype="image/png")
    return Response(jpeg, mimetype="image/jpeg")


# ── カウンター ───────────────────────────────

@app.route("/api/counters")
def api_counters() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/api/counters"))


@app.route("/api/counters/reset", methods=["POST"])
def api_counters_reset() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/counters/reset", {}))


# ── チェーン検出 ─────────────────────────────

@app.route("/api/chain/status")
def api_chain_status() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/api/chain/status"))


@app.route("/api/chain/control", methods=["POST"])
def api_chain_control() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/chain/control", request.json or {}))


@app.route("/api/chain/config", methods=["GET", "POST"])
def api_chain_config() -> Response:
    engine = _engine_from_query() if request.method == "GET" else _engine_from_request()
    if request.method == "GET":
        return jsonify(_svc().get(engine, "/api/chain/config"))
    return jsonify(_svc().post(engine, "/api/chain/config", request.json or {}))


@app.route("/api/chain/acknowledge_error", methods=["POST"])
def api_chain_acknowledge_error() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/chain/acknowledge_error", {}))


@app.route("/api/chains/saved")
def api_chains_saved() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/api/chains/saved"))


@app.route("/api/chains/save", methods=["POST"])
def api_chains_save() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/chains/save", request.json or {}))


@app.route("/api/chains/load", methods=["POST"])
def api_chains_load() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/chains/load", request.json or {}))


@app.route("/api/chains/delete", methods=["POST"])
def api_chains_delete() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/api/chains/delete", request.json or {}))


# ── トレーニング ─────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def api_train_start() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/train/start", request.json or {}))


@app.route("/api/train/status")
def api_train_status() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/train/status"))


@app.route("/api/autotrain/start", methods=["POST"])
def api_autotrain_start() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/autotrain/start", request.json or {}))


@app.route("/api/autotrain/status")
def api_autotrain_status() -> Response:
    engine = _engine_from_query()
    return jsonify(_svc().get(engine, "/autotrain/status"))


@app.route("/api/relabel/start", methods=["POST"])
def api_relabel_start() -> Response:
    engine = _engine_from_request()
    return jsonify(_svc().post(engine, "/relabel/start", request.json or {}))


# ── プロジェクト管理 ─────────────────────────

@app.route("/api/projects")
def api_projects() -> Response:
    """ローカルプロジェクト一覧を返す（バックエンド不要）。"""
    return jsonify({"projects": _list_projects()})


@app.route("/api/projects/create", methods=["POST"])
def api_projects_create() -> Response:
    """
    新規プロジェクトを作成する。

    ローカルディレクトリを作成するのみ。
    実際のアノテーションファイルはUIが管理する。
    """
    from werkzeug.utils import secure_filename as _sf
    data = request.json or {}
    name = _sf(data.get("name", "").strip())
    if not name:
        return jsonify({"error": "プロジェクト名が空です"}), 400

    proj_path = Path(_dirs.projects) / name  # type: ignore[union-attr]
    (proj_path / "images").mkdir(parents=True, exist_ok=True)
    (proj_path / "labels").mkdir(parents=True, exist_ok=True)
    (proj_path / "exports").mkdir(parents=True, exist_ok=True)
    return jsonify({"success": True, "path": str(proj_path)})


@app.route("/api/projects/<project_name>/images")
def api_project_images(project_name: str) -> Response:
    """プロジェクト内の画像ファイル一覧を返す。"""
    from werkzeug.utils import secure_filename as _sf
    safe_name = _sf(project_name)
    if not safe_name:
        return jsonify({"images": []})
    proj_path = Path(_dirs.projects) / safe_name / "images"  # type: ignore[union-attr]
    if not proj_path.exists():
        return jsonify({"images": []})
    images = [f.name for f in proj_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    return jsonify({"images": sorted(images)})


# ── ファイル配信 ─────────────────────────────

@app.route("/api/projects/<project_name>/image/<filename>")
def api_project_image(project_name: str, filename: str) -> Response:
    """アノテーション用の画像ファイルを返す。"""
    from flask import send_from_directory
    from werkzeug.utils import secure_filename as _sf
    safe_project = _sf(project_name)
    safe_file    = _sf(filename)
    if not safe_project or not safe_file:
        return jsonify({"error": "Invalid path parameter"}), 400
    img_dir = Path(_dirs.projects) / safe_project / "images"  # type: ignore[union-attr]
    return send_from_directory(str(img_dir), safe_file)


@app.route("/api/projects/<project_name>/label/<filename>")
def api_project_label_get(project_name: str, filename: str) -> Response:
    """YOLOフォーマットのラベルファイルを返す。存在しない場合は空テキストを返す。"""
    from werkzeug.utils import secure_filename as _sf
    safe_project = _sf(project_name)
    safe_file    = _sf(filename)
    if not safe_project or not safe_file:
        return Response("", mimetype="text/plain")
    label_name = Path(safe_file).stem + ".txt"
    label_path = Path(_dirs.projects) / safe_project / "labels" / label_name  # type: ignore[union-attr]
    if not label_path.exists():
        return Response("", mimetype="text/plain")
    return Response(label_path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route("/api/projects/<project_name>/label/<filename>", methods=["POST"])
def api_project_label_save(project_name: str, filename: str) -> Response:
    """YOLOフォーマットのラベルファイルを保存する。"""
    from werkzeug.utils import secure_filename as _sf
    safe_project = _sf(project_name)
    safe_file    = _sf(filename)
    if not safe_project or not safe_file:
        return jsonify({"error": "Invalid path parameter"}), 400
    label_name = Path(safe_file).stem + ".txt"
    label_path = Path(_dirs.projects) / safe_project / "labels" / label_name  # type: ignore[union-attr]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = request.data.decode("utf-8")
    label_path.write_text(content, encoding="utf-8")
    return jsonify({"success": True})


@app.route("/api/projects/<project_name>/classes")
def api_project_classes(project_name: str) -> Response:
    """プロジェクトのクラス定義を返す。"""
    from werkzeug.utils import secure_filename as _sf
    safe_name = _sf(project_name)
    if not safe_name:
        return jsonify({"classes": []})
    classes_path = Path(_dirs.projects) / safe_name / "classes.txt"  # type: ignore[union-attr]
    if not classes_path.exists():
        return jsonify({"classes": []})
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return jsonify({"classes": classes})


@app.route("/api/projects/<project_name>/classes", methods=["POST"])
def api_project_classes_save(project_name: str) -> Response:
    """クラス定義を保存する。"""
    from werkzeug.utils import secure_filename as _sf
    safe_name = _sf(project_name)
    if not safe_name:
        return jsonify({"error": "Invalid project name"}), 400
    data = request.json or {}
    classes = data.get("classes", [])
    classes_path = Path(_dirs.projects) / safe_name / "classes.txt"  # type: ignore[union-attr]
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text("\n".join(classes), encoding="utf-8")
    return jsonify({"success": True})


@app.route("/api/projects/<project_name>/upload", methods=["POST"])
def api_project_upload(project_name: str) -> Response:
    """
    画像ファイルをプロジェクトにアップロードする。

    複数ファイルの同時アップロードに対応する。
    セキュリティ: ファイル名をサニタイズし、許可拡張子のみ受け付ける。
    """
    from werkzeug.utils import secure_filename

    safe_project = secure_filename(project_name)
    if not safe_project:
        return jsonify({"error": "Invalid project name"}), 400

    # 許可する画像拡張子
    ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    uploaded = []
    img_dir = Path(_dirs.projects) / safe_project / "images"  # type: ignore[union-attr]
    img_dir.mkdir(parents=True, exist_ok=True)

    for file in request.files.getlist("files"):
        original_name = file.filename or ""
        safe_name = secure_filename(original_name)
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED:
            continue
        dest = img_dir / safe_name
        file.save(str(dest))
        uploaded.append(safe_name)

    return jsonify({"uploaded": uploaded, "count": len(uploaded)})


# ═══════════════════════════════════════════════
# ヘルパー関数
# ═══════════════════════════════════════════════

def _engine_from_request() -> EngineType:
    """
    POSTリクエストのJSONボディから engine フィールドを取得する。
    省略時は BILT をデフォルトとする。
    """
    data = request.json or {}
    raw = data.get("engine", EngineType.BILT.value)
    try:
        return EngineType(raw)
    except ValueError:
        return EngineType.BILT


def _engine_from_query() -> EngineType:
    """
    GETリクエストのクエリパラメータから engine を取得する。
    ?engine=yolo のように指定する。
    """
    raw = request.args.get("engine", EngineType.BILT.value)
    try:
        return EngineType(raw)
    except ValueError:
        return EngineType.BILT


def _list_projects() -> list[str]:
    """
    プロジェクトディレクトリ内のサブフォルダ一覧を返す。
    _dirs が未初期化の場合は空リストを返す。
    """
    if _dirs is None:
        return []
    proj_root = Path(_dirs.projects)
    if not proj_root.exists():
        return []
    return sorted(
        d.name for d in proj_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


# ═══════════════════════════════════════════════
# エントリポイント (開発用直接実行)
# 本番ではランチャーから init_server() → run() の順で呼ぶ
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    init_server(AppDirectories())
    app.run(
        host="127.0.0.1",
        port=Ports.UI_WEB,
        debug=True,
        use_reloader=False,  # QWebEngineView との組み合わせで reloader は無効化
    )