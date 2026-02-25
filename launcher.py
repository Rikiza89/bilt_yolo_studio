"""
launcher.py — BILT+YOLO Studio アプリケーションランチャー

このスクリプトはアプリケーション全体のエントリポイントとして機能する。
以下の3プロセスを管理する:

  1. BILTサービス  (AGPL-3.0) — services/bilt_service/app.py
  2. YOLOサービス  (AGPL-3.0) — services/yolo_service/app.py
  3. UIサーバー    (LGPL)     — ui/web/server.py (このプロセス内で起動)
  4. PySide6シェル (LGPL)     — ui/pyside/main_window.py (このプロセス内)

ライセンス分離戦略:
  - AGPL サービスは subprocess として起動する。
    これにより、PySide6 (LGPL) のUIプロセスと静的リンクが発生しない。
  - UIプロセスはHTTP経由でのみ AGPL コードと通信する。
  - LGPL の要件: PySide6 のソースコードへのリンクを提供するだけでよく、
    このアプリケーションのソースコードを公開する義務はない。

起動順序:
  1. BILT/YOLOサービスのサブプロセスを起動する。
  2. FlaskサーバーをバックグラウンドスレッドとStart。
  3. PySide6 のQApplicationを起動する。
  4. メインウィンドウが Flask に接続を確認してから表示する。
  5. QApplication 終了時に全サービスを停止する。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

# ──────────────────────────────────────────────
# プロジェクトルートを sys.path に追加する
# このランチャーはプロジェクトルートから実行されることを前提とする
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.contracts import AppDirectories, Ports

# ──────────────────────────────────────────────
# ロギング設定
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launcher")


# ═══════════════════════════════════════════════
# サービスプロセス管理
# ═══════════════════════════════════════════════

class ServiceProcess:
    """
    バックエンドサービスのサブプロセスを管理するクラス。

    起動・停止・ヘルスチェックを提供する。
    プロセスが予期せず停止した場合の再起動ロジックは
    将来の拡張ポイントとして設計されている。
    """

    def __init__(
        self,
        name: str,
        module_path: str,
        port: int,
        health_endpoint: str = "/health",
    ) -> None:
        """
        Args:
            name:            サービスの識別名 (ログ用)。
            module_path:     Python スクリプトのパス (プロジェクトルートからの相対パス)。
            port:            サービスのリッスンポート。
            health_endpoint: ヘルスチェック用エンドポイントパス。
        """
        self._name     = name
        self._module   = module_path
        self._port     = port
        self._health   = health_endpoint
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """
        サービスをサブプロセスとして起動する。

        Returns:
            True: 起動に成功した場合。
            False: 既に起動済み、またはスクリプトが存在しない場合。
        """
        if self._process and self._process.poll() is None:
            logger.info("[%s] 既に起動済み", self._name)
            return True

        script = _PROJECT_ROOT / self._module
        if not script.exists():
            logger.error("[%s] スクリプトが見つかりません: %s", self._name, script)
            return False

        # 環境変数を引き継いでサービスを起動する
        env = os.environ.copy()

        self._process = subprocess.Popen(
            [sys.executable, str(script)],
            env=env,
            # 標準出力/エラーをパイプしてランチャーがログを集約する
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # プロセスグループを作成してシグナル伝達を制御する
            start_new_session=True,
        )

        # ログ転送スレッドを起動する
        threading.Thread(
            target=self._pipe_logs,
            daemon=True,
            name=f"log-{self._name}",
        ).start()

        logger.info("[%s] サービス起動 (PID: %d)", self._name, self._process.pid)
        return True

    def wait_ready(self, timeout: float = 30.0, interval: float = 0.5) -> bool:
        """
        サービスがHTTPリクエストに応答するまで待機する。

        Args:
            timeout:  最大待機時間 (秒)。
            interval: ポーリング間隔 (秒)。

        Returns:
            True: タイムアウト前に応答が確認できた場合。
            False: タイムアウトした場合。
        """
        import urllib.request

        url      = f"http://127.0.0.1:{self._port}{self._health}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            # プロセスが既に終了している場合は待機を中断する
            if self._process and self._process.poll() is not None:
                logger.error("[%s] プロセスが予期せず終了", self._name)
                return False
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        logger.info("[%s] 起動確認 OK", self._name)
                        return True
            except Exception:
                pass
            time.sleep(interval)

        logger.error("[%s] 起動タイムアウト (%ds)", self._name, timeout)
        return False

    def stop(self) -> None:
        """
        サービスプロセスを停止する。

        SIGTERM を送信し、3秒以内に停止しない場合は SIGKILL を送る。
        """
        if not self._process:
            return
        if self._process.poll() is not None:
            return  # 既に停止済み

        logger.info("[%s] 停止要求 (PID: %d)", self._name, self._process.pid)
        try:
            self._process.terminate()  # SIGTERM
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logger.warning("[%s] 強制終了 (SIGKILL)", self._name)
            self._process.kill()
        except Exception as e:
            logger.error("[%s] 停止エラー: %s", self._name, e)

    def is_alive(self) -> bool:
        """プロセスが生存しているかどうかを返す。"""
        return self._process is not None and self._process.poll() is None

    def _pipe_logs(self) -> None:
        """
        サービスの stdout をランチャーのロガーに転送するスレッド関数。

        ライブのサービスログをランチャーコンソールに集約する。
        """
        if self._process is None or self._process.stdout is None:
            return
        svc_logger = logging.getLogger(f"svc.{self._name}")
        for raw_line in self._process.stdout:
            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                svc_logger.info(line)
            except Exception:
                pass


# ═══════════════════════════════════════════════
# Flask UIサーバー (バックグラウンドスレッド)
# ═══════════════════════════════════════════════

def _start_flask_server(dirs: AppDirectories, port: int) -> threading.Thread:
    """
    Flask UIサーバーをバックグラウンドスレッドで起動する。

    PySide6 のメインループと同一プロセス内で動作するため、
    スレッドセーフなWerkzeugサーバーを使用する。

    Args:
        dirs: アプリケーションディレクトリ設定。
        port: UIサーバーのリッスンポート。

    Returns:
        起動したスレッド。
    """
    from ui.web.server import app, init_server

    def _run() -> None:
        init_server(dirs)
        # use_reloader=False: 別スレッドでの起動に必須
        # threaded=True: 複数の同時リクエストを処理する
        app.run(
            host="127.0.0.1",
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    t = threading.Thread(target=_run, daemon=True, name="flask-ui")
    t.start()
    logger.info("Flaskサーバー起動 (port: %d)", port)
    return t


# ═══════════════════════════════════════════════
# メインエントリポイント
# ═══════════════════════════════════════════════

def main() -> int:
    """
    アプリケーション全体のエントリポイント。

    全プロセスを起動し、PySide6 のイベントループを実行する。
    イベントループ終了後に全サービスを停止してプロセスを終了する。

    Returns:
        終了コード。0: 正常終了、1: エラー終了。
    """
    dirs = AppDirectories()
    dirs.ensure_all()

    logger.info("═══ BILT+YOLO Studio 起動 ═══")
    logger.info("プロジェクトルート: %s", _PROJECT_ROOT)

    # ── 1. バックエンドサービスを起動する ─────────
    services: List[ServiceProcess] = [
        ServiceProcess(
            name="BILT",
            module_path="services/bilt_service/app.py",
            port=Ports.BILT_SERVICE,
        ),
        ServiceProcess(
            name="YOLO",
            module_path="services/yolo_service/app.py",
            port=Ports.YOLO_SERVICE,
        ),
    ]

    # BUG FIX: Populate _services_global so the signal handler can
    # actually clean up services on SIGINT/SIGTERM.
    global _services_global
    _services_global = services

    for svc in services:
        if not svc.start():
            logger.error("サービス起動失敗: %s", svc._name)
            # サービスが起動できなくても UI は立ち上げる
            # (サービス未接続状態でもUIは動作する設計)

    # ── 2. サービスの準備を並列待機する ──────────
    # 最大 30 秒待機する（タイムアウトしても続行する）
    logger.info("バックエンドサービスの起動を待機中...")
    threads = [
        threading.Thread(target=svc.wait_ready, daemon=True)
        for svc in services
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=30)
    logger.info("バックエンドサービス待機完了")

    # ── 3. Flask UIサーバーを起動する ─────────────
    _start_flask_server(dirs, port=Ports.UI_WEB)

    # Flask が起動するまで少し待機する
    time.sleep(1.0)

    # ── 4. PySide6 アプリケーションを起動する ─────
    # QApplication は必ずメインスレッドで初期化する
    try:
        from PySide6.QtWidgets import QApplication
        from ui.pyside.main_window import StudioMainWindow
    except ImportError as e:
        logger.error("PySide6 が見つかりません: %s", e)
        logger.error("pip install PySide6 PyQt6-WebEngine を実行してください")
        _cleanup(services)
        return 1

    # Platform-specific Qt backend selection (must be set before QApplication)
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    elif sys.platform == "linux":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    app = QApplication(sys.argv)
    app.setApplicationName("BILT+YOLO Studio")
    app.setOrganizationName("Studio")

    # ウィンドウが閉じられてもトレイがある場合はプロセスを継続する
    # app.setQuitOnLastWindowClosed(False) はメインウィンドウ側で管理

    window = StudioMainWindow(ui_port=Ports.UI_WEB)
    window.show()

    logger.info("PySide6 イベントループ開始")
    exit_code = app.exec()
    logger.info("PySide6 イベントループ終了 (code: %d)", exit_code)

    # ── 5. クリーンアップ ─────────────────────────
    _cleanup(services)

    return exit_code


def _cleanup(services: List[ServiceProcess]) -> None:
    """
    全サービスプロセスを停止する。

    シグナルハンドラやアプリ終了時に呼び出される。
    各サービスを順次停止し、エラーがあってもすべてのプロセスを
    停止しようとする（フェイルセーフ設計）。

    Args:
        services: 停止対象のサービスリスト。
    """
    logger.info("クリーンアップ開始...")
    for svc in services:
        try:
            svc.stop()
        except Exception as e:
            logger.error("サービス停止エラー [%s]: %s", svc._name, e)
    logger.info("クリーンアップ完了")


# ═══════════════════════════════════════════════
# シグナルハンドラ (Unix環境)
# ═══════════════════════════════════════════════

_services_global: List[ServiceProcess] = []


def _signal_handler(signum: int, frame: object) -> None:
    """
    SIGINT/SIGTERM を受信したときのハンドラ。

    Ctrl+C などで終了する場合にサービスを適切に停止する。

    Args:
        signum: シグナル番号。
        frame:  現在のスタックフレーム (未使用)。
    """
    logger.info("シグナル %d を受信。シャットダウン中...", signum)
    _cleanup(_services_global)
    sys.exit(0)


if __name__ == "__main__":
    # シグナルハンドラを登録する (Unix/macOS のみ)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    sys.exit(main())
