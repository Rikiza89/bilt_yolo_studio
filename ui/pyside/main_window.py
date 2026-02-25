"""
ui/pyside/main_window.py — PySide6 メインウィンドウ

QWebEngineView を使ってローカルの Flask Web UI を表示するネイティブデスクトップシェル。
このモジュールは LGPL 対象の PySide6 のみを使用し、
BILT(AGPL) や Ultralytics(AGPL) を一切 import しない。

設計上の判断:
  - QWebEngineView を選択した理由:
    既存の Flask+HTML UI 資産をそのまま再利用できる。
    Electron と異なり Python ネイティブ統合が容易。
    将来的に WebSocket 経由のライブ更新も可能。
  - システムトレイ: バックグラウンド動作をサポートする。
    ウィンドウを閉じてもサービスはバックグラウンドで継続実行する。
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from PySide6.QtCore import (
    QSize, QTimer, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QCloseEvent, QIcon,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMenuBar,
    QMessageBox, QSplashScreen, QStatusBar,
    QSystemTrayIcon, QMenu, QLabel, QWidget,
    QHBoxLayout, QPushButton,
)
from PySide6.QtWebEngineCore import QWebEnginePage

from shared.contracts import Ports

logger = logging.getLogger(__name__)


class _WebPage(QWebEnginePage):
    """
    カスタム QWebEnginePage。

    console.log() の出力を Python ロガーに転送する。
    開発時のデバッグに有用。
    """

    def javaScriptConsoleMessage(
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        lineNumber: int,
        sourceID: str,
    ) -> None:
        """JavaScript コンソールメッセージを Python ロガーに転送する。"""
        log = {
            QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:    logger.info,
            QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel: logger.warning,
            QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:   logger.error,
        }.get(level, logger.debug)
        log("[JS %s:%d] %s", sourceID, lineNumber, message)


class StudioMainWindow(QMainWindow):
    """
    BILT+YOLO Studio のメインウィンドウ。

    Flask Web UI を QWebEngineView で表示する。
    システムトレイアイコン経由でバックグラウンド動作をサポートする。

    設計上の判断:
      - ウィンドウタイトルバーは最小限に抑え、
        ナビゲーションは Web UI 側に委ねる。
      - リロードボタンとサービス状態表示のみをネイティブ側に持つ。
    """

    # バックエンドサービスの起動完了を通知するシグナル
    services_ready = Signal()

    def __init__(
        self,
        ui_port: int = Ports.UI_WEB,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        メインウィンドウを初期化する。

        Args:
            ui_port: Flask UIサーバーがリッスンするポート番号。
            parent:  親ウィジェット。通常は None。
        """
        super().__init__(parent)
        self._ui_url = f"http://127.0.0.1:{ui_port}"
        self._tray: Optional[QSystemTrayIcon] = None

        self._setup_window()
        self._setup_menu()
        self._setup_webview()
        self._setup_statusbar()
        self._setup_tray()

        # UIサーバーが立ち上がるまで待機してからロード
        # 最大20秒間500ms間隔でリトライする
        self._load_attempts = 0
        self._load_timer = QTimer(self)
        self._load_timer.timeout.connect(self._try_load)
        self._load_timer.start(500)

        logger.info("メインウィンドウ初期化完了: %s", self._ui_url)

    # ══════════════════════════════════════════
    # セットアップメソッド
    # ══════════════════════════════════════════

    def _setup_window(self) -> None:
        """ウィンドウの基本プロパティを設定する。"""
        self.setWindowTitle("BILT+YOLO Studio")
        self.resize(1400, 900)
        self.setMinimumSize(QSize(1000, 600))

    def _setup_menu(self) -> None:
        """メニューバーを設定する。"""
        menubar = self.menuBar()

        # ── ファイルメニュー ───────────────
        file_menu = menubar.addMenu("ファイル (&F)")

        reload_action = QAction("ページを再読み込み (&R)", self)
        reload_action.setShortcut("F5")
        reload_action.triggered.connect(self._reload_page)
        file_menu.addAction(reload_action)

        file_menu.addSeparator()

        quit_action = QAction("終了 (&Q)", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self._quit_app)
        file_menu.addAction(quit_action)

        # ── 表示メニュー ───────────────────
        view_menu = menubar.addMenu("表示 (&V)")

        devtools_action = QAction("開発者ツール (&D)", self)
        devtools_action.setShortcut("F12")
        devtools_action.triggered.connect(self._open_devtools)
        view_menu.addAction(devtools_action)

    def _setup_webview(self) -> None:
        """QWebEngineView を設定してセントラルウィジェットとして配置する。"""
        self._webview = QWebEngineView(self)
        self._page    = _WebPage(self._webview)
        self._webview.setPage(self._page)

        # ローカルホストへのアクセスを許可する設定
        # CORS/SameSite ポリシーをローカル開発向けに緩和する
        settings = self._webview.settings()
        try:
            from PySide6.QtWebEngineCore import QWebEngineSettings
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
            )
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptEnabled, True
            )
        except Exception:
            pass  # 設定が利用できない環境では無視する

        self.setCentralWidget(self._webview)

    def _setup_statusbar(self) -> None:
        """ステータスバーを設定する。"""
        sb = self.statusBar()

        self._status_label = QLabel("サービス起動中...")
        sb.addWidget(self._status_label)

        # サービス状態インジケーター
        self._bilt_indicator = QLabel("● BILT")
        self._bilt_indicator.setStyleSheet("color: gray; padding: 0 8px;")
        sb.addPermanentWidget(self._bilt_indicator)

        self._yolo_indicator = QLabel("● YOLO")
        self._yolo_indicator.setStyleSheet("color: gray; padding: 0 8px;")
        sb.addPermanentWidget(self._yolo_indicator)

        # ステータス更新タイマー (5秒間隔)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_service_status)
        self._status_timer.start(5000)

    def _setup_tray(self) -> None:
        """
        システムトレイアイコンを設定する。

        システムトレイが利用できない環境では設定をスキップする。
        トレイアイコン経由でウィンドウの表示/非表示と終了操作を提供する。
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("システムトレイが利用不可のため、トレイ機能をスキップ")
            return

        self._tray = QSystemTrayIcon(self)

        # アイコンがない場合は標準アイコンを使用する
        app = QApplication.instance()
        icon = app.windowIcon() if app else QIcon()
        self._tray.setIcon(icon)
        self._tray.setToolTip("BILT+YOLO Studio")

        tray_menu = QMenu()
        show_action = QAction("ウィンドウを表示", self)
        show_action.triggered.connect(self._show_window)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()

        quit_action = QAction("終了", self)
        quit_action.triggered.connect(self._quit_app)
        tray_menu.addAction(quit_action)

        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    # ══════════════════════════════════════════
    # UIサーバー待機ロジック
    # ══════════════════════════════════════════

    @Slot()
    def _try_load(self) -> None:
        """
        UIサーバーへの接続を試みる。

        サーバーが応答するまで最大 40 回 (20秒) リトライする。
        成功したらページをロードし、タイマーを停止する。
        失敗が続く場合はエラーダイアログを表示する。
        """
        import urllib.request
        self._load_attempts += 1

        try:
            # /api/health への簡易リクエストでサーバー起動を確認する
            with urllib.request.urlopen(
                f"{self._ui_url}/api/health", timeout=1
            ) as resp:
                if resp.status == 200:
                    self._load_timer.stop()
                    self._webview.setUrl(QUrl(f"{self._ui_url}/detection"))
                    self._status_label.setText("サービス稼働中")
                    logger.info("UIサーバー接続成功 (%d回目)", self._load_attempts)
                    return
        except Exception:
            pass  # まだ起動中の場合はリトライ

        if self._load_attempts >= 40:
            # 20秒経過しても接続できない場合
            self._load_timer.stop()
            self._status_label.setText("サービス起動タイムアウト")
            QMessageBox.critical(
                self,
                "接続エラー",
                "バックエンドサービスに接続できませんでした。\n"
                "ランチャーからアプリを再起動してください。",
            )
            logger.error("UIサーバー接続タイムアウト (%d回試行)", self._load_attempts)

    # ══════════════════════════════════════════
    # ステータス更新
    # ══════════════════════════════════════════

    @Slot()
    def _update_service_status(self) -> None:
        """
        バックエンドサービスの状態を定期的に確認してステータスバーを更新する。

        UIスレッドで実行されるため、重いI/Oは避ける。
        タイムアウトを短く設定してUIのブロックを防ぐ。
        """
        import urllib.request
        import json

        try:
            with urllib.request.urlopen(
                f"{self._ui_url}/api/health", timeout=2
            ) as resp:
                data = json.loads(resp.read())
                # BUG FIX: Health endpoint returns dicts like {"status": "healthy"},
                # not booleans. Check the "status" key inside each dict.
                bilt_data = data.get("bilt", {})
                yolo_data = data.get("yolo", {})
                bilt_ok = isinstance(bilt_data, dict) and bilt_data.get("status") in ("healthy", "ok")
                yolo_ok = isinstance(yolo_data, dict) and yolo_data.get("status") in ("healthy", "ok")
        except Exception:
            bilt_ok = yolo_ok = False

        def _color(ok: bool) -> str:
            return "green" if ok else "red"

        self._bilt_indicator.setStyleSheet(
            f"color: {_color(bilt_ok)}; padding: 0 8px;"
        )
        self._yolo_indicator.setStyleSheet(
            f"color: {_color(yolo_ok)}; padding: 0 8px;"
        )

    # ══════════════════════════════════════════
    # アクションハンドラ
    # ══════════════════════════════════════════

    @Slot()
    def _reload_page(self) -> None:
        """現在のページを再読み込みする。"""
        self._webview.reload()

    @Slot()
    def _open_devtools(self) -> None:
        """
        開発者ツールウィンドウを開く。

        開発・デバッグ時に使用する。
        本番環境ではメニューから非表示にすることを推奨する。
        """
        devtools = QWebEngineView()
        self._page.setDevToolsPage(devtools.page())
        devtools.resize(1200, 800)
        devtools.setWindowTitle("DevTools — BILT+YOLO Studio")
        devtools.show()
        # 参照を保持してガベージコレクションを防ぐ
        self._devtools_window = devtools

    @Slot()
    def _show_window(self) -> None:
        """ウィンドウを前面に表示する。"""
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @Slot()
    def _quit_app(self) -> None:
        """
        アプリケーションを終了する。

        QApplication.quit() を呼び出すことでランチャーのクリーンアップが
        実行され、バックエンドサービスも停止される。
        """
        logger.info("ユーザーによるアプリケーション終了要求")
        QApplication.quit()

    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """
        トレイアイコンのアクティベーションイベントを処理する。

        ダブルクリックでウィンドウを表示する。
        """
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_window()

    # ══════════════════════════════════════════
    # ウィンドウ閉じるイベント
    # ══════════════════════════════════════════

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        ウィンドウを閉じる際の動作を制御する。

        システムトレイが利用可能な場合はバックグラウンドに隠れる。
        トレイが利用不可の場合は終了確認ダイアログを表示する。

        Args:
            event: ウィンドウクローズイベント。
        """
        if self._tray and self._tray.isVisible():
            # トレイに最小化してバックグラウンド動作を継続する
            self.hide()
            self._tray.showMessage(
                "BILT+YOLO Studio",
                "バックグラウンドで動作中です。\nトレイアイコンをダブルクリックで再表示。",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            # トレイなしの場合は終了確認を行う
            reply = QMessageBox.question(
                self,
                "終了確認",
                "アプリケーションを終了しますか？\n"
                "バックエンドサービスも停止されます。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
                QApplication.quit()
            else:
                event.ignore()
