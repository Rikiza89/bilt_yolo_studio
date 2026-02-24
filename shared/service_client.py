"""
shared/service_client.py — バックエンドサービスへの統合HTTPクライアント

このクライアントはUIプロセスからBILTサービスとYOLOサービスの
両方と通信するための統一インターフェースを提供する。

設計上の判断:
  - UIコードはこのクライアントのみを使用し、
    サービスURLを直接参照しない。
  - エンジン種別 (EngineType) に基づき自動的にルーティングする。
  - 接続エラーは例外を送出せず error キーを持つ辞書を返す。
    理由: UIがサービス停止中でもクラッシュしないようにするため。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from shared.contracts import EngineType, Ports

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    BILTサービス・YOLOサービスへの統合クライアント。

    使用例:
        client = ServiceClient()
        result = client.health_check(EngineType.BILT)
        models = client.get_models(EngineType.YOLO)
    """

    def __init__(
        self,
        bilt_url: str = f"http://127.0.0.1:{Ports.BILT_SERVICE}",
        yolo_url: str = f"http://127.0.0.1:{Ports.YOLO_SERVICE}",
        timeout:  int = 30,
    ) -> None:
        """
        クライアントを初期化する。

        Args:
            bilt_url: BILTサービスのベースURL。
            yolo_url: YOLOサービスのベースURL。
            timeout:  HTTPリクエストのタイムアウト秒数。
        """
        self._urls = {
            EngineType.BILT: bilt_url.rstrip("/"),
            EngineType.YOLO: yolo_url.rstrip("/"),
        }
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ──────────────────────────────────────────
    # 内部ルーティングヘルパー
    # ──────────────────────────────────────────
    def _url(self, engine: EngineType, endpoint: str) -> str:
        """エンジン種別とエンドポイントパスからURLを構築する。"""
        return f"{self._urls[engine]}/{endpoint.lstrip('/')}"

    def _get(
        self,
        engine:   EngineType,
        endpoint: str,
        timeout:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        GETリクエストを送信する。
        接続エラー時は {'error': str} を返し、例外は送出しない。
        """
        url = self._url(engine, endpoint)
        try:
            resp = self._session.get(url, timeout=timeout or self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"{engine.value} サービスに接続できません。起動していますか？"}
        except requests.exceptions.Timeout:
            return {"error": f"{engine.value} サービスへのリクエストがタイムアウトしました"}
        except requests.exceptions.HTTPError as exc:
            try:
                return exc.response.json()
            except Exception:
                return {"error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.exception("予期しないエラー: GET %s", url)
            return {"error": str(exc)}

    def _post(
        self,
        engine:   EngineType,
        endpoint: str,
        data:     Optional[Dict[str, Any]] = None,
        timeout:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        POSTリクエストを送信する。
        接続エラー時は {'error': str} を返し、例外は送出しない。
        """
        url = self._url(engine, endpoint)
        try:
            resp = self._session.post(
                url,
                json=data or {},
                timeout=timeout or self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"{engine.value} サービスに接続できません。起動していますか？"}
        except requests.exceptions.Timeout:
            return {"error": f"{engine.value} サービスへのリクエストがタイムアウトしました"}
        except requests.exceptions.HTTPError as exc:
            try:
                return exc.response.json()
            except Exception:
                return {"error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.exception("予期しないエラー: POST %s", url)
            return {"error": str(exc)}

    # ──────────────────────────────────────────
    # ヘルスチェック
    # ──────────────────────────────────────────
    def health_check(self, engine: EngineType) -> Dict[str, Any]:
        """サービスの死活確認を行う。"""
        return self._get(engine, "/health", timeout=5)

    def is_available(self, engine: EngineType) -> bool:
        """サービスが応答可能か真偽値で返す。"""
        result = self.health_check(engine)
        return result.get("status") in ("healthy", "ok")

    # ──────────────────────────────────────────
    # モデル管理
    # ──────────────────────────────────────────
    def get_models(self, engine: EngineType, task_type: str = "detect") -> Dict[str, Any]:
        """利用可能なモデル一覧を取得する。"""
        return self._post(engine, "/models/available", {"task_type": task_type})

    def load_model(self, engine: EngineType, model_name: str) -> Dict[str, Any]:
        """指定モデルをサービスにロードする。"""
        return self._post(engine, "/api/model/load", {"model_name": model_name})

    def get_model_info(self, engine: EngineType) -> Dict[str, Any]:
        """現在ロードされているモデルの情報を取得する。"""
        return self._get(engine, "/api/model/info")

    # ──────────────────────────────────────────
    # カメラ管理
    # ──────────────────────────────────────────
    def get_cameras(self, engine: EngineType) -> Dict[str, Any]:
        """利用可能なカメラ一覧を取得する。"""
        return self._get(engine, "/api/cameras")

    def select_camera(self, engine: EngineType, camera_index: int) -> Dict[str, Any]:
        """指定インデックスのカメラを選択する。"""
        return self._post(engine, "/api/camera/select", {"camera_index": camera_index})

    def set_camera_resolution(
        self, engine: EngineType, width: int, height: int
    ) -> Dict[str, Any]:
        """カメラ解像度を設定する。"""
        return self._post(engine, "/api/camera/resolution", {"width": width, "height": height})

    # ──────────────────────────────────────────
    # 検出制御
    # ──────────────────────────────────────────
    def start_detection(self, engine: EngineType) -> Dict[str, Any]:
        """リアルタイム検出を開始する。"""
        return self._post(engine, "/api/detection/start")

    def stop_detection(self, engine: EngineType) -> Dict[str, Any]:
        """リアルタイム検出を停止する。"""
        return self._post(engine, "/api/detection/stop")

    def get_detection_settings(self, engine: EngineType) -> Dict[str, Any]:
        """現在の検出設定を取得する。"""
        return self._get(engine, "/api/detection/settings")

    def update_detection_settings(
        self, engine: EngineType, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """検出設定を更新する。"""
        return self._post(engine, "/api/detection/settings", settings)

    def get_detection_stats(self, engine: EngineType) -> Dict[str, Any]:
        """検出統計を取得する。"""
        return self._get(engine, "/api/detection/stats")

    def get_latest_frame(self, engine: EngineType) -> Optional[bytes]:
        """
        最新フレームをJPEGバイト列で取得する。
        取得失敗時は None を返す。
        """
        url = self._url(engine, "/api/frame/latest")
        try:
            resp = self._session.get(url, timeout=5)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.debug("フレーム取得失敗: %s", exc)
            return None

    # ──────────────────────────────────────────
    # カウンター管理
    # ──────────────────────────────────────────
    def get_counters(self, engine: EngineType) -> Dict[str, Any]:
        """オブジェクトカウンターを取得する。"""
        return self._get(engine, "/api/counters")

    def reset_counters(self, engine: EngineType) -> Dict[str, Any]:
        """オブジェクトカウンターをリセットする。"""
        return self._post(engine, "/api/counters/reset")

    # ──────────────────────────────────────────
    # チェーン検出
    # ──────────────────────────────────────────
    def get_chain_status(self, engine: EngineType) -> Dict[str, Any]:
        """チェーン検出の現在状態を取得する。"""
        return self._get(engine, "/api/chain/status")

    def chain_control(self, engine: EngineType, action: str) -> Dict[str, Any]:
        """チェーン検出を制御する (start/stop/reset)。"""
        return self._post(engine, "/api/chain/control", {"action": action})

    def get_chain_config(self, engine: EngineType) -> Dict[str, Any]:
        """チェーン設定を取得する。"""
        return self._get(engine, "/api/chain/config")

    def update_chain_config(
        self, engine: EngineType, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """チェーン設定を更新する。"""
        return self._post(engine, "/api/chain/config", config)

    def acknowledge_chain_error(self, engine: EngineType) -> Dict[str, Any]:
        """チェーン検出のエラーを確認し次ステップへ進む。"""
        return self._post(engine, "/api/chain/acknowledge_error")

    # ──────────────────────────────────────────
    # チェーン保存・読み込み
    # ──────────────────────────────────────────
    def get_saved_chains(self, engine: EngineType) -> Dict[str, Any]:
        """保存済みチェーン一覧を取得する。"""
        return self._get(engine, "/api/chains/saved")

    def save_chain(
        self, engine: EngineType, chain_name: str, model_name: str
    ) -> Dict[str, Any]:
        """現在のチェーン設定を保存する。"""
        return self._post(engine, "/api/chains/save", {
            "chain_name": chain_name,
            "model_name": model_name,
        })

    def load_chain(self, engine: EngineType, chain_name: str) -> Dict[str, Any]:
        """保存済みチェーンを読み込む。"""
        return self._post(engine, "/api/chains/load", {"chain_name": chain_name})

    def delete_chain(self, engine: EngineType, chain_name: str) -> Dict[str, Any]:
        """保存済みチェーンを削除する。"""
        return self._post(engine, "/api/chains/delete", {"chain_name": chain_name})

    # ──────────────────────────────────────────
    # トレーニング管理
    # ──────────────────────────────────────────
    def start_training(
        self, engine: EngineType, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """モデルトレーニングを開始する。"""
        return self._post(engine, "/train/start", config, timeout=120)

    def get_training_status(self, engine: EngineType) -> Dict[str, Any]:
        """トレーニング状態を取得する。"""
        return self._get(engine, "/train/status")

    def start_autotrain(
        self, engine: EngineType, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """オートトレーニングを開始する。"""
        return self._post(engine, "/autotrain/start", config, timeout=120)

    def get_autotrain_status(self, engine: EngineType) -> Dict[str, Any]:
        """オートトレーニング状態を取得する。"""
        return self._get(engine, "/autotrain/status")

    def start_relabel(
        self, engine: EngineType, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """リラベリングを開始する。"""
        return self._post(engine, "/relabel/start", config, timeout=300)

    # ──────────────────────────────────────────
    # プロジェクト管理
    # ──────────────────────────────────────────
    def get_projects(self, engine: EngineType) -> Dict[str, Any]:
        """プロジェクト一覧を取得する。"""
        return self._get(engine, "/api/projects")

    def create_project(
        self,
        engine:       EngineType,
        project_name: str,
        description:  str = "",
        classes:      Optional[list] = None,
    ) -> Dict[str, Any]:
        """新規プロジェクトを作成する。"""
        return self._post(engine, "/api/projects/create", {
            "project_name": project_name,
            "description":  description,
            "classes":      classes or [],
        })

    def close(self) -> None:
        """セッションを閉じる（コンテキストマネージャ対応）。"""
        self._session.close()

    def __enter__(self) -> "ServiceClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
