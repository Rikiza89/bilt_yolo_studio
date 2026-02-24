"""
shared/contracts.py — プロセス間共有データ契約モジュール

このモジュールはUIプロセス・BILTサービス・YOLOサービスの
すべてのレイヤーで使用される共通のデータ型と定数を定義する。

設計上の判断:
  - Pydanticではなく dataclass + Enum を採用。
    理由: サービスプロセスがサードパーティライブラリを
    最小限にとどめる必要があるため。
  - すべての設定値を環境変数で上書き可能にする。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# エンジン種別: UIがどのバックエンドサービスを呼ぶかを示す
# ──────────────────────────────────────────────
class EngineType(str, Enum):
    """検出エンジン種別。サービスURLのルーティングに使用される。"""
    BILT = "bilt"
    YOLO = "yolo"


# ──────────────────────────────────────────────
# タスク種別: データセット・モデルの用途分類
# ──────────────────────────────────────────────
class TaskType(str, Enum):
    """
    検出タスク種別。
    BILTは DETECT のみ対応。
    YOLOは全タスクに対応。
    """
    DETECT  = "detect"
    SEGMENT = "segment"
    OBB     = "obb"
    POSE    = "pose"


# ──────────────────────────────────────────────
# ポート定数: 各サービスのデフォルトポート
# ──────────────────────────────────────────────
class Ports:
    """
    各プロセスのリッスンポート。
    環境変数で上書き可能。

    ライセンス分離の観点:
      UI (PySide6/Flask) は LGPL コード。
      BILT_SERVICE と YOLO_SERVICE は AGPL コード。
      HTTP境界により静的リンクを回避し AGPL 伝播を防ぐ。
    """
    UI_WEB: int        = int(os.getenv("STUDIO_UI_PORT",    "5100"))
    BILT_SERVICE: int  = int(os.getenv("BILT_SERVICE_PORT", "5101"))
    YOLO_SERVICE: int  = int(os.getenv("YOLO_SERVICE_PORT", "5102"))


# ──────────────────────────────────────────────
# ベースディレクトリ解決
# ──────────────────────────────────────────────
def get_base_dir() -> str:
    """
    アプリケーションのベースディレクトリを返す。
    PyInstaller でビルドされた場合は実行ファイルの親ディレクトリ、
    通常実行の場合はこのファイルの2階層上を使用する。
    """
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────
# アプリケーション全体のディレクトリ設定
# ──────────────────────────────────────────────
@dataclass
class AppDirectories:
    """
    アプリケーションが使用するすべてのディレクトリパス。
    インスタンス生成時に存在しないディレクトリを作成する。
    """
    base: str = field(default_factory=get_base_dir)

    @property
    def models(self) -> str:
        return os.path.join(self.base, "models")

    @property
    def projects(self) -> str:
        return os.path.join(self.base, "projects")

    @property
    def chains(self) -> str:
        return os.path.join(self.base, "chains")

    @property
    def datasets(self) -> str:
        return os.path.join(self.base, "datasets")

    @property
    def saved_images(self) -> str:
        return os.path.join(self.base, "saved_images")

    def ensure_all(self) -> None:
        """全ディレクトリを作成する（存在しない場合のみ）。"""
        for path in [
            self.models, self.projects, self.chains,
            self.datasets, self.saved_images,
        ]:
            os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# 検出結果の共通データ構造
# ──────────────────────────────────────────────
@dataclass
class Detection:
    """
    単一オブジェクトの検出結果。
    BILTとYOLOの両エンジンが同じ形式で返す。

    bbox: [x1, y1, x2, y2] ピクセル絶対座標
    points: セグメント/OBB/ポーズ用ポイント列（オプション）
    """
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[float]                    # [x1, y1, x2, y2]
    task_type:  TaskType = TaskType.DETECT
    points:     Optional[List[Dict[str, float]]] = None   # segment/obb/pose
    keypoints:  Optional[List[Dict[str, Any]]]   = None   # pose のみ

    def to_dict(self) -> Dict[str, Any]:
        """JSON シリアライズ可能な辞書に変換する。"""
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox":       self.bbox,
            "task_type":  self.task_type.value,
            "points":     self.points,
            "keypoints":  self.keypoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """辞書からインスタンスを復元する。"""
        return cls(
            class_id   = data["class_id"],
            class_name = data["class_name"],
            confidence = data["confidence"],
            bbox       = data["bbox"],
            task_type  = TaskType(data.get("task_type", "detect")),
            points     = data.get("points"),
            keypoints  = data.get("keypoints"),
        )


# ──────────────────────────────────────────────
# トレーニング設定の共通データ構造
# ──────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """
    モデルトレーニングの共通設定。
    BILTとYOLOで共有されるフィールドを定義。
    エンジン固有のパラメータは extra_kwargs に格納する。
    """
    engine:           EngineType
    task_type:        TaskType
    data_yaml:        str
    project_path:     str
    model_name:       str              = "scratch"
    custom_save_name: Optional[str]    = None
    epochs:           int              = 100
    batch_size:       int              = 8
    img_size:         int              = 640
    learning_rate:    float            = 0.001
    device:           str              = "cpu"
    extra_kwargs:     Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine":           self.engine.value,
            "task_type":        self.task_type.value,
            "data_yaml":        self.data_yaml,
            "project_path":     self.project_path,
            "model_name":       self.model_name,
            "custom_save_name": self.custom_save_name,
            "epochs":           self.epochs,
            "batch_size":       self.batch_size,
            "img_size":         self.img_size,
            "learning_rate":    self.learning_rate,
            "device":           self.device,
            **self.extra_kwargs,
        }


# ──────────────────────────────────────────────
# 検出設定の共通データ構造
# ──────────────────────────────────────────────
@dataclass
class DetectionSettings:
    """
    リアルタイム検出の設定。
    UIとサービス間でやり取りされる。
    """
    engine:          EngineType    = EngineType.BILT
    conf:            float         = 0.60
    iou:             float         = 0.10
    max_det:         int           = 10
    classes:         Optional[List[int]] = None
    counter_mode:    bool          = False
    save_images:     bool          = False
    dataset_capture: bool          = False
    project_folder:  str           = ""
    chain_mode:      bool          = False
    chain_steps:     List[Dict]    = field(default_factory=list)
    chain_timeout:   float         = 50.0
    chain_pause_time: float        = 10.0
    chain_auto_advance: bool       = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine":            self.engine.value,
            "conf":              self.conf,
            "iou":               self.iou,
            "max_det":           self.max_det,
            "classes":           self.classes,
            "counter_mode":      self.counter_mode,
            "save_images":       self.save_images,
            "dataset_capture":   self.dataset_capture,
            "project_folder":    self.project_folder,
            "chain_mode":        self.chain_mode,
            "chain_steps":       self.chain_steps,
            "chain_timeout":     self.chain_timeout,
            "chain_pause_time":  self.chain_pause_time,
            "chain_auto_advance": self.chain_auto_advance,
        }
