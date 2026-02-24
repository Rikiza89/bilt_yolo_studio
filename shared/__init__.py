"""
shared — UIとバックエンドサービス間の共通モジュールパッケージ

このパッケージに含まれるコードは:
  - UIプロセス (PySide6 / Flask)
  - BILTサービスプロセス
  - YOLOサービスプロセス
のすべてからインポートされる。

依存ライブラリ: Python標準ライブラリ + requests のみ。
AGPLライブラリ (bilt, ultralytics) はここでは一切インポートしない。
"""

from shared.contracts import (
    AppDirectories,
    Detection,
    DetectionSettings,
    EngineType,
    Ports,
    TaskType,
    TrainingConfig,
    get_base_dir,
)
from shared.service_client import ServiceClient

__all__ = [
    "AppDirectories",
    "Detection",
    "DetectionSettings",
    "EngineType",
    "Ports",
    "ServiceClient",
    "TaskType",
    "TrainingConfig",
    "get_base_dir",
]
