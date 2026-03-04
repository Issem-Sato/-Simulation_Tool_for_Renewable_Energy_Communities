from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path as _Path
import os as _os

@dataclass
class AppConfig:
    data_dir: _Path = _Path(_os.environ.get("CER_DATA_DIR", "data"))
    timezone: str = _os.environ.get("CER_TIMEZONE", "Europe/Rome")

def get_config() -> AppConfig:
    return AppConfig()
