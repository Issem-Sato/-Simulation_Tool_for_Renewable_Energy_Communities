# src/cer_core/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

try:
    import torch  # facoltativo
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class AppConfig:
    app_name: str = "CER Simulator"
    # <- quello che app.py usa
    data_root: Path = Path("data")
    timezone: str = "Europe/Rome"

    default_w_per_m2: float = 200.0
    default_packing: float = 0.7

    device: str = (
        "cuda"
        if (_HAS_TORCH and getattr(torch.cuda, "is_available", lambda: False)())
        else "cpu"
    )


def get_config() -> AppConfig:
    cfg = AppConfig()
    # override da env
    app_name = os.getenv("CER_APP_NAME")
    if app_name:
        cfg.app_name = app_name

    tz = os.getenv("CER_TIMEZONE")
    if tz:
        cfg.timezone = tz

    dd = os.getenv("CER_DATA_DIR")
    if dd:
        cfg.data_dir = Path(dd)

    dev = os.getenv("CER_DEVICE")
    if dev in {"cpu", "cuda"}:
        cfg.device = dev

    return cfg
