from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from .config import Config

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{name}/float16/latest/{name}.task"
)


def model_dir(cfg: Config) -> Path:
    base = cfg.model_dir or os.environ.get("SAFEWORK_CV_HOME")
    root = Path(base) if base else Path.home() / ".cache" / "safework_cv"
    path = root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_pose_model(cfg: Config) -> Path:
    name = f"pose_landmarker_{cfg.pose_variant}"
    target = model_dir(cfg) / f"{name}.task"
    if target.exists():
        return target
    partial = target.with_suffix(".part")
    try:
        with (
            urllib.request.urlopen(POSE_URL.format(name=name), timeout=60) as response,
            open(partial, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
