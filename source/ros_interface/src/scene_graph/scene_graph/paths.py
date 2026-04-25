from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_ROOT / "models"

DEFAULT_SAVE_ROOT = Path.home() / ".tmp" / "cg_live"
SAVE_ROOT = Path(os.environ.get("SCENE_GRAPH_SAVE_ROOT", DEFAULT_SAVE_ROOT)).expanduser()
MAP_DIR = SAVE_ROOT / "map"
MAP_PATH = MAP_DIR / "scene_map.pkl.gz"
LIDAR_PATH = MAP_DIR / "lidar_bg.pkl.gz"
GRAPH_PATH = MAP_DIR / "scene_graph.json"
DEBUG_IMAGE_PATH = SAVE_ROOT / "yolo_debug.jpg"


def ensure_save_dirs() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_file_path(env_key: str, default_path: Path) -> Path:
    value = os.environ.get(env_key)
    path = Path(value).expanduser() if value else default_path
    if not path.exists():
        raise FileNotFoundError(
            f"Required model file not found: {path}. Set {env_key} to override the location."
        )
    return path


def resolve_yolo_world_path() -> Path:
    return _resolve_file_path(
        "SCENE_GRAPH_YOLO_WORLD_PATH",
        MODELS_DIR / "yolov8l-world.pt",
    )


def resolve_mobile_sam_path() -> Path:
    return _resolve_file_path(
        "SCENE_GRAPH_MOBILE_SAM_PATH",
        MODELS_DIR / "mobile_sam.pt",
    )


def resolve_openclip_pretrained() -> str:
    weights_path = os.environ.get("SCENE_GRAPH_OPENCLIP_WEIGHTS_PATH")
    if weights_path:
        path = Path(weights_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                "Configured OpenCLIP weights were not found: "
                f"{path}. Set SCENE_GRAPH_OPENCLIP_WEIGHTS_PATH to a valid file."
            )
        return str(path)

    bundled_safetensors_path = MODELS_DIR / "open_clip_pytorch_model.safetensors"
    if bundled_safetensors_path.exists():
        return str(bundled_safetensors_path)

    bundled_bin_path = MODELS_DIR / "open_clip_pytorch_model.bin"
    if bundled_bin_path.exists():
        return str(bundled_bin_path)

    return os.environ.get("SCENE_GRAPH_OPENCLIP_PRETRAINED", "laion2b_s32b_b79k")
