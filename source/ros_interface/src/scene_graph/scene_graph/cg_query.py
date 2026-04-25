"""Query interface for a concept-graphs scene map (pkl.gz).

Loads MapObjectList from a map file and supports:
  • text → object  (CLIP text-image similarity)
  • position → nearby objects  (3D Euclidean distance)
  • list all objects
  • get object centroid (world position)

The map file is the output of cg_runner.run_mapping_stage().
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cg.slam_classes import MapObjectList
def _get_clip():
    """Reuse the CLIP model already loaded by IncrementalMapper._Models."""
    from .incremental_mapper import _Models
    if not _Models.ready:
        raise RuntimeError("CLIP model not yet loaded — mapper has not run its first update.")
    return _Models.clip, _Models.tok, _Models.cfg_device if hasattr(_Models, 'cfg_device') else "cuda:0"


# ── Map loader ────────────────────────────────────────────────────────────────

def load_map(map_path: str | Path) -> MapObjectList:
    """Load a scene map from pkl.gz produced by cg_runner."""
    with gzip.open(Path(map_path), "rb") as f:
        data = pickle.load(f)

    obj_list = MapObjectList()
    if isinstance(data, dict) and "objects" in data:
        obj_list.load_serializable(data["objects"])
    elif isinstance(data, list):
        obj_list.load_serializable(data)
    else:
        raise ValueError(f"Unexpected map format in {map_path}")

    return obj_list


# ── Query functions ───────────────────────────────────────────────────────────

def query_by_text(
    obj_list: MapObjectList,
    query: str,
    top_k: int = 5,
    device: str = "cuda:0",
    min_detections: int = 1,
) -> list[dict[str, Any]]:
    """Return top-k objects whose CLIP image features best match *query*.

    Returns list of dicts:
        label, class_id, centroid [x,y,z], similarity, num_detections
    """
    clip_model, clip_tokenizer, clip_device = _get_clip()

    candidates = [
        o for o in obj_list
        if o.get("clip_ft") is not None
        and o.get("num_detections", 0) >= min_detections
    ]
    if not candidates:
        return []

    # Text embedding
    with torch.no_grad():
        tokens = clip_tokenizer([query]).to(clip_device)
        text_ft = clip_model.encode_text(tokens)[0]   # (D,)
        text_ft = text_ft / text_ft.norm()

    # Stack image embeddings
    img_fts = torch.stack([
        torch.as_tensor(o["clip_ft"], dtype=torch.float32, device=clip_device)
        for o in candidates
    ])  # (N, D)
    img_fts = img_fts / img_fts.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    sims = (img_fts @ text_ft).cpu().tolist()  # (N,)

    scored = sorted(
        zip(sims, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for sim, obj in scored[:top_k]:
        centroid = _centroid(obj)
        results.append({
            "class_id": int(obj.get("class_id", [-1])[0]),
            "label": _label(obj),
            "centroid": [round(float(v), 3) for v in centroid],
            "similarity": round(float(sim), 4),
            "num_detections": int(obj.get("num_detections", 0)),
        })
    return results


def query_by_position(
    obj_list: MapObjectList,
    x: float,
    y: float,
    z: float = 0.0,
    radius: float = 3.0,
    min_detections: int = 1,
) -> list[dict[str, Any]]:
    """Return objects within *radius* metres of (x, y, z)."""
    query_pt = np.array([x, y, z], dtype=np.float64)
    results = []
    for obj in obj_list:
        if obj.get("num_detections", 0) < min_detections:
            continue
        centroid = _centroid(obj)
        dist = float(np.linalg.norm(centroid - query_pt))
        if dist <= radius:
            results.append({
                "label": _label(obj),
                "class_id": int(obj.get("class_id", [-1])[0]),
                "centroid": [round(float(v), 3) for v in centroid],
                "distance_m": round(dist, 3),
                "num_detections": int(obj.get("num_detections", 0)),
            })
    results.sort(key=lambda d: d["distance_m"])
    return results


def list_objects(
    obj_list: MapObjectList,
    min_detections: int = 1,
) -> list[dict[str, Any]]:
    """Return all objects with their centroids and detection counts."""
    out = []
    for i, obj in enumerate(obj_list):
        if obj.get("num_detections", 0) < min_detections:
            continue
        centroid = _centroid(obj)
        out.append({
            "index": i,
            "label": _label(obj),
            "class_id": int(obj.get("class_id", [-1])[0]),
            "centroid": [round(float(v), 3) for v in centroid],
            "num_detections": int(obj.get("num_detections", 0)),
        })
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centroid(obj: dict) -> np.ndarray:
    """Compute the centroid of an object's point cloud."""
    pcd = obj.get("pcd")
    if pcd is not None:
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            return pts.mean(axis=0)
    # Fallback: bounding box centre
    bbox = obj.get("bbox")
    if bbox is not None:
        pts = np.asarray(bbox.get_box_points())
        if len(pts) > 0:
            return pts.mean(axis=0)
    return np.zeros(3)


def _label(obj: dict) -> str:
    """Best-effort label: prefer class_name list, fall back to class_id int."""
    from collections import Counter

    # class_name is stored by incremental_mapper as a list of strings per detection
    cnames = obj.get("class_name")
    if cnames and len(cnames) > 0:
        most_common = Counter(
            str(c) for c in cnames if c is not None
        ).most_common(1)
        if most_common:
            return most_common[0][0]

    # Fall back to integer class id
    cids = obj.get("class_id")
    if cids and len(cids) > 0:
        most_common = Counter(
            int(c) for c in cids if c is not None
        ).most_common(1)
        if most_common:
            return str(most_common[0][0])

    return "object"
