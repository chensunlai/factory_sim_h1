"""Geometric scene-graph relation extractor.

Given a MapObjectList (from incremental_mapper), computes pairwise spatial
relations between objects and outputs a scene graph with nodes + edges as JSON.

Relation types (computed from 3D centroids + oriented bounding boxes):
    near          Euclidean distance below threshold (default 2.0 m)
    left_of       centroid_A.y > centroid_B.y  (Isaac Sim: +y is left)
    right_of      centroid_A.y < centroid_B.y
    in_front_of   centroid_A.x > centroid_B.x  (+x is forward)
    behind        centroid_A.x < centroid_B.x
    above         centroid_A.z > centroid_B.z + half_height_B
    below         centroid_A.z < centroid_B.z - half_height_B
    on_top_of     above + bboxes overlap in XY plane
    inside        centroid_A is within bbox_B (containment)
    next_to       near + approx same height (|dz| < 0.5 m)

Output schema (also saved to SCENE_GRAPH_SAVE_ROOT/map/scene_graph.json):
    {
      "nodes": [
        { "id": 0, "label": "shelf", "centroid": [x,y,z],
          "bbox_extent": [dx,dy,dz], "num_detections": 5 },
        ...
      ],
      "edges": [
        { "src": 0, "dst": 1, "relation": "left_of", "distance_m": 1.3 },
        ...
      ],
      "updated_at": 1713000000.0
    }

Public API
----------
    build_scene_graph(obj_list, **kw) -> dict
    save_scene_graph(graph, path)
    load_scene_graph(path) -> dict
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .paths import GRAPH_PATH

SCENE_GRAPH_PATH = GRAPH_PATH

# ── Relation thresholds ───────────────────────────────────────────────────────
NEAR_DIST_M        = 2.0    # max distance to be considered "near"
HEIGHT_SAME_M      = 0.5    # |dz| threshold for "next_to" (same level)
ABOVE_MARGIN_M     = 0.1    # centroid_A.z must exceed top of B by this margin
XY_OVERLAP_THRESH  = 0.3    # IoU in XY plane to qualify as "on_top_of"
MIN_DETECTIONS     = 1      # ignore objects seen fewer times


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centroid(obj: dict) -> np.ndarray:
    pcd = obj.get("pcd")
    if pcd is not None:
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            return pts.mean(axis=0)
    bbox = obj.get("bbox")
    if bbox is not None:
        pts = np.asarray(bbox.get_box_points())
        if len(pts) > 0:
            return pts.mean(axis=0)
    return np.zeros(3)


def _bbox_extent(obj: dict) -> np.ndarray:
    """Return (dx, dy, dz) half-extents of the oriented bounding box."""
    bbox = obj.get("bbox")
    if bbox is not None:
        try:
            ext = np.asarray(bbox.extent)   # open3d OBB .extent = full side lengths
            return ext / 2.0
        except Exception:
            pass
    return np.array([0.3, 0.3, 0.3])


def _xy_interval(centroid: np.ndarray, half_ext: np.ndarray):
    """Return (xmin, xmax, ymin, ymax) for XY-plane footprint."""
    return (
        centroid[0] - half_ext[0], centroid[0] + half_ext[0],
        centroid[1] - half_ext[1], centroid[1] + half_ext[1],
    )


def _xy_iou(a_c, a_e, b_c, b_e) -> float:
    """1D IoU in both X and Y, multiplied (axis-aligned approximation)."""
    ax1, ax2, ay1, ay2 = _xy_interval(a_c, a_e)
    bx1, bx2, by1, by2 = _xy_interval(b_c, b_e)

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _inside_bbox(point: np.ndarray, obj: dict) -> bool:
    """True if *point* lies inside obj's bounding box."""
    bbox = obj.get("bbox")
    if bbox is None:
        return False
    try:
        # open3d OBB: get_point_indices_within_bounding_box is for point clouds,
        # but we can use the axis-aligned approximation via min_bound/max_bound
        aabb = bbox.get_axis_aligned_bounding_box()
        mn = np.asarray(aabb.min_bound)
        mx = np.asarray(aabb.max_bound)
        return bool(np.all(point >= mn) and np.all(point <= mx))
    except Exception:
        return False


def _label(obj: dict) -> str:
    from collections import Counter
    cnames = obj.get("class_name")
    if cnames and len(cnames) > 0:
        mc = Counter(str(c) for c in cnames if c is not None).most_common(1)
        if mc:
            return mc[0][0]
    cids = obj.get("class_id")
    if cids and len(cids) > 0:
        mc = Counter(int(c) for c in cids if c is not None).most_common(1)
        if mc:
            return str(mc[0][0])
    return "object"


# ── Core builder ──────────────────────────────────────────────────────────────

def build_scene_graph(
    obj_list,                           # MapObjectList
    near_dist_m: float    = NEAR_DIST_M,
    min_detections: int   = MIN_DETECTIONS,
) -> dict[str, Any]:
    """Build a scene graph dict with nodes and typed edges.

    Args:
        obj_list: local MapObjectList (in-memory, after mapping)
        near_dist_m: max distance for "near" / directional relations
        min_detections: skip poorly-observed objects

    Returns:
        {"nodes": [...], "edges": [...], "updated_at": float}
    """
    # Filter to well-observed objects
    candidates = [
        (i, obj) for i, obj in enumerate(obj_list)
        if obj.get("num_detections", 0) >= min_detections
    ]

    # Build node list
    nodes: list[dict[str, Any]] = []
    centroids: list[np.ndarray] = []
    half_exts: list[np.ndarray] = []

    for node_id, (orig_idx, obj) in enumerate(candidates):
        c   = _centroid(obj)
        ext = _bbox_extent(obj)
        centroids.append(c)
        half_exts.append(ext)
        nodes.append({
            "id":             node_id,
            "orig_idx":       orig_idx,
            "label":          _label(obj),
            "centroid":       [round(float(v), 3) for v in c],
            "bbox_extent":    [round(float(v), 3) for v in ext * 2],  # full side lengths
            "num_detections": int(obj.get("num_detections", 0)),
        })

    # Build edges: iterate all pairs (i, j) with i < j
    edges: list[dict[str, Any]] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            c_i, c_j   = centroids[i], centroids[j]
            e_i, e_j   = half_exts[i], half_exts[j]
            dist = float(np.linalg.norm(c_i - c_j))

            if dist > near_dist_m * 2.0:
                continue  # too far — no relation at all

            dx = float(c_i[0] - c_j[0])   # +x = forward
            dy = float(c_i[1] - c_j[1])   # +y = left
            dz = float(c_i[2] - c_j[2])   # +z = up
            abs_dz_ij = abs(dz)

            def _edge(src, dst, rel):
                edges.append({
                    "src": src, "dst": dst,
                    "relation": rel,
                    "distance_m": round(dist, 3),
                })

            # ── near / next_to ────────────────────────────────────────────────
            if dist <= near_dist_m:
                if abs_dz_ij < HEIGHT_SAME_M:
                    _edge(i, j, "next_to")
                else:
                    _edge(i, j, "near")

            # ── directional (horizontal plane) ────────────────────────────────
            horiz = math.hypot(dx, dy)
            if horiz > 0.3:
                if abs(dx) >= abs(dy):
                    if dx > 0:
                        _edge(i, j, "in_front_of")
                    else:
                        _edge(j, i, "in_front_of")
                else:
                    if dy > 0:
                        _edge(i, j, "left_of")
                    else:
                        _edge(j, i, "left_of")

            # ── vertical ─────────────────────────────────────────────────────
            top_j   = c_j[2] + e_j[2]
            bot_j   = c_j[2] - e_j[2]
            top_i   = c_i[2] + e_i[2]
            bot_i   = c_i[2] - e_i[2]

            if c_i[2] > top_j + ABOVE_MARGIN_M:
                xy_iou = _xy_iou(c_i, e_i, c_j, e_j)
                if xy_iou > XY_OVERLAP_THRESH:
                    _edge(i, j, "on_top_of")
                else:
                    _edge(i, j, "above")

            elif c_j[2] > top_i + ABOVE_MARGIN_M:
                xy_iou = _xy_iou(c_j, e_j, c_i, e_i)
                if xy_iou > XY_OVERLAP_THRESH:
                    _edge(j, i, "on_top_of")
                else:
                    _edge(j, i, "above")

            # ── containment ───────────────────────────────────────────────────
            if _inside_bbox(c_i, candidates[j][1]):
                _edge(i, j, "inside")
            elif _inside_bbox(c_j, candidates[i][1]):
                _edge(j, i, "inside")

    return {
        "nodes":      nodes,
        "edges":      edges,
        "updated_at": time.time(),
    }


# ── Text rendering for brain context ─────────────────────────────────────────

def graph_to_text(graph: dict[str, Any]) -> str:
    """Render the scene graph as a readable paragraph for an LLM.

    Example output:
        Scene contains 5 objects:
          [0] shelf  at (3.20, -1.50, 1.00)  (seen 8x)
          [1] forklift  at (1.80, 0.30, 0.50)  (seen 5x)
          ...
        Relations:
          shelf [0]  is  left_of  forklift [1]  (1.3 m)
          box [2]  is  on_top_of  shelf [0]  (0.8 m)
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    id_to_label = {n["id"]: n["label"] for n in nodes}

    lines = [f"Scene contains {len(nodes)} objects:"]
    for n in nodes:
        c = n["centroid"]
        lines.append(
            f"  [{n['id']}] {n['label']}  at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
            f"  (seen {n['num_detections']}x)"
        )

    if edges:
        lines.append(f"\nRelations ({len(edges)}):")
        for e in edges:
            sl = id_to_label.get(e["src"], "?")
            dl = id_to_label.get(e["dst"], "?")
            lines.append(
                f"  {sl} [{e['src']}]  is  {e['relation']}  "
                f"{dl} [{e['dst']}]  ({e['distance_m']} m)"
            )
    else:
        lines.append("\nNo spatial relations found yet.")

    return "\n".join(lines)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_scene_graph(graph: dict[str, Any], path: Path | str = SCENE_GRAPH_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scene_graph(path: Path | str = SCENE_GRAPH_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
