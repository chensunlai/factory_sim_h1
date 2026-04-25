"""Generate test_scene.json — fake warehouse scene for offline testing.

Writes a serialized MapObjectList (same format as scene_map.pkl.gz objects key)
to test_scene.json so offline_mapper_demo.py can load it without ROS or Isaac Sim.

Object layout (top view, X=forward, Y=left):
    pallet   (0.5,  0.0, 0.1)   small, floor level
    shelf_A  (3.0, -1.5, 1.0)   tall, against right wall
    shelf_B  (3.0,  1.5, 1.0)   tall, against left wall
    box      (3.0, -1.5, 2.1)   on top of shelf_A
    forklift (1.5,  0.0, 0.5)   in the aisle
    door     (0.0,  3.0, 1.2)   at the far wall

Run:
    python tools/generate_test_fixture.py  → writes test_scene.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).parent / "test_scene.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)


def _unit_vec(size: int = 1024) -> list[float]:  # ViT-H-14 outputs 1024-d
    """Random L2-normalised float32 vector."""
    v = rng.standard_normal(size).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _box_points(cx, cy, cz, dx, dy, dz) -> np.ndarray:
    """8 corners of an axis-aligned box centred at (cx,cy,cz)."""
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append([cx + sx * hx, cy + sy * hy, cz + sz * hz])
    return np.array(corners, dtype=np.float64)


def _make_object(
    class_name: str,
    class_id: int,
    cx: float, cy: float, cz: float,
    dx: float, dy: float, dz: float,
    num_detections: int = 6,
) -> dict:
    """Build one serializable object entry (matches MapObjectList.to_serializable)."""
    pts = _box_points(cx, cy, cz, dx, dy, dz)

    # Dense point cloud: random points inside the box
    n_pts = 10
    pcd_np = rng.uniform(
        [cx - dx/2, cy - dy/2, cz - dz/2],
        [cx + dx/2, cy + dy/2, cz + dz/2],
        size=(n_pts, 3),
    ).astype(np.float64)

    color = rng.uniform(0.3, 0.9, size=(n_pts, 3)).astype(np.float64)
    ft = _unit_vec()

    return {
        "class_id":       [class_id] * num_detections,
        "class_name":     [class_name] * num_detections,
        "conf":           [0.85] * num_detections,
        "num_detections": num_detections,
        "n_points":       [n_pts],
        "inst_color":     [0.5, 0.6, 0.9],
        # MapObjectList.load_serializable 要求 np.ndarray，JSON 存为嵌套 list
        "clip_ft":        ft,            # list[float] → to_tensor 会收到 list，需 patch
        "text_ft":        ft,
        "pcd_np":         pcd_np.tolist(),
        "bbox_np":        pts.tolist(),
        "pcd_color_np":   color.tolist(),
    }


# ── Scene definition ──────────────────────────────────────────────────────────

OBJECTS = [
    # name,      id,  cx,   cy,   cz,   dx,   dy,   dz,  seen
    ("pallet",    0,  0.5,  0.0,  0.10, 1.2,  0.8,  0.2,  4),
    ("shelf_A",   1,  3.0, -1.5,  1.00, 0.5,  2.0,  2.0, 12),
    ("shelf_B",   1,  3.0,  1.5,  1.00, 0.5,  2.0,  2.0, 10),
    ("box",       2,  3.0, -1.5,  2.10, 0.4,  0.4,  0.4,  5),
    ("forklift",  3,  1.5,  0.0,  0.50, 2.0,  1.0,  1.0,  8),
    ("door",      4,  0.0,  3.0,  1.20, 0.2,  1.0,  2.4,  3),
]


def main():
    objs = [_make_object(*row) for row in OBJECTS]
    data = {
        "objects":      objs,
        "update_count": 1,
        "rgb_frames":   20,
        "class_names":  [r[0] for r in OBJECTS],
        "saved_at":     0.0,
        "_note":        "Fake fixture for offline testing — not from Isaac Sim",
    }
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written {len(objs)} objects → {OUTPUT}")


if __name__ == "__main__":
    main()
