"""Offline smoke-test driver for the scene_graph package.

Runs the geometry and query layers without ROS 2 or Isaac Sim:
    1. Load a fake MapObjectList from test_scene.json
    2. Build scene relations
    3. Run position/object listing queries
    4. Optionally try CLIP text search
    5. Optionally start the MCP HTTP server on the configured endpoint

Usage:
    python tools/offline_mapper_demo.py
    python tools/offline_mapper_demo.py --no-server
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph import cg_query
from scene_graph.cg.slam_classes import MapObjectList
from scene_graph.paths import GRAPH_PATH
from scene_graph.scene_relations import build_scene_graph, graph_to_text, save_scene_graph


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [offline-demo] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parent
FIXTURE = TOOLS_DIR / "test_scene.json"


def ensure_fixture() -> dict:
    if not FIXTURE.exists():
        log.info("Generating fixture at %s", FIXTURE)
        import generate_test_fixture

        generate_test_fixture.main()
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    log.info("Loaded fixture: %d objects", len(data["objects"]))
    return data


def load_map_object_list(data: dict) -> MapObjectList:
    """Load serialized objects into the local MapObjectList implementation."""
    import numpy as np

    objs = data["objects"]
    for obj in objs:
        for key in ("clip_ft", "text_ft"):
            if isinstance(obj.get(key), list):
                obj[key] = np.array(obj[key], dtype=np.float32)
        for key in ("pcd_np", "bbox_np", "pcd_color_np"):
            if isinstance(obj.get(key), list):
                obj[key] = np.array(obj[key], dtype=np.float64)

    obj_list = MapObjectList()
    obj_list.load_serializable(objs)
    log.info("MapObjectList built: %d entries", len(obj_list))
    return obj_list


def run_relations_check(obj_list: MapObjectList) -> dict:
    graph = build_scene_graph(obj_list, near_dist_m=3.0, min_detections=1)
    save_scene_graph(graph, GRAPH_PATH)

    print("\n" + "=" * 60)
    print("SCENE GRAPH (text)")
    print("=" * 60)
    print(graph_to_text(graph))
    print("=" * 60 + "\n")

    log.info(
        "Relations: %d nodes, %d edges -> %s",
        len(graph["nodes"]),
        len(graph["edges"]),
        GRAPH_PATH,
    )
    return graph


def run_position_queries(obj_list: MapObjectList) -> None:
    print("--- nearby_objects(x=3.0, y=-1.5, radius=2.0) ---")
    results = cg_query.query_by_position(
        obj_list,
        x=3.0,
        y=-1.5,
        radius=2.0,
        min_detections=1,
    )
    for item in results:
        print(
            f"  {item['label']:12s}  dist={item['distance_m']:.2f}m"
            f"  {item['centroid']}"
        )

    print("\n--- list_objects(min_detections=1) ---")
    for item in cg_query.list_objects(obj_list, min_detections=1):
        print(
            f"  [{item['index']}] {item['label']:12s}  centroid={item['centroid']}"
            f"  seen={item['num_detections']}x"
        )
    print()


def run_clip_query(obj_list: MapObjectList) -> None:
    try:
        print("--- query_by_text('forklift') ---")
        results = cg_query.query_by_text(
            obj_list,
            "forklift",
            top_k=3,
            device="cpu",
            min_detections=1,
        )
        for item in results:
            print(
                f"  {item['label']:12s}  sim={item['similarity']:.4f}"
                f"  {item['centroid']}"
            )
        print()
    except Exception as exc:
        log.warning("CLIP query skipped (models not available): %s", exc)


class MockMapper:
    """Stand-in for IncrementalMapper using fixture data only."""

    def __init__(self, obj_list: MapObjectList):
        self.objects = obj_list
        self.lock = threading.RLock()
        self._count = len(obj_list)

    def status(self) -> dict:
        return {
            "rgb_frames_received": 20,
            "update_cycles": 1,
            "objects_in_map": self._count,
            "update_running": False,
            "map_path": str(FIXTURE),
            "last_update_ago_s": 0.0,
            "_note": "MockMapper fixture data; no live ROS",
        }

    def get_lidar_cloud(self):
        return None


def start_test_server(mapper: MockMapper) -> None:
    from scene_graph.scene_graph_mcp import (
        MCP_HOST,
        MCP_PATH,
        MCP_PORT,
        MCP_TRANSPORT,
        build_server,
    )

    server = build_server(mapper)  # type: ignore[arg-type]
    log.info("Test MCP server -> http://%s:%d%s", MCP_HOST, MCP_PORT, MCP_PATH)
    log.info("Try:")
    log.info(
        "  curl -X POST http://%s:%d%s "
        "-H 'Content-Type: application/json' "
        "-d '{\"method\":\"tools/call\",\"params\":{\"name\":\"mapper_status\",\"arguments\":{}}}'",
        MCP_HOST,
        MCP_PORT,
        MCP_PATH,
    )
    server.run(transport=MCP_TRANSPORT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Run checks only and do not start the MCP server",
    )
    args = parser.parse_args()

    print("\n[1/4] Loading fixture...")
    data = ensure_fixture()
    obj_list = load_map_object_list(data)

    print("\n[2/4] Building scene relations...")
    run_relations_check(obj_list)

    print("\n[3/4] Running position queries...")
    run_position_queries(obj_list)

    print("\n[4/4] Running CLIP text query...")
    run_clip_query(obj_list)

    if args.no_server:
        print("Done (--no-server, skipping HTTP server).")
        return

    print("\n[5/5] Starting test HTTP server (Ctrl-C to stop)...")
    start_test_server(MockMapper(obj_list))


if __name__ == "__main__":
    main()
