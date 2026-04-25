"""Scene Graph MCP Server — streamable-http edition.

Starts IncrementalMapper as an in-process background module, then exposes
MCP tools over HTTP for OpenComposer brain to call.

Requires ROS2 Jazzy sourced before running:
    source /opt/ros/jazzy/setup.bash

OpenComposer config (~/.composer/config.json):
    "scene_graph": {
      "type": "streamable-http",
      "url":  "http://127.0.0.1:11452/mcp"
    }

Tools
-----
    query_objects(query, top_k?)         CLIP text search → positions
    list_scene_objects(min_detections?)  all objects with centroids
    get_scene_graph(format?)             nodes + typed edges
    nearby_objects(x, y, radius_m?)     objects near a coordinate
    get_object_position(query)           centroid of best-matching object
    mapper_status()                      frame counts, update cycles
    get_lidar_map()                      background occupancy summary
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scene_graph_mcp] %(levelname)s %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger(__name__)

from . import cg_query
from .incremental_mapper import DEVICE, IncrementalMapper
from .paths import GRAPH_PATH, LIDAR_PATH, MAP_PATH
from .scene_relations import build_scene_graph, graph_to_text, load_scene_graph, save_scene_graph

import rclpy
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import Response

# ── Server config ─────────────────────────────────────────────────────────────
MCP_HOST = os.environ.get("SCENE_GRAPH_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("SCENE_GRAPH_MCP_PORT", "11452"))
MCP_PATH = os.environ.get("SCENE_GRAPH_MCP_PATH", "/mcp")
MCP_TRANSPORT = os.environ.get("SCENE_GRAPH_MCP_TRANSPORT", "streamable-http")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _snapshot(mapper: IncrementalMapper):
    """Thread-safe snapshot of current MapObjectList."""
    from .cg.slam_classes import MapObjectList
    with mapper.lock:
        serial = mapper.objects.to_serializable()
    snap = MapObjectList()
    snap.load_serializable(serial)
    return snap


# ── MCP server ────────────────────────────────────────────────────────────────

def build_server(mapper: IncrementalMapper) -> FastMCP:
    server = FastMCP(
        "scene_graph_memory",
        host=MCP_HOST,
        port=MCP_PORT,
        streamable_http_path=MCP_PATH,
    )

    # ── query_objects ─────────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Search the live scene graph with a natural-language query "
            "using CLIP text-to-image cosine similarity. "
            "Returns object labels, 3D world-frame centroids (x,y,z), "
            "and similarity scores."
        )
    )
    def query_objects(query: str, top_k: int = 5) -> str:
        snap = _snapshot(mapper)
        if len(snap) == 0:
            return _ok({"error": "Scene graph is empty. Mapper is still building — try again shortly."})
        results = cg_query.query_by_text(snap, query, top_k=top_k, device=DEVICE)
        return _ok({"query": query, "results": results, "total_objects": len(snap)})

    # ── get_object_position ───────────────────────────────────────────────────

    @server.tool(
        description=(
            "Get the 3D world-frame centroid (x, y, z) of the best-matching "
            "object for a natural-language query. "
            "Returns label, centroid, and similarity score."
        )
    )
    def get_object_position(query: str) -> str:
        snap = _snapshot(mapper)
        if len(snap) == 0:
            return _ok({"error": "Scene graph is empty."})
        results = cg_query.query_by_text(snap, query, top_k=1, device=DEVICE)
        if not results:
            return _ok({"error": "No matching object found."})
        best = results[0]
        return _ok({
            "label":      best["label"],
            "centroid":   best["centroid"],
            "similarity": best["similarity"],
        })

    # ── list_scene_objects ────────────────────────────────────────────────────

    @server.tool(
        description=(
            "List all objects in the live scene graph "
            "with 3D centroids and detection counts. "
            "min_detections: skip objects seen fewer than N times (default 1)."
        )
    )
    def list_scene_objects(min_detections: int = 1) -> str:
        snap = _snapshot(mapper)
        objects = cg_query.list_objects(snap, min_detections=min_detections)
        return _ok({"count": len(objects), "objects": objects})

    # ── get_scene_graph ───────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Return the full scene graph: objects (nodes) and their spatial "
            "relations (edges). Relation types: next_to, near, left_of, "
            "in_front_of, above, on_top_of, inside. "
            "format='text'  → readable paragraph for LLM reasoning. "
            "format='json'  → structured {nodes, edges} dict."
        )
    )
    def get_scene_graph(format: str = "text") -> str:
        graph = load_scene_graph(GRAPH_PATH)
        if graph is None or len(graph.get("nodes", [])) == 0:
            snap = _snapshot(mapper)
            if len(snap) == 0:
                return _ok({"error": "Scene graph not yet built."})
            graph = build_scene_graph(snap)
            save_scene_graph(graph, GRAPH_PATH)
        return graph_to_text(graph) if format == "text" else _ok(graph)

    # ── nearby_objects ────────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Find all scene-graph objects within radius_m metres "
            "of world coordinate (x, y)."
        )
    )
    def nearby_objects(x: float, y: float, radius_m: float = 3.0) -> str:
        snap = _snapshot(mapper)
        results = cg_query.query_by_position(snap, x, y, radius=radius_m)
        return _ok({"x": x, "y": y, "radius_m": radius_m, "results": results})

    # ── clear_map ─────────────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Clear all objects from the scene graph and delete persisted map files. "
            "Use this to reset spatial memory before exploring a new environment."
        )
    )
    def clear_map() -> str:
        with mapper.lock:
            mapper.objects.clear()
            mapper._update_count = 0
            mapper._total_objs   = 0
            mapper._last_update  = 0.0
        for path in (MAP_PATH, GRAPH_PATH, LIDAR_PATH):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        log.info("Map cleared by MCP request.")
        return _ok({"cleared": True})

    # ── mapper_status ─────────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Return mapper status: RGB frames received, update cycles completed, "
            "objects in map, and whether an update is currently running."
        )
    )
    def mapper_status() -> str:
        return _ok(mapper.status())

    # ── get_lidar_map ─────────────────────────────────────────────────────────

    @server.tool(
        description=(
            "Return a summary of the accumulated lidar background point cloud "
            "(world-frame obstacle map). Shows point count and bounding box."
        )
    )
    def get_lidar_map() -> str:
        cloud = mapper.get_lidar_cloud()
        if cloud is None or len(cloud) == 0:
            return _ok({"error": "No lidar data accumulated yet."})
        mins = cloud.min(axis=0).tolist()
        maxs = cloud.max(axis=0).tolist()
        return _ok({
            "point_count": len(cloud),
            "bbox_min":    [round(v, 2) for v in mins],
            "bbox_max":    [round(v, 2) for v in maxs],
            "path":        str(LIDAR_PATH),
        })

    # Handle session termination (DELETE /mcp) — FastMCP doesn't implement this
    # endpoint but clients like OpenComposer send it to clean up sessions.
    @server.custom_route(MCP_PATH, methods=["DELETE"])
    async def _delete_session(request: Request) -> Response:
        return Response(status_code=200)

    return server


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init(args=None)

    mapper = IncrementalMapper()
    mapper.start()

    log.info("Pre-loading detection models before serving MCP requests…")
    from .incremental_mapper import _Models
    _Models.load(DEVICE)
    log.info("Models ready.")

    log.info("Scene graph MCP server on http://%s:%d%s", MCP_HOST, MCP_PORT, MCP_PATH)

    server = build_server(mapper)
    try:
        server.run(transport=MCP_TRANSPORT)
    finally:
        mapper.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
