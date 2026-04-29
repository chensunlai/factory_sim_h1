#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/share/InteriorAgent")


def process(scene_dir: Path):
    scene = scene_dir / f"{scene_dir.name}.usda"
    if not scene.is_file():
        return

    out = scene.with_name(f"{scene.stem}_plane.usda")
    shutil.copy2(scene, out)

    stage = Usd.Stage.Open(str(out))
    root = stage.GetDefaultPrim()
    floor = stage.GetPrimAtPath(f"{root.GetPath()}/Meshes/floor")
    if not floor.IsValid():
        raise RuntimeError(f"floor scope not found: {scene}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    tiles = [prim for prim in floor.GetChildren() if prim.IsActive()]
    if not tiles:
        raise RuntimeError(f"no floor mesh found: {scene}")

    floor_z = sum(float(cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMax()[2]) for prim in tiles) / len(tiles)
    scene_box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    min_pt = scene_box.GetMin()
    max_pt = scene_box.GetMax()
    cx = (float(min_pt[0]) + float(max_pt[0])) * 0.5
    cy = (float(min_pt[1]) + float(max_pt[1])) * 0.5
    hx = (float(max_pt[0]) - float(min_pt[0])) * 0.5 + 0.05
    hy = (float(max_pt[1]) - float(min_pt[1])) * 0.5 + 0.05

    plane_path = f"{root.GetPath()}/floor_plane"
    if stage.GetPrimAtPath(plane_path).IsValid():
        stage.RemovePrim(plane_path)

    plane = UsdGeom.Mesh.Define(stage, plane_path)
    plane.CreatePointsAttr(
        [
            Gf.Vec3f(-hx, -hy, 0.0),
            Gf.Vec3f(hx, -hy, 0.0),
            Gf.Vec3f(hx, hy, 0.0),
            Gf.Vec3f(-hx, hy, 0.0),
        ]
    )
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateExtentAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, hy, 0.0)])
    plane.CreateDoubleSidedAttr(True)
    plane.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    plane.AddTranslateOp().Set(Gf.Vec3d(cx, cy, floor_z + 0.02))
    UsdGeom.Imageable(plane).CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Imageable(plane).MakeInvisible()

    stage.GetRootLayer().Save()
    print(out)


for scene_dir in sorted(ROOT.iterdir()):
    if scene_dir.is_dir() and not scene_dir.name.startswith("."):
        process(scene_dir)
