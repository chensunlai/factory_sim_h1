from collections import Counter
import copy

import faiss
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from .general_utils import to_tensor, to_numpy
from .slam_classes import MapObjectList, DetectionList
from .ious import compute_3d_iou, compute_3d_iou_accuracte_batch, compute_iou_batch


# ── Point cloud helpers ───────────────────────────────────────────────────────

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    counter = Counter(labels[labels >= 0].tolist())
    if not counter:
        return pcd
    best, _ = counter.most_common(1)[0]
    mask = labels == best
    pts = np.asarray(pcd.points)[mask]
    cols = np.asarray(pcd.colors)[mask] if len(pcd.colors) else np.zeros((mask.sum(), 3))
    if len(pts) < 5:
        return pcd
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(cols)
    return out


def process_pcd(pcd: o3d.geometry.PointCloud, cfg, run_dbscan: bool = True):
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(pcd, eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points)
    return pcd


def get_bounding_box(cfg, pcd: o3d.geometry.PointCloud):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError:
            return pcd.get_axis_aligned_bounding_box()
    return pcd.get_axis_aligned_bounding_box()


# ── Object merge ──────────────────────────────────────────────────────────────

def merge_obj2_into_obj1(cfg, obj1: dict, obj2: dict, run_dbscan: bool = True) -> dict:
    n1 = obj1['num_detections']
    n2 = obj2['num_detections']

    for k in obj1:
        if k in ('pcd', 'bbox', 'clip_ft', 'text_ft', 'caption', 'inst_color'):
            continue
        if isinstance(obj1[k], (list, int)):
            obj1[k] = obj1[k] + obj2[k]

    obj1['pcd'] = obj1['pcd'] + obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0, 1, 0]

    obj1['clip_ft'] = F.normalize(
        (to_tensor(obj1['clip_ft'], cfg.device) * n1 + to_tensor(obj2['clip_ft'], cfg.device) * n2) / (n1 + n2), dim=0)
    obj1['text_ft'] = F.normalize(
        (to_tensor(obj1['text_ft'], cfg.device) * n1 +
         to_tensor(obj2['text_ft'], cfg.device) * n2) / (n1 + n2), dim=0)

    return obj1


# ── Overlap matrix ────────────────────────────────────────────────────────────

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    m, n = len(objects_map), len(objects_new)
    overlap = np.zeros((m, n))

    pts_map = [np.asarray(o['pcd'].points, np.float32) for o in objects_map]
    indices = [faiss.IndexFlatL2(a.shape[1]) for a in pts_map]
    for idx, arr in zip(indices, pts_map):
        idx.add(arr)
    pts_new = [np.asarray(o['pcd'].points, np.float32) for o in objects_new]

    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new)
    except Exception:
        iou = compute_iou_batch(bbox_map, bbox_new)

    thresh2 = cfg.downsample_voxel_size ** 2
    for i in range(m):
        for j in range(n):
            if iou[i, j] < 1e-6:
                continue
            D, _ = indices[i].search(pts_new[j], 1)
            overlap[i, j] = (D < thresh2).sum() / max(len(pts_new[j]), 1)

    return overlap


def compute_overlap_matrix(cfg, objects: MapObjectList) -> np.ndarray:
    n = len(objects)
    overlap = np.zeros((n, n))

    pts = [np.asarray(o['pcd'].points, np.float32) for o in objects]
    indices = [faiss.IndexFlatL2(a.shape[1]) for a in pts]
    for idx, arr in zip(indices, pts):
        idx.add(arr)

    thresh2 = cfg.downsample_voxel_size ** 2
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if compute_3d_iou(objects[i]['bbox'], objects[j]['bbox']) == 0:
                continue
            D, _ = indices[j].search(pts[i], 1)
            overlap[i, j] = (D < thresh2).sum() / max(len(pts[i]), 1)

    return overlap


# ── Post-processing ───────────────────────────────────────────────────────────

def denoise_objects(cfg, objects: MapObjectList) -> MapObjectList:
    for i in range(len(objects)):
        orig = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(orig, cfg, run_dbscan=True)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = orig
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0, 1, 0]
    return objects


def filter_objects(cfg, objects: MapObjectList) -> MapObjectList:
    kept = [o for o in objects
            if len(o['pcd'].points) >= cfg.obj_min_points
            and o['num_detections'] >= cfg.obj_min_detections]
    return MapObjectList(kept)


def _merge_overlap_objects(cfg, objects: MapObjectList, overlap: np.ndarray) -> MapObjectList:
    xs, ys = overlap.nonzero()
    ratios = overlap[xs, ys]
    order = np.argsort(ratios)[::-1]
    xs, ys, ratios = xs[order], ys[order], ratios[order]

    kept = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(xs, ys, ratios):
        if ratio <= cfg.merge_overlap_thresh:
            break
        vis = F.cosine_similarity(to_tensor(objects[i]['clip_ft'], cfg.device),
                                   to_tensor(objects[j]['clip_ft'], cfg.device), dim=0).item()
        txt = F.cosine_similarity(to_tensor(objects[i]['text_ft'], cfg.device),
                                   to_tensor(objects[j]['text_ft'], cfg.device), dim=0).item()
        if vis > cfg.merge_visual_sim_thresh and txt > cfg.merge_text_sim_thresh and kept[j]:
            objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
            kept[i] = False

    return MapObjectList([o for o, k in zip(objects, kept) if k])


def merge_objects(cfg, objects: MapObjectList) -> MapObjectList:
    if cfg.merge_overlap_thresh > 0:
        overlap = compute_overlap_matrix(cfg, objects)
        objects = _merge_overlap_objects(cfg, objects, overlap)
    return objects
