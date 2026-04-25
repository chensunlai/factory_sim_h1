import numpy as np
import torch
import open3d as o3d


def compute_3d_iou(bbox1, bbox2, padding=0):
    b1_min = np.asarray(bbox1.get_min_bound()) - padding
    b1_max = np.asarray(bbox1.get_max_bound()) + padding
    b2_min = np.asarray(bbox2.get_min_bound()) - padding
    b2_max = np.asarray(bbox2.get_max_bound()) + padding

    overlap_size = np.maximum(np.minimum(b1_max, b2_max) - np.maximum(b1_min, b2_min), 0.0)
    overlap_vol = np.prod(overlap_size)
    b1_vol = np.prod(b1_max - b1_min)
    b2_vol = np.prod(b2_max - b2_min)
    return overlap_vol / (b1_vol + b2_vol - overlap_vol + 1e-10)


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """bbox1: (M,8,3)  bbox2: (N,8,3)  → (M,N)"""
    b1_min, _ = bbox1.min(dim=1)  # (M,3)
    b1_max, _ = bbox1.max(dim=1)
    b2_min, _ = bbox2.min(dim=1)  # (N,3)
    b2_max, _ = bbox2.max(dim=1)

    b1_min = b1_min.unsqueeze(1)  # (M,1,3)
    b1_max = b1_max.unsqueeze(1)
    b2_min = b2_min.unsqueeze(0)  # (1,N,3)
    b2_max = b2_max.unsqueeze(0)

    inter_vol = torch.prod(torch.clamp(torch.min(b1_max, b2_max) - torch.max(b1_min, b2_min), min=0), dim=2)
    b1_vol = torch.prod(b1_max - b1_min, dim=2)
    b2_vol = torch.prod(b2_max - b2_min, dim=2)
    return inter_vol / (b1_vol + b2_vol - inter_vol + 1e-10)


def compute_3d_iou_accuracte_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Oriented-box IoU. Falls back to axis-aligned if pytorch3d is unavailable."""
    try:
        import pytorch3d.ops as ops
        bbox1 = _expand_3d_box(bbox1, 0.02)
        bbox2 = _expand_3d_box(bbox2, 0.02)
        order = [0, 2, 5, 3, 1, 7, 4, 6]
        _, iou = ops.box3d_overlap(bbox1[:, order].float(), bbox2[:, order].float())
        return iou
    except Exception:
        return compute_iou_batch(bbox1, bbox2)


def _expand_3d_box(bbox: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    center = bbox.mean(dim=1)
    va = bbox[:, 1] - bbox[:, 0]
    vb = bbox[:, 2] - bbox[:, 0]
    vc = bbox[:, 3] - bbox[:, 0]
    for v in (va, vb, vc):
        n = torch.linalg.vector_norm(v, ord=2, dim=1, keepdim=True)
        v[:] = torch.where(n < eps, v / n.clamp(min=1e-9) * eps, v)
    return torch.stack([
        center - va/2 - vb/2 - vc/2,
        center + va/2 - vb/2 - vc/2,
        center - va/2 + vb/2 - vc/2,
        center - va/2 - vb/2 + vc/2,
        center + va/2 + vb/2 + vc/2,
        center - va/2 + vb/2 + vc/2,
        center + va/2 - vb/2 + vc/2,
        center + va/2 + vb/2 - vc/2,
    ], dim=1).to(bbox.device).type(bbox.dtype)


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    inter_areas = np.prod((rb - lt).clip(min=0), axis=2)
    iob1 = inter_areas / (areas[:, None] + 1e-10)
    iob2 = iob1.T
    contained = (iob1 < th2) & (iob2 > th1)
    ci, cj = contained.nonzero()
    mask_sub = mask.copy()
    for i, j in zip(ci, cj):
        mask_sub[i] = mask_sub[i] & (~mask_sub[j])
    return mask_sub
