import torch
import torch.nn.functional as F

from .slam_classes import MapObjectList, DetectionList
from .slam_utils import merge_obj2_into_obj1, compute_overlap_matrix_2set
from .ious import compute_iou_batch, compute_3d_iou_accuracte_batch


def compute_spatial_similarities(cfg, detections: DetectionList, objects: MapObjectList) -> torch.Tensor:
    """M detections × N objects → (M, N) spatial similarity tensor."""
    det_bboxes = detections.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        return compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        mat = compute_overlap_matrix_2set(cfg, objects, detections)  # (N, M)
        return torch.from_numpy(mat).T  # (M, N)
    else:
        return compute_iou_batch(det_bboxes, obj_bboxes)


def compute_visual_similarities(cfg, detections: DetectionList, objects: MapObjectList) -> torch.Tensor:
    """M detections × N objects → (M, N) CLIP cosine similarity tensor."""
    det_fts = detections.get_stacked_values_torch('clip_ft').unsqueeze(-1)   # (M, D, 1)
    obj_fts = objects.get_stacked_values_torch('clip_ft').T.unsqueeze(0).to(det_fts.device)  # (1, D, N)
    return F.cosine_similarity(det_fts, obj_fts, dim=1)                      # (M, N)


def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    return (1 + cfg.phys_bias) * spatial_sim.to(visual_sim.device) + (1 - cfg.phys_bias) * visual_sim


def merge_detections_to_objects(
    cfg,
    detections: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
) -> MapObjectList:
    for i in range(agg_sim.shape[0]):
        if agg_sim[i].max() == float('-inf'):
            objects.append(detections[i])
        else:
            j = int(agg_sim[i].argmax())
            objects[j] = merge_obj2_into_obj1(cfg, objects[j], detections[i], run_dbscan=False)
    return objects
