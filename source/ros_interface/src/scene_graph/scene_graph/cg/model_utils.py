import numpy as np
import torch


def get_sam_segmentation_from_xyxy_batched(sam_model, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """Run ultralytics SAM on a batch of xyxy boxes and return boolean masks (N, H, W)."""
    if len(bboxes) == 0:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)

    results = sam_model(image, bboxes=bboxes.tolist(), verbose=False)

    H, W = image.shape[:2]
    masks = []

    # Ultralytics SAM returns one Result per image; r.masks.data is (N, H, W) for N bboxes
    r = results[0]
    if r.masks is not None and len(r.masks.data) > 0:
        for i in range(len(r.masks.data)):
            m = r.masks.data[i].cpu().numpy().astype(bool)
            if m.shape != (H, W):
                import cv2
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(m)

    # Pad with empty masks if SAM returned fewer masks than bboxes
    while len(masks) < len(bboxes):
        masks.append(np.zeros((H, W), dtype=bool))

    return np.stack(masks[:len(bboxes)], axis=0)
