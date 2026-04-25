import numpy as np
import torch


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(array, device=None):
    if isinstance(array, torch.Tensor):
        return array if device is None else array.to(device)
    t = torch.from_numpy(np.asarray(array))
    return t if device is None else t.to(device)
