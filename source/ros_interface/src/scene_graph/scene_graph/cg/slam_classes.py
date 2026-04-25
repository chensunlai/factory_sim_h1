from collections.abc import Iterable
import copy

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from .general_utils import to_numpy, to_tensor


class DetectionList(list):
    def get_values(self, key, idx: int = None):
        if idx is None:
            return [d[key] for d in self]
        return [d[key][idx] for d in self]

    def get_stacked_values_torch(self, key, idx: int = None):
        values = []
        for d in self:
            v = d[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, (o3d.geometry.OrientedBoundingBox,
                               o3d.geometry.AxisAlignedBoundingBox)):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx: int = None):
        return to_numpy(self.get_stacked_values_torch(key, idx))

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.extend(other)
        return new

    def __iadd__(self, other):
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        new = type(self)()
        for i in index:
            new.append(self[i])
        return new

    def slice_by_mask(self, mask: Iterable[bool]):
        new = type(self)()
        for i, m in enumerate(mask):
            if m:
                new.append(self[i])
        return new

    def get_most_common_class(self):
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            classes.append(values[np.argmax(counts)])
        return classes


class MapObjectList(DetectionList):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_similarities(self, new_clip_ft):
        new_clip_ft = to_tensor(new_clip_ft)
        clip_fts = self.get_stacked_values_torch('clip_ft')
        return F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)

    def to_serializable(self):
        out = []
        for obj in self:
            d = copy.deepcopy(obj)
            d['clip_ft'] = to_numpy(d['clip_ft'])
            d['text_ft'] = to_numpy(d['text_ft'])
            d['pcd_np'] = np.asarray(d['pcd'].points)
            d['bbox_np'] = np.asarray(d['bbox'].get_box_points())
            d['pcd_color_np'] = np.asarray(d['pcd'].colors)
            del d['pcd'], d['bbox']
            out.append(d)
        return out

    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList must be empty before loading'
        for s in s_obj_list:
            d = copy.deepcopy(s)
            d['clip_ft'] = to_tensor(d['clip_ft'])
            d['text_ft'] = to_tensor(d['text_ft'])
            d['pcd'] = o3d.geometry.PointCloud()
            d['pcd'].points = o3d.utility.Vector3dVector(d['pcd_np'])
            d['pcd'].colors = o3d.utility.Vector3dVector(d['pcd_color_np'])
            d['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(d['bbox_np']))
            if len(d['pcd_color_np']) > 0:
                d['bbox'].color = d['pcd_color_np'][0]
            del d['pcd_np'], d['bbox_np'], d['pcd_color_np']
            self.append(d)
