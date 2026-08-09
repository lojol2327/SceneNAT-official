# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

from .utils_text import reverse_rel, rotate_rel


class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset
        self.discrete = False
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    ################################ For InstructScene BEGIN ################################

    # Get the number of predicate types in scene graphs
    @property
    def n_predicate_types(self):
        return self._dataset.n_predicate_types

    # Get the predicate types in scene graphs
    @property
    def predicate_types(self):
        return self._dataset.predicate_types

    # Get the max input length for diffusion models
    @property
    def max_length(self):
        return self._dataset.max_length 

    ################################ For InstructScene END ################################

    def post_process(self, s):
        return self._dataset.post_process(s)

class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class Rotation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0

    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = Rotation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        sample_params["aug_angle"] = rot_angle  # for check in `Add_SceneGraph`
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)

            elif k == "angles":
                angle_min, _ = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min

        return sample_params

class Scale(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in bounds:
                sample_params[k] = Scale.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k in bounds:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_Disc_Deg(DatasetDecoratorBase):
    def __init__(self, dataset, t_disc_dim=64, s_disc_dim=64, degree_step=10):
        super().__init__(dataset)
        self.t_disc_dim = t_disc_dim
        self.s_disc_dim = s_disc_dim
        self.degree_step = degree_step
        # Calculate r_disc_dim based on degree_step (e.g., 36 for 10-degree steps, 12 for 30-degree steps)
        self.r_disc_dim = int(360 / degree_step)  # Exclude 180° as it's identical to -180°

    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x
    
    @staticmethod
    def rad_to_deg(x, minimum, maximum):
        X = np.clip(x, minimum, maximum)
        return X * 180 / np.pi

    @staticmethod
    def deg_to_rad(x):
        return x * np.pi / 180

    @staticmethod
    def cont_to_disc(x, disc_dim, degree_step=None):
        if degree_step is not None:
            # Discretize angles
            num_bins = int(360 / degree_step)  # Exclude 180° as it's identical to -180°
            bin_centers = np.linspace(-180, 180, num_bins, endpoint=False)
            x_ids = np.argmin(np.abs(x[:, :, None] - bin_centers[None, None, :]), axis=-1)
        else:
            # General discretization (translations, sizes)
            bin_width = 2 / disc_dim
            x_ids = np.clip(np.rint((x + 1) / bin_width - 0.5), 0, disc_dim - 1).astype(np.int64)
        return x_ids

    @staticmethod
    def disc_to_cont(x_ids, disc_dim, degree_step=None):
        if degree_step is not None:
            # Restore angles
            bin_centers = torch.arange(-180, 180, degree_step, device=x_ids.device, dtype=torch.float32)
            x_cont = bin_centers[x_ids]
        else:
            # General restoration (translations, sizes)
            bin_width = 2 / disc_dim
            x_cont = (x_ids.float() + 0.5) * bin_width - 1
        return x_cont
    
    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                v_deg = self.rad_to_deg(v, bounds[k][0], bounds[k][1])
                sample_params[k] = self.cont_to_disc(v_deg, self.r_disc_dim, self.degree_step)
            elif k == "translations":
                v_norm = self.scale(v, bounds[k][0], bounds[k][1])
                sample_params[k] = self.cont_to_disc(v_norm, self.t_disc_dim)
            elif k == "sizes":
                v_norm = self.scale(v, bounds[k][0], bounds[k][1])
                sample_params[k] = self.cont_to_disc(v_norm, self.s_disc_dim)
            elif k in bounds:
                sample_params[k] = self.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k == "angles":
                v_deg = self.disc_to_cont(v, self.r_disc_dim, self.degree_step)
                sample_params[k] = self.deg_to_rad(v_deg)
            elif k == "translations":
                v_cont = self.disc_to_cont(v, self.t_disc_dim)
                sample_params[k] = self.descale(v_cont, bounds[k][0], bounds[k][1])
            elif k == "sizes":
                v_cont = self.disc_to_cont(v, self.s_disc_dim)
                sample_params[k] = self.descale(v_cont, bounds[k][0], bounds[k][1])
            elif k in bounds:
                sample_params[k] = self.descale(v, bounds[k][0], bounds[k][1])
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_CosinAngle(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k == "angles":
                sample_params[k] = np.arctan2(v[..., 1:2], v[..., 0:1])

            elif k in bounds:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2

class Scale_CosinAngle_ObjfeatsNorm(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                # [cos, sin]
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k == "objfeats" or k == "objfeats_32": #ys- check. Only Diffuscene comes here.
                sample_params[k] = Scale.scale(v, bounds[k][1], bounds[k][2])
            
            elif k in bounds:
                sample_params[k] = Scale.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "class_labels" or k == "objfeats" or k == "relations" or k == "description":
                sample_params[k] = v
                
            elif k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
                
            else:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])
        sample_params["permutation"] = ordering

        for k in self._permutation_keys:
            if k not in sample_params:
                continue

            ################################ For InstructScene BEGIN ################################

            if k == "relations":
                if sample_params[k].shape[0] > 0:
                    idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                    sample_params[k][:, 0] = np.vectorize(idx_mapping.get)(sample_params[k][:, 0])
                    sample_params[k][:, 2] = np.vectorize(idx_mapping.get)(sample_params[k][:, 2])
            elif k == "descriptions":
                sample_params[k]["obj_class_ids"] = [sample_params[k]["obj_class_ids"][i] for i in ordering]
                idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                for i in range(len(sample_params[k]["obj_relations"])):
                    s, p, o = sample_params[k]["obj_relations"][i]
                    s_new, o_new = idx_mapping[s], idx_mapping[o]
                    sample_params[k]["obj_relations"][i] = (s_new, p, o_new)

            ################################ For InstructScene END ################################

            else:
                sample_params[k] = sample_params[k][ordering]
        return sample_params

################################ For InstructScene BEGIN ################################

class Add_SceneGraph(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        sample_params["relations"] = np.load(sample_params["relation_path"], allow_pickle=True)

        if "aug_angle" in sample_params:
            for i in range(len(sample_params["relations"])):
                p = sample_params["relations"][i, 1]
                sample_params["relations"][i, 1] = self.predicate_types.index(
                    rotate_rel(self.predicate_types[p], sample_params["aug_angle"]))
        return sample_params

class Add_Description(DatasetDecoratorBase):
    def __init__(self, dataset, seed=None):
        super().__init__(dataset)
        self.seed = seed

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        with open(sample_params["description_path"], 'rb') as f:
            descriptions = pickle.load(f)

        if "aug_angle" in sample_params:
            for i in range(len(descriptions["obj_relations"])):
                s_class_id, p, o_class_id = descriptions["obj_relations"][i]
                descriptions["obj_relations"][i] = (
                    int(s_class_id),
                    int(self.predicate_types.index(rotate_rel(self.predicate_types[p], sample_params["aug_angle"]))),
                    int(o_class_id))
        sample_params["descriptions"] = descriptions

        return sample_params

## Helper functions
def trs_to_corners(t: np.ndarray, r: float, s: np.ndarray) -> np.ndarray:
    """Get the corners of the bounding box from the translation, rotation and size."""
    # Points in `template` are in the same order as `trimesh`,
    # which is used in `ThreedFutureModel` for loading corners
    template = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]
    ])
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(r)
    R[0, 2] = -np.sin(r)
    R[2, 0] = np.sin(r)
    R[2, 2] = np.cos(r)
    R[1, 1] = 1.

    return (template * s).dot(R) + t

################################ For InstructScene END ################################


