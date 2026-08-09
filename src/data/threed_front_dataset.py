from .utils_text import reverse_rel
from .threed_front_dataset_base import *
from torch.utils.data import dataloader



class MTransformer(DatasetDecoratorBase):
    def __init__(self, dataset, 
        t_disc_dim=None, 
        s_disc_dim=None, 
        degree_step=None):
        super().__init__(dataset)
        if t_disc_dim is not None:
            self.discrete = True
            self.t_disc_dim = t_disc_dim
            self.s_disc_dim = s_disc_dim
            self.degree_step = degree_step
            # Calculate r_disc_dim based on degree_step (e.g., 36 for 10-degree steps, 12 for 30-degree steps)
            self.r_disc_dim = int(360 / degree_step)  # Exclude 180° as it's identical to -180°
            # self.check_ranges()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self.max_length

        sample_params_new = {}
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                p = np.copy(v)
                L, C = p.shape
                if self.discrete:
                    if k == "translations":
                        padding = np.full((max_length - L, C), self.t_disc_dim)
                    elif k == "angles":
                        padding = np.full((max_length - L, C), self.r_disc_dim)
                    elif k == "sizes":
                        padding = np.full((max_length - L, C), self.s_disc_dim)
                    sample_params_new[k] = np.vstack([p, padding]).astype(np.int64)
                else:
                    padding = np.zeros((max_length - L, C))
                    sample_params_new[k] = np.vstack([p, padding])

            elif k == "class_labels":
                class_labels = np.copy(v)
                # Delete the start label
                # Represent objectness as the last channel of class label
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1)
                L, C = new_class_labels.shape
                # Pad the end label in the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_new["objs"] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).argmax(axis=-1)  # (n,)
                # Add the number of bounding boxes in the scene
                sample_params_new["length"] = L

            elif k == "relations":
                triples = np.copy(v)
                edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
                for s, p, o in triples:
                    rev_p = self.predicate_types.index(
                        reverse_rel(self.predicate_types[p])
                    )
                    edges[s, o] = p
                    edges[o, s] = rev_p
                uppertri_edges = edges[np.triu_indices(max_length, k=1)]  # (n*(n-1)/2,)
                assert uppertri_edges.shape[0] == max_length * (max_length - 1) // 2
                sample_params_new["edges"] = uppertri_edges

        sample_params_new["scene_uid"] = sample_params["scene_uid"]

        if "descriptions" in sample_params:
            sample_params_new["descriptions"] = sample_params["descriptions"]

        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        # objfeat_rvq_indices = [
        #     np.array(model_info["objfeat_rvq_indices"])
        #     for model_info in models_info
        # ]
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        object_descs = [
            model_info["chatgpt_caption"]
            for model_info in models_info
        ]
        
        # num_quantizer = objfeat_rvq_indices[0].shape[0]
        
        # Canonical > for models_info relabeling
        # openshape_features = np.load(sample_params["openshape_vitg14_path"])
        
        # Permutation augmentation
        if "permutation" in sample_params:
            # objfeat_rvq_indices = [objfeat_rvq_indices[i] for i in sample_params["permutation"]]
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]
        
        # objfeat_rvq_indices = np.stack(objfeat_rvq_indices)  # (n_obj in scene, num_q, n_token)
        # objfeat_rvq_indices_pad = 64 * np.ones([max_length, num_quantizer, objfeat_rvq_indices.shape[1]])
        # objfeat_rvq_indices_pad[:objfeat_rvq_indices.shape[0]] = objfeat_rvq_indices  # pad with new empty indices
        
        objfeat_vq_indices = np.stack(objfeat_vq_indices)  # (n_obj in scene, num_q, n_token)
        objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]])
        objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with new empty indices
        # sample_params_new["objfeat_rvq_indices"] = objfeat_rvq_indices_pad  # (n, q, k)
        sample_params_new["objfeat_vq_indices"] = objfeat_vq_indices_pad  # (n, q, k)
        sample_params_new["object_descs"] = object_descs  # ["a corner side table with a round top", ...]

        return sample_params_new

    def collate_fn(self, samples):
        sample_params_batch = {
            "scene_uids": [],    # str; (bs,)
            "lengths": [],       # LongTensor; (bs,)
            "objs": [],          # LongTensor; (bs, n）
            "edges": [],         # LongTensor; (bs, n*(n-1)//2)
            "boxes": [],         # Tensor; (bs, n, 8)
            "descriptions": [],  # dict; (bs,)
            # "room_masks": [],    # Tensor; (bs, 1, 64, 64)
            # "objfeat_rvq_indices": [],  # LongTensor; (bs, n*k)
            "objfeat_vq_indices": [],  # LongTensor; (bs, n*k)
            "object_descs": [],        # list of strings; (bs,)
            "openshape_features": []   # Tensor; (bs, n', 1280)
        }

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            length = sample_params["length"]
            objs = sample_params["objs"]
            edges = sample_params["edges"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)

            sample_params_batch["scene_uids"].append(scene_uid)
            sample_params_batch["lengths"].append(length)
            sample_params_batch["objs"].append(objs)
            sample_params_batch["edges"].append(edges)
            sample_params_batch["boxes"].append(boxes)

            if "descriptions" in sample_params:
                descriptions = sample_params["descriptions"]
                sample_params_batch["descriptions"].append(descriptions)

            # objfeat_rvq_indices = sample_params["objfeat_rvq_indices"]  # (n, k)
            objfeat_vq_indices = sample_params["objfeat_vq_indices"]  # (n, k)
            # sample_params_batch["objfeat_rvq_indices"].append(objfeat_rvq_indices)  # (n*k,)
            sample_params_batch["objfeat_vq_indices"].append(objfeat_vq_indices)  # (n*k,)

            if "object_descs" in sample_params:
                object_descs = sample_params["object_descs"]  # ["a corner side table with a round top", ...]
                sample_params_batch["object_descs"].append(object_descs)

        # Make torch tensors from the numpy tensors
        for k, v in sample_params_batch.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif k == "openshape_features":
                continue
            elif k == "boxes":
                if self.discrete:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()
                else:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0))
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_batch

# InstructScene
class SG2SC(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type=None):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_new = {}
        for k, v in sample_params.items():
            if k == "class_labels":
                class_labels = np.copy(v)
                class_ids = np.argmax(class_labels, axis=-1).astype(np.int64)
                sample_params_new["objs"] = class_ids

        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]

        sample_params_new["objfeat_vq_indices"] = np.vstack(objfeat_vq_indices)  # (n, k)

        sample_params.update(sample_params_new)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        # Pad the batch to the local maximum number of objects
        sample_params_pad = {
            "scene_uids": [],  # str; (bs,)
            "boxes": [],       # Tensor; (bs, n, 8)
            "objs": [],        # Tensor; (bs, n)
            "edges": [],       # Tensor; (bs, n, n)
            "obj_masks": [],   # LongTensor; (bs, n)
            "objfeat_vq_indices": []  # LongTensor; (bs, n, k)
        }

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            objs = sample_params["objs"]
            triples = sample_params["relations"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)

            sample_params_pad["scene_uids"].append(scene_uid)

            sample_params_pad["objs"].append(np.pad(
                objs, (0, max_length - objs.shape[0]),
                mode="constant", constant_values=self.n_object_types
            ))  # (n,)
            sample_params_pad["boxes"].append(np.pad(
                boxes, ((0, max_length - boxes.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            ))  # (n, 8)
            
            edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
            for s, p, o in triples:
                rev_p = self.predicate_types.index(
                    reverse_rel(self.predicate_types[p])
                )
                edges[s, o] = p
                edges[o, s] = rev_p
            sample_params_pad["edges"].append(edges)

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]
            objfeat_vq_indices_pad = np.random.randint(0, 64, size=(max_length, objfeat_vq_indices.shape[1]))  # TODO: make `64` configurable
            objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with random indices (not really used)
            sample_params_pad["objfeat_vq_indices"].append(objfeat_vq_indices_pad)  # (n, k)

            obj_mask = np.zeros(max_length, dtype=np.int64)  # (n,)
            obj_mask[:sample_params["length"]] = 1
            sample_params_pad["obj_masks"].append(obj_mask)

        # Make torch tensors from the numpy tensors
        for k, v in sample_params_pad.items():
            if k == "scene_uids":
                sample_params_pad[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_pad

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims

class SGDiffusion(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self.max_length

        sample_params_new = {}
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                L, C = p.shape
                sample_params_new[k] = np.vstack([p, np.tile(np.zeros(C)[None, :], [max_length - L, 1])]).astype(np.float32)

            elif k == "class_labels":
                class_labels = np.copy(v)
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1)
                L, C = new_class_labels.shape
                end_label = np.eye(C)[-1]
                
                sample_params_new["objs"] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).argmax(axis=-1)  # (n,)
                # Add the number of bounding boxes in the scene
                sample_params_new["length"] = L

            elif k == "relations":
                triples = np.copy(v)
                edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
                for s, p, o in triples:
                    rev_p = self.predicate_types.index(reverse_rel(self.predicate_types[p]))
                    edges[s, o] = p
                    edges[o, s] = rev_p
                uppertri_edges = edges[np.triu_indices(max_length, k=1)]  # (n*(n-1)/2,)
                assert uppertri_edges.shape[0] == max_length * (max_length - 1) // 2
                sample_params_new["edges"] = uppertri_edges

        sample_params_new["scene_uid"] = sample_params["scene_uid"]
        sample_params_new["descriptions"] = sample_params["descriptions"]

        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [np.array(model_info["objfeat_vq_indices"]) for model_info in models_info]
        object_descs = [model_info["chatgpt_caption"] for model_info in models_info]
        
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)  # (n', k)
        objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]])  # TODO: make `64` configurable
        objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with new empty indices
        sample_params_new["objfeat_vq_indices"] = objfeat_vq_indices_pad  # (n, k)
        objfeats_vq = np.eye(64)[objfeat_vq_indices]  # (n', k, m); TODO: make `64` configurable
        objfeats_vq_pad = np.zeros([max_length, objfeats_vq.shape[1], objfeats_vq.shape[2]])  # (n, k, m)
        objfeats_vq_pad[:objfeats_vq.shape[0]] = objfeats_vq
        sample_params_new["objfeats_vq"] = objfeats_vq_pad * 2. - 1.  # {0, 1} -> {-1, 1}; (n, k, m)
        sample_params_new["object_descs"] = object_descs  # ["a corner side table with a round top", ...]

        return sample_params_new

    def collate_fn(self, samples):
        sample_params_batch = {
            "scene_uids": [],    # str; (bs,)
            "lengths": [],       # LongTensor; (bs,)
            "objs": [],          # LongTensor; (bs, n）
            "edges": [],         # LongTensor; (bs, n*(n-1)//2)
            "boxes": [],         # Tensor; (bs, n, 8)
            "descriptions": [],  # dict; (bs,)
            "objfeat_vq_indices": [],  # LongTensor; (bs, n*k)
            "objfeats_vq": [],         # Tensor; (bs, n, k*m)
            "object_descs": []         # list of strings; (bs,)
        }

        for sample_params in samples:
            sample_params_batch["scene_uids"].append(str(sample_params["scene_uid"]))
            sample_params_batch["lengths"].append(sample_params["length"])
            sample_params_batch["objs"].append(sample_params["objs"])
            sample_params_batch["edges"].append(sample_params["edges"])
            sample_params_batch["descriptions"].append(sample_params["descriptions"])
            sample_params_batch["object_descs"].append(sample_params["object_descs"])
            
            boxes = np.concatenate([sample_params["translations"],
                                    sample_params["sizes"],
                                    sample_params["angles"]], axis=-1)
            objfeat_vq_indices = sample_params["objfeat_vq_indices"]  # (n, k)
            objfeats_vq = sample_params["objfeats_vq"]  # (n, k, m)
            
            sample_params_batch["boxes"].append(boxes)
            sample_params_batch["objfeat_vq_indices"].append(objfeat_vq_indices.reshape(-1))  # (n*k,)
            sample_params_batch["objfeats_vq"].append(objfeats_vq.reshape(objfeats_vq.shape[0], -1))  # (n, k*m)
            
        # Make torch tensors from the numpy tensors
        for k, v in sample_params_batch.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_batch

# DiffuScene
class Diffusion(DatasetDecoratorBase): 
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self._dataset.max_length

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]
        
        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                L, C = p.shape
                sample_params_target[k] = np.vstack([p, np.tile(np.zeros(C)[None, :], [max_length - L, 1])]).astype(np.float32)

            elif k == "class_labels":
                class_labels = np.copy(v)
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1) #hstack
                L, C = new_class_labels.shape
                end_label = np.eye(C)[-1]
                
                sample_params_target[k] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).astype(np.float32) * 2.0 - 1.0
            
        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [np.array(model_info["objfeat_vq_indices"]) for model_info in models_info]
        object_descs = [model_info["chatgpt_caption"] for model_info in models_info]
        
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)  # (n', k)
        objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]])  # TODO: make `64` configurable
        objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices  # pad with new empty indices
        sample_params_target["objfeat_vq_indices"] = objfeat_vq_indices_pad  # (n, k)
        
        objfeats_vq = np.eye(64)[objfeat_vq_indices]  # (n', k, m); TODO: make `64` configurable
        objfeats_vq_pad = np.zeros([max_length, objfeats_vq.shape[1], objfeats_vq.shape[2]])  # (n, k, m)
        objfeats_vq_pad[:objfeats_vq.shape[0]] = objfeats_vq 
        
        # {0, 1} -> {-1, 1} 
        sample_params_target["objfeats_vq"] = objfeats_vq_pad * 2. - 1.  # (n, k*m) (4*64)
        sample_params_target["object_descs"] = object_descs  # ["a corner side table with a round top", ...]
        
        sample_params.update(sample_params_target)

        return sample_params
    
    def collate_fn(self, samples):
        sample_params_batch = {
            "class_labels": [],         # LongTensor; (bs, n, 22）
            "translations": [],         # LongTensor; (bs, n, 3)
            "sizes": [],                # Tensor; (bs, n, 3)
            "angles": [],               # Tensor; (bs, 12, 2)
            "objfeat_vq_indices": [],   # LongTensor; (bs, n*k)
            "objfeats_vq": [],          # Tensor; (bs, n, k*m)
            "lengths": [],              # LongTensor; (bs,)
            "scene_uids": [],           # str; (bs,)
            "descriptions": [],         # dict; (bs,)
            "object_descs": []          # list of strings; (bs,)
        }
        for sample_params in samples:
            sample_params_batch["class_labels"].append(sample_params["class_labels"])
            sample_params_batch["translations"].append(sample_params["translations"])
            sample_params_batch["sizes"].append(sample_params["sizes"])
            sample_params_batch["angles"].append(sample_params["angles"])
            sample_params_batch["lengths"].append(sample_params["length"])
            sample_params_batch["scene_uids"].append(str(sample_params["scene_uid"]))
            sample_params_batch["descriptions"].append(sample_params["descriptions"])
            sample_params_batch["object_descs"].append(sample_params["object_descs"])

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]  # (n, k)
            objfeats_vq = sample_params["objfeats_vq"]  # (n, k, m)

            sample_params_batch["objfeat_vq_indices"].append(objfeat_vq_indices.reshape(-1))  # (n*k,)
            sample_params_batch["objfeats_vq"].append(objfeats_vq.reshape(objfeats_vq.shape[0], -1))  # (n, k*m)
    
        # Make torch tensors from the numpy tensors
        for k, v in sample_params_batch.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()

        return sample_params_batch

    @property
    def bbox_dims(self):
        return 7

# ATISS
class Autoregressive(DatasetDecoratorBase): 
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        
        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]
        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles", "objfeats"]:
                p = np.copy(v)
                L, C = p.shape # L= number of objects in this scene, C=(trans,size=3, angles=1)
                sample_params_target[k+"_tr"] = np.vstack([p, np.zeros(C)]) #(L,C) -> (L+1,C) end_label added as last row (zero matrix)
            
            elif k == "class_labels":
                class_labels = np.copy(v)
                L, C = class_labels.shape # L= number of objects in this scene, C=number of classes
                end_label = np.eye(C)[-1]
                sample_params_target[k+"_tr"] = np.vstack([class_labels, end_label]) #(L,C) -> (L+1,C) end_label added as last row (zero matrix with last class=1)
                
        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [np.array(model_info["objfeat_vq_indices"]) for model_info in models_info]
        object_descs = [model_info["chatgpt_caption"] for model_info in models_info]
        
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)  # (L, C=k) L=number of objects, k=4(codebook order)
        sample_params_target["objfeat_vq_indices"] = objfeat_vq_indices
        # sample_params_target["objfeat_vq_indices_tr"] = np.vstack([objfeat_vq_indices, np.full((1, 4), 64)]) # (L,4)->(L+1,4) (64,64,64,64) would be better? 
        sample_params_target["objfeat_vq_indices_tr"] = np.vstack([objfeat_vq_indices, np.zeros((1, 4))]) # end token not needed.
        sample_params_target["object_descs"] = object_descs 
        
        # objfeats_vq = np.eye(64)[objfeat_vq_indices]  # (L, k, 64) << need to check if end token should be added here
        # sample_params_target["objfeats_vq"] = objfeats_vq  # ["a corner side table with a round top", ...]
        # objfeats_vq = np.eye(65)[sample_params_target["objfeat_vq_indices_tr"]]  # (L+1, k, 65)
        # sample_params_target["objfeats_vq_tr"] = objfeats_vq  # Save one-hot encoded features

        sample_params.update(sample_params_target)

        return sample_params

    def collate_fn(self, samples):
        sample_params_batch = {
            "angles_tr": [],                # Tensor; (bs, 1)
            "sizes_tr":[],                  # Tensor; (bs, 3)
            "translations_tr":[],           # Tensor; (bs, 3)
            "class_labels_tr":[],           # Tensor; (bs, 23) #bedroom
            "objfeat_vq_indices_tr":[],     # Tensor; (bs, 4) 
            "angles":[],                    # Tensor; (bs, l, 1)
            "sizes":[],                     # Tensor; (bs, l, 3)
            "translations":[],              # Tensor; (bs, l, 3)
            "class_labels":[],              # Tensor; (bs, l, 23)
            "objfeat_vq_indices": [],       # Tensor; (bs, l, 4)
            "lengths":[],                   # Tensor; (bs)
            "scene_uids": [],               # str; (bs,)
            "descriptions": [],             # dict; (bs,)
            "object_descs": []              # list of strings; (bs,)
        }
        
        # Initialize containers
        max_length = max(sample["length"] for sample in samples)
        key_set = set(samples[0].keys()) - set(["length", "scene_uid", "descriptions", "object_descs"])
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)

        for sample_params in samples:
            sample_params_batch["lengths"].append(sample_params["length"])
            sample_params_batch["scene_uids"].append(str(sample_params["scene_uid"]))
            sample_params_batch["descriptions"].append(sample_params["descriptions"])
            sample_params_batch["object_descs"].append(sample_params["object_descs"])
            
            for k in (key_set-padding_keys):
                sample_params_batch[k].append(sample_params[k])
            for k in padding_keys:
                if max_length - len(sample_params[k])>0:
                    sample_params_batch[k].append(np.vstack([sample_params[k], np.zeros((max_length - len(sample_params[k]), sample_params[k].shape[1]))]))
                else:
                    sample_params_batch[k].append(sample_params[k].reshape(sample_params[k].shape[0], -1))
                
        # Convert lists to tensors
        for k, v in sample_params_batch.items():
            if k == "lengths":
                sample_params_batch[k] = torch.tensor(v, dtype=torch.float32)
            elif k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif "vq" in k:
                if "_tr" not in k:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()
                else:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()[:, None]
            else:
                if "_tr" not in k:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()
                else:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()[:, None]

        return sample_params_batch

    @property
    def bbox_dims(self):
        return 7

class AutoregressiveWOCM(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)
        sample_params_new = {}

        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape
        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k in ["scene_uid", "descriptions", "object_descs"]:
                sample_params_new[k] = v
            elif k in ["class_labels", "translations", "sizes", "angles", "objfeat_vq_indices"]:
                sample_params_new[k] = v[:n_boxes]
            else:
                if "_tr" in k:
                    sample_params_new[k] = v[n_boxes]
                else:
                    pass
        sample_params_new["length"] = n_boxes

        return sample_params_new

# ATISS
class Autoregressive_comp(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        
        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]
        sample_params_target = {}
       
        # Load information file for every object
        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [np.array(model_info["objfeat_vq_indices"]) for model_info in models_info]
        object_descs = [model_info["chatgpt_caption"] for model_info in models_info]
        
        # Permutation augmentation
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)  # (L, C=k) L=number of objects, k=4(codebook order)
        sample_params_target["objfeat_vq_indices"] = objfeat_vq_indices
        sample_params_target["object_descs"] = object_descs 

        sample_params.update(sample_params_target)

        return sample_params

    def collate_fn(self, samples):
        sample_params_batch = {
            "angles":[],                    # Tensor; (bs, l, 1)
            "sizes":[],                     # Tensor; (bs, l, 3)
            "translations":[],              # Tensor; (bs, l, 3)
            "class_labels":[],              # Tensor; (bs, l, 23)
            "objfeat_vq_indices": [],       # Tensor; (bs, l, 4)
            "lengths":[],                   # Tensor; (bs)
            "scene_uids": [],               # str; (bs,)
            "descriptions": [],             # dict; (bs,)
            "object_descs": []              # list of strings; (bs,)
        }
        
        # Initialize containers
        max_length = self.max_length
        key_set = set(samples[0].keys()) - set(["length", "scene_uid", "descriptions", "object_descs"])
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)

        for sample_params in samples:
            sample_params_batch["lengths"].append(sample_params["length"])
            sample_params_batch["scene_uids"].append(str(sample_params["scene_uid"]))
            sample_params_batch["descriptions"].append(sample_params["descriptions"])
            sample_params_batch["object_descs"].append(sample_params["object_descs"])
            
            for k in (key_set-padding_keys):
                sample_params_batch[k].append(sample_params[k])
            for k in padding_keys:
                sample_params_batch[k].append(np.vstack([sample_params[k], np.zeros((max_length - len(sample_params[k]), sample_params[k].shape[1]))]))
                
        # Convert lists to tensors
        for k, v in sample_params_batch.items():
            if k == "lengths":
                sample_params_batch[k] = torch.tensor(v, dtype=torch.long)
            elif k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif "vq" in k:
                    sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()

        return sample_params_batch

    @property
    def bbox_dims(self):
        return 7
    
class AutoregressiveWOCM_COMP(Autoregressive_comp):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)
        sample_params_new = {}

        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape

        for k, v in sample_params.items():
            if k in ["scene_uid", "descriptions", "object_descs"]:
                sample_params_new[k] = v
            elif k in ["class_labels", "translations", "sizes", "angles", "objfeat_vq_indices"]:
                sample_params_new[k] = v
            else:
                pass
                
        sample_params_new["length"] = L

        return sample_params_new

################################################################


## Dataset encoding API
def dataset_encoding_factory(
    name,
    dataset,
    config=None,
    augmentations=None,
    no_mask=False # for atiss completion task- no_mask=True makes dataset to not slice data
) -> DatasetDecoratorBase:
    dataset_collection = CachedDatasetCollection(dataset)
    
    if isinstance(augmentations, list):
        for aug_type in augmentations:
            # if aug_type == "rotation":
                # print("Apply [rotation] augmentation")
                # dataset_collection = Rotation(dataset_collection)
            # elif aug_type == "fixed_rotation":
            # The above case causes an error
            print("Applying [fixed rotation] augmentation")
            dataset_collection = Rotation(dataset_collection, fixed=True)

    dataset_collection = Add_SceneGraph(dataset_collection)
    dataset_collection = Add_Description(dataset_collection, seed=None)

    # Scale the input
    print(f"Scale {list(dataset_collection.bounds.keys())}")
    # DiffuScene
    if "cosin_angle" in name:
        dataset_collection = Scale_CosinAngle_ObjfeatsNorm(dataset_collection)
        
    # InstructScene-sg2sc, Ours (continuous)
    elif "sincos_angle" in name:
        print("Use [cos, sin] for angle encoding")
        dataset_collection = Scale_CosinAngle(dataset_collection)
    
    # Ours (discrete)
    elif "discrete" in name:
        dataset_collection = Scale_Disc_Deg(dataset_collection,
                                          t_disc_dim=config["t_disc_dim"],
                                          s_disc_dim=config["s_disc_dim"],
                                          degree_step=config["degree_step"])
    # InstructScene-sg, ATISS
    else:
        dataset_collection = Scale(dataset_collection)
        
    if "eval" not in name:
        permute_keys = ["class_labels", "translations", "sizes", "angles", "relations", "descriptions"]
        dataset_collection = Permutation(dataset_collection, permute_keys,)
    ################################################################
    
    if "atiss" in name:
        if no_mask:
            return AutoregressiveWOCM_COMP(dataset_collection)
        else:
            return AutoregressiveWOCM(dataset_collection)
    
    if "diffuscene" in name:
        return Diffusion(dataset_collection)
        
    elif "sg2sc" in name:
        return SG2SC(dataset_collection)

    elif "sgdiffusion" in name:
        return SGDiffusion(dataset_collection)
    elif "mtrans" in name:
        if "discrete" in name:
            return MTransformer(dataset_collection,
                            t_disc_dim=config["t_disc_dim"],
                            s_disc_dim=config["s_disc_dim"],
                            degree_step=config["degree_step"])
        else:
            return MTransformer(dataset_collection)
    else:
        raise NotImplementedError()


