from torch import Tensor, LongTensor
import torch
import torch.nn.functional as F
import math

from typing import *
from torch.nn import Module
from torch.optim import Optimizer
import numpy as np
import os
from torch.distributions.categorical import Categorical
import torch
import torch.nn.functional as F
from src.data.utils_text import reverse_rel



# ===========================================================================================================
#tmp- you might remove this before submission
def _anneal(T_hi: float, T_lo: float, alpha: float):
    # step 진행률 alpha∈[0,1]에 따라 T_hi -> T_lo로 감소
    return (T_hi ** (1.0 - alpha)) * (T_lo ** alpha)

def _apply_temperature(logits: torch.Tensor, T: float, dim: int):
    if T is None or abs(T - 1.0) < 1e-6:
        return logits
    return logits / max(T, 1e-6)

def _logits_to_logprobs_with_top_p_top_k(
    logits: torch.Tensor,
    dim: int,
    top_p: float = 1.0,
    top_k: int = None
):
    probs = F.softmax(logits, dim=dim)

    # top-p (nucleus)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=dim)
        cutoff = (cumsum > top_p)
        cutoff[..., 0] = False
        keep = torch.zeros_like(probs, dtype=torch.bool).scatter(dim, sorted_idx, ~cutoff)
        probs = probs * keep
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)

    # top-k
    if top_k is not None:
        topk_val, topk_idx = torch.topk(probs, k=top_k, dim=dim)
        zeros = torch.zeros_like(probs)
        probs = zeros.scatter(dim, topk_idx, topk_val)
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)

    return torch.log(probs.clamp(min=1e-8))

def _gumbel_noise_like(x: torch.Tensor):
    u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))

def _sample_from_logprobs(logprobs: torch.Tensor, dim: int, use_gumbel: bool):
    # logprobs: (B, V, L*)
    if use_gumbel:
        z = logprobs + _gumbel_noise_like(logprobs)
        return torch.argmax(z, dim=dim)
    else:
        probs = logprobs.exp()
        # reshape to 2D for multinomial
        B, V, L = probs.shape
        probs_2d = probs.permute(0, 2, 1).reshape(B*L, V)
        idx = torch.multinomial(probs_2d, num_samples=1).squeeze(1)  # (B*L,)
        return idx.view(B, L)  # (B, L)


#tmp- you might remove this before submission
# ===========================================================================================================

# ===========================================================================================================
##  masking utils

# get non_pad_mask
def get_non_pad_mask(bs, max_length, lengths, device=None) -> Tensor:
    mask = torch.arange(max_length, device=device).expand(bs, max_length) < lengths.unsqueeze(1)
    return mask

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def prepare_all_masked_input(
    B, L, device,
    obj_lengths,
    mask_ids,
    pad_ids,
):
    """
    objects: (B, L, 12), token dim: [x(1), o(4), t(3), s(3), r(1)]
    """
    D = 12
    split_sizes = [1, 4, 3, 3, 1]
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(split_sizes), dim=0)[:-1]), device=device)

    objects_in = torch.zeros((B, L, D), device=device, dtype=torch.long)
    token_mask = torch.zeros_like(objects_in, dtype=torch.bool)

    for start, size, mask_id, pad_id in zip(offsets.tolist(), split_sizes, mask_ids, pad_ids):
        mask = get_non_pad_mask(B, L, obj_lengths, device=device)
        mask = mask.unsqueeze(-1).expand(-1, -1, size)
        ids = torch.where(mask, mask_id, pad_id)

        sl = slice(start, start + size)
        token_mask[:, :, sl] = mask
        objects_in[:, :, sl] = ids

    scores = torch.where(token_mask, 0., 1e5)

    return objects_in, token_mask, scores

def prepare_masked_input(
    objects, rand_mask_probs,
    x_mask_id, o_mask_id, t_mask_id, s_mask_id, r_mask_id,
    mask_prob, remask_prob,  # unified probs for all modalities
    masking_level="both", # "both" / "object" / "token"
    non_pad_mask=None
    ):
    """
    objects: (B, L, 12), token dim: [x(1), o(4), t(3), s(3), r(1)]
    """
    B, L, D = objects.shape
    device = objects.device

    split_sizes = [1, 4, 3, 3, 1]
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(split_sizes), dim=0)[:-1]), device=device)
    ids = [x_mask_id, o_mask_id, t_mask_id, s_mask_id, r_mask_id]

    # Step 1: object-level masking ratio p1 ~ Uniform[0, p]
    valid_obj_count = non_pad_mask.sum(dim=1)  # (B,), number of valid objects

    total_tokens = valid_obj_count * D
    target_mask_tokens = (rand_mask_probs * total_tokens).round().clamp(min=1)

    if masking_level == "both":
        p1 = torch.rand(1).item() * rand_mask_probs # original
    elif masking_level == "object":
        p1 = rand_mask_probs # w/o token-level masking (only object masking)
    elif masking_level == "token":
        p1 = torch.zeros_like(rand_mask_probs) # w/o object-level masking (only token masking)
    else:
        raise ValueError(f"Invalid masking level: {masking_level}")

    if masking_level == "token":
        obj_num_masked = (p1*valid_obj_count).round() #for w/o object-level masking
        num_obj_masked_tokens = (p1*valid_obj_count*D).round()  # (B,)#for w/o object-level masking
    else:
        obj_num_masked = (p1*valid_obj_count).round().clamp(min=1)
        num_obj_masked_tokens = (p1*valid_obj_count*D).round().clamp(min=1)  # (B,)

    # Number of remaining tokens to be masked at the token-level
    token_num_masked = (target_mask_tokens - num_obj_masked_tokens).clamp(min=0)  # (B,)


    # Determine masking targets
    rand = torch.rand((B, L), device=device)
    rand[~non_pad_mask] = 2.0  # Padded positions are always excluded
    batch_obj_perm = rand.argsort(dim=-1)  # (B, L)
    obj_mask_idx = batch_obj_perm < obj_num_masked.unsqueeze(-1)  # (B, L)
    object_mask = obj_mask_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)

    # Step 2: token-level masking (for the remaining count, from non-object-masked positions)
    token_num_masked = token_num_masked.long()
    valid_token_mask = ~object_mask  # (B, L, D)
    if non_pad_mask is not None:
        valid_token_mask = valid_token_mask & non_pad_mask.unsqueeze(-1)  # (B, L, D)
    valid_token_flat = valid_token_mask.view(B, -1)  # (B, L*D)

    rand = torch.rand((B, L * D), device=device)
    rand[~valid_token_flat] = 2.0  # Only unmasked elements are candidates (if rand > 1, it's never selected)

    batch_token_perm = rand.argsort(dim=-1)  # (B, L*D)
    token_mask_idx = torch.arange(L * D, device=device).unsqueeze(0).expand(B, -1) < token_num_masked.unsqueeze(1)  # (B, L*D)
    token_mask = torch.zeros_like(valid_token_flat, dtype=torch.bool)  # (B, L*D)
    token_mask.scatter_(dim=1, index=batch_token_perm, src=token_mask_idx)  # (B, L*D)
    token_mask = token_mask.view(B, L, D)

    # Final mask (object-level ∪ token-level)
    final_mask = object_mask | token_mask

    # Step 3: Apply masking (differently for each region)
    objects_in = objects.clone()
    # labels = torch.zeros(B, L * D, device=device, dtype=objects.dtype)
    labels = torch.full_like(objects, fill_value=0)  # (B, L, D)

    for start, size, mask_id in zip(offsets.tolist(), split_sizes, ids):
        sl = slice(start, start + size)
        cur_mask = final_mask[:, :, sl]           # (B, L, size)
        cur_data = objects[:, :, sl]              # (B, L, size)

        # mask_token: broadcasting-safe mask_id tensor
        mask_token = torch.full_like(cur_data, mask_id)  # (B, L, size)

        # Step 1: rand replace 10%
        mask_rid = get_mask_subset_prob(cur_mask, mask_prob)  # (B, L, size)
        rand_id = torch.randint_like(cur_data, high=mask_id)
        masked = torch.where(mask_rid, rand_id, cur_data)

        # Step 2: remask 90% * remask_prob → mask token
        mask_mid = get_mask_subset_prob(cur_mask & ~mask_rid, remask_prob)
        masked = torch.where(mask_mid, mask_token, masked)

        # Apply final masked input
        objects_in[:, :, sl] = masked

        # Create labels
        labels[:, :, sl] = torch.where(cur_mask, cur_data, mask_token)  # (B, L, size)
        # labels[:, L * start : L * (start + size)] = torch.where(cur_mask, cur_data, mask_token).reshape(B, -1)

    return objects_in, final_mask, object_mask[...,0], labels

# ===========================================================================================================
## Helper functions
def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# ===========================================================================================================
## loss functions

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

def index_to_log_onehot(x: LongTensor, num_classes: int):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"

    x_onehot: Tensor = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.shape)))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def to_log(prob):
    log_prob = F.log_softmax(prob.double(), dim=1).float()
    zero_vector = torch.zeros(prob.shape[0], 1, *prob.shape[2:]).type_as(prob) - 70.
    log_prob = torch.cat([log_prob, zero_vector], dim=1)
    log_prob = torch.clamp(log_prob, -70., 0.)
    
    return log_prob

def cal_performance(pred, labels, ignore_mask=None, smoothing=0., tk=1):
    log_labels = index_to_log_onehot(labels, pred.shape[1] + 1)
    log_pred = to_log(pred)

    kl = multinomial_kl(log_labels, log_pred)  # (B, N)
    kl = kl * ignore_mask.float()
    loss = avg_except_batch(kl).mean()
    
    # Convert log_labels to indices for accuracy calculation
    pred_id_k = torch.topk(log_pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(ignore_mask)
    acc = torch.mean(n_correct.float())

    return loss, pred_id, acc

def cal_performance_objfeat(pred, labels, ignore_mask=None, tk=1):
    log_labels = index_to_log_onehot(labels, pred.shape[1] + 1)
    log_pred = to_log(pred)

    kl_o = multinomial_kl(log_labels, log_pred)  # (B, NK)
    mask_region_o = ignore_mask.float()
    kl_o = kl_o * mask_region_o
    kl_loss = avg_except_batch(kl_o).mean()

    # === Prediction ===
    pred_id_k = torch.topk(log_pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]  # (B, NK)

    B, NK = labels.shape
    K = 4
    N = NK // K

    # Reshape
    labels = labels.reshape(B, N, K)
    pred_id = pred_id.reshape(B, N, K)
    mask_region_o = mask_region_o.reshape(B, N, K)

    # === Strict Accuracy (all 4 match) ===
    objfeat_correct = (pred_id == labels).all(dim=2)  # (B, N)
    valid_obj_mask = (mask_region_o.sum(dim=2) == K)    # (B, N)
    strict_acc_masked = (objfeat_correct.float() * valid_obj_mask.float()).sum()
    strict_acc_total = valid_obj_mask.sum() + 1e-6
    acc_strict = strict_acc_masked / strict_acc_total

    # === Partial Accuracy (how many of 4 matched per object) ===
    token_match = (pred_id == labels).float()  # (B, N, K)
    partial_match = token_match.sum(dim=2) / K   # (B, N)
    partial_acc_masked = (partial_match * valid_obj_mask).sum()
    acc_partial = partial_acc_masked / strict_acc_total

    # === Per-token Accuracy ===
    valid_token_mask = mask_region_o.bool()  # (B, N, K)
    total_token = valid_token_mask.sum()
    correct_token = (token_match * valid_token_mask.float()).sum()
    acc_token = correct_token / (total_token + 1e-6)

    # === Masking ratio (number of elements contributing to acc / total number) ===
    acc_mask_stats = {
        "mask_ratio_strict": strict_acc_total / (B * N + 1e-6),
        "mask_ratio_token": total_token / (B * N * K + 1e-6)
    }

    # === Record accuracy
    acc_info = {
        "acc_strict": acc_strict,
        "acc_partial": acc_partial,
        "acc_token": acc_token,
        **acc_mask_stats
    }

    return kl_loss, pred_id, acc_info

def cal_performance_dist(pred, labels, ignore_mask, tk=1, sigma=1.0):
    log_pred = to_log(pred)
    log_labels = index_to_log_onehot(labels, pred.shape[1] + 1)
    kl = multinomial_kl(log_labels, log_pred)  # (B, N)
   
    kl = kl * ignore_mask.float()
    loss = avg_except_batch(kl).mean()

    # L2 distance
    pred_ids = torch.argmax(log_pred, dim=1)  # (B, N)
    l2 = ((pred_ids - labels).float().pow(2)).masked_select(ignore_mask).mean().sqrt()

    return loss, pred_ids, l2

def multinomial_kl(log_prob1: Tensor, log_prob2: Tensor):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl

def avg_except_batch(x: Tensor, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).mean(dim=-1)

# ===========================================================================================================
## Sampling utils
def init_tokens(shape_, mask_id, pad_id, device, obj_len=None):
    if obj_len is not None:
        mask = get_non_pad_mask(shape_[0], shape_[1], obj_len, device=device)
        if len(shape_) == 3:
            mask = mask.unsqueeze(-1).expand(-1, -1, shape_[2])
        ids = torch.where(mask, mask_id, pad_id)
    else:
        ids = torch.full(shape_, mask_id, device=device)
        mask = torch.ones_like(ids, device=device, dtype=torch.bool)
    return ids, mask

def pred_from_logits_(logit, L, gsample, topk_filter_thres, temperature):
    logit = logit.permute(0, 2, 1)
    B = logit.shape[0]
    filtered_logit = top_k(logit, topk_filter_thres, dim=-1)
    if gsample:
        pred_ids = gumbel_sample(filtered_logit, temperature=temperature, dim=-1)
    else:
        # BUG FIX: Apply temperature to logits before sampling
        pred_ids = Categorical(logits = filtered_logit / temperature).sample()

    prob = logit.softmax(dim=-1)
    score = prob.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)

    return pred_ids.reshape(B, L, -1), score.reshape(B, L, -1)

def remask_w_scores_total(ids, score, mask_prob, scale, mask_ids):
    bs = score.shape[0]
    score = score.view(bs, -1)
    num_mask = torch.round(mask_prob * scale)
    # print(f"mask_prob: {mask_prob}, num_mask: {num_mask}/{score[0].numel()}")
    
    sorted_indices = score.argsort(dim=1) # (B,L*12)
    ranks = sorted_indices.argsort(dim=1) # (B,L*12)
    new_mask = (ranks < num_mask.unsqueeze(-1)).reshape(ids.shape)

    split_sizes = [1, 4, 3, 3, 1]
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(split_sizes), dim=0)[:-1]), device=score.device)

    for start, size, mask_id in zip(offsets.tolist(), split_sizes, mask_ids):
        sl = slice(start, start + size)
        is_mask = new_mask[:, :, sl]
        id = ids[:, :, sl]
        id_remask = torch.where(is_mask, mask_id, id)
        ids[:, :, sl] = id_remask
                       
    return ids, new_mask

def remask_w_scores_att_wise(ids, score, mask_prob, scale, mask_id):
    num_mask = torch.round(mask_prob * scale)
    # print(f"mask_prob: {mask_prob}, num_mask: {num_mask}/{score[0].numel()}")
    bs = score.shape[0]
    score = score.reshape(bs, -1)
    sorted_indices = score.argsort(dim=1)
    ranks = sorted_indices.argsort(dim=1)
    new_mask = (ranks < num_mask.unsqueeze(-1)).reshape(ids.shape)

    ids = torch.where(new_mask, mask_id, ids)
    return ids, new_mask

def pred_from_logits_prefix_w_confidence(th_confidence, logit, ids, mask, mask_id, gsample, topk_filter_thres, temperature):
    logit = logit.permute(0, 2, 1)
    filtered_logit = top_k(logit, topk_filter_thres, dim=-1)
    if gsample:
        pred_ids = gumbel_sample(filtered_logit, temperature=temperature, dim=-1)
    else:
        prob = F.softmax(filtered_logit, dim=-1)
        pred_ids = Categorical(prob / temperature).sample()

    ids = torch.where(mask, pred_ids.reshape(mask.shape), ids)

    prob = logit.softmax(dim=-1)
    score = prob.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
    # score = score.masked_fill(~mask.reshape(mask.shape[0], -1), 1e5)

    # threshold로 마스킹
    is_mask = (score < th_confidence).reshape(ids.shape)
    ids_remask = torch.where(is_mask, mask_id, ids)
                       
    return ids, ids_remask, is_mask

def process_model_output(model_output, dataset, vqvae_model, pad_ids=None):
    """Function to post-process model output
    
    Args:
        model_output: The output of the model (step_num, (objs, objfeat_vq_indices, t_ids, s_ids, r_ids))
        dataset: The dataset object
        vqvae_model: The VQ-VAE model
    
    Returns:
        objfeats: (bs, step_num, N, -1) shaped objfeat embeddings
        bbox_params_t: (bs, step_num, N, 7 + n_object_types+1) shaped bbox parameters
    """

    # Stack results from all steps into a single tensor
    objs = torch.stack([step[0] for step in model_output])  # (step_num, bs, N)
    objfeat_vq_indices = torch.stack([step[1] for step in model_output])  # (step_num, bs, N, 4)
    t_ids = torch.stack([step[2] for step in model_output])  # (step_num, bs, N, 3)
    s_ids = torch.stack([step[3] for step in model_output])  # (step_num, bs, N, 3)
    r_ids = torch.stack([step[4] for step in model_output])  # (step_num, bs, N)

    if pad_ids is not None:
        x_pad_id, o_pad_id, t_pad_id, s_pad_id, r_pad_id = pad_ids
        objs = torch.where(objs == x_pad_id, dataset.n_object_types, objs)
        objfeat_vq_indices_rand = torch.randint_like(objfeat_vq_indices, 0, 64)
        # Replace empty token with random token
        objfeat_vq_indices = torch.where(objfeat_vq_indices == o_pad_id, 
                                        objfeat_vq_indices_rand, 
                                        objfeat_vq_indices)
        t_ids = torch.where(t_ids == t_pad_id, 0, t_ids)
        s_ids = torch.where(s_ids == s_pad_id, 0, s_ids)    
        r_ids = torch.where(r_ids == r_pad_id, 0, r_ids)
    
    # Decode objfeat indices to objfeat embeddings
    step_num, bs, N, _ = objfeat_vq_indices.shape
    objfeats = vqvae_model.reconstruct_from_indices(
        objfeat_vq_indices.reshape(step_num*bs*N, -1)
    ).reshape(step_num, bs, N, -1)
    objfeats = objfeats.cpu().numpy()
    
    # Move to CPU
    objs = objs.cpu()
    t_ids = t_ids.cpu()
    s_ids = s_ids.cpu()
    r_ids = r_ids.cpu()

    # Handle padding tokens
    if dataset.discrete:
        t_ids = t_ids.masked_fill(t_ids == dataset.t_disc_dim, 0)
        s_ids = s_ids.masked_fill(s_ids == dataset.s_disc_dim, 0)
        r_ids = r_ids.masked_fill(r_ids == dataset.r_disc_dim, 0)

    # Convert discretized outputs to continuous values
    # flatten from (step_num, bs, N, ...) -> (step_num*bs*N, ...)
    objs_flat = objs.reshape(-1)
    t_ids_flat = t_ids.reshape(-1, 3)
    s_ids_flat = s_ids.reshape(-1, 3)
    r_ids_flat = r_ids.reshape(-1)

    bbox_params = {
        "class_labels": F.one_hot(objs_flat, num_classes=dataset.n_object_types+1).float(),
        "translations": t_ids_flat,
        "sizes": s_ids_flat,
        "angles": r_ids_flat
    }

    # post_process is handled in a flattened state
    boxes = dataset.post_process(bbox_params)
    
    # Reshape the results back to (step_num, bs, N, ...)
    bbox_params_t = torch.cat([
        boxes["class_labels"].reshape(step_num, bs, N, -1),
        boxes["translations"].reshape(step_num, bs, N, 3),
        boxes["sizes"].reshape(step_num, bs, N, 3),
        boxes["angles"].reshape(step_num, bs, N, 1)
    ], dim=-1).numpy()
    
    # Change dimension order: (step_num, bs, ...) -> (bs, step_num, ...)
    objfeats = np.transpose(objfeats, (1, 0, 2, 3))
    bbox_params_t = np.transpose(bbox_params_t, (1, 0, 2, 3))
    
    assert bbox_params_t.shape[-1] == 7 + dataset.n_object_types+1
    
    return objfeats, bbox_params_t

def extract_partial_obj_list_and_remap(obj_list, object_ids):
    # Extract unique indices from all object_ids
    used_indices = []
    for s, p, o in object_ids:
        for idx in [s, o]:
            if idx not in used_indices:
                used_indices.append(idx)

    # Create partial_obj_list
    partial_obj_list = [obj_list[i] for i in used_indices]

    # Index mapping: original index -> partial_obj_list index
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}

    # Create mapped_triples
    mapped_triples = [(index_map[s], p, index_map[o]) for (s, p, o) in object_ids]

    return partial_obj_list, mapped_triples
