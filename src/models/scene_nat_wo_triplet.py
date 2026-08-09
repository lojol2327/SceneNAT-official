from typing import *
from torch import Tensor, LongTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks.mtransformer_wo_triplet import MTransformer
from .networks.basic_transformer import BasicTransformerBlock
from .networks.hungarian_matcher import HungarianMatcher
from .networks.set_criterion import SetCriterion
from .utils import *

from einops.layers.torch import Rearrange
from src.utils.logger import StatsLogger


class SceneNAT_wo_triplet(nn.Module):
    def __init__(self,
        node_dim: int,
        n_predicate_types: int,
        objfeat_dim: int=64,
        text_dim: Optional[int]=None,
        t_disc_dim: int=16,
        r_disc_dim: int=8,
        s_disc_dim: int=16,
        loss_weights=None,
        attn_dim=512,
        n_heads=8,
        scene_dec_layers=3,
        triplet_decoder_layers=2,
        max_obj_num=21,
        dropout=0.1,
        masking_probs=None,
        remasking_probs=None,
        predict_pad=True,
        cfg_drop_ratio=0.1,
        max_num_rel: int = 4,
        triplet_context: Optional[str]="text",
        masking_level: Optional[str]="both",
        **kwargs
    ):
        super().__init__()

        self.max_obj_num = max_obj_num
        self.predict_pad = predict_pad
        self.cfg_drop_ratio = cfg_drop_ratio
        self.max_num_rel = max_num_rel
        self.masking_level = masking_level
        self.node_dim = node_dim + 2 # +2 for [Pad] and [mask] token
        self.objfeat_dim = objfeat_dim + 2
        self.t_dim = t_disc_dim + 2
        self.s_dim = s_disc_dim + 2
        self.r_dim = r_disc_dim + 2

        self.text_dim = text_dim

        self.global_condition_embed = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim)
        )

        # === 입력 임베딩: 기저 테이블을 분리해 노출 ===
        self.x_token_embed = nn.Embedding(self.node_dim, attn_dim)
        self.x_embed_proj  = nn.Sequential(nn.SiLU(), nn.Linear(attn_dim, attn_dim))  # 선택적

        # o (4 tokens)
        self.objfeat_token_embed = nn.Embedding(self.objfeat_dim, attn_dim)  # (C_o_all, D)
        self.objfeat_agg = nn.Sequential(                                    # (B,N,4,D)->(B,N,D)
            Rearrange('b n k d -> b n (k d)'),
            nn.SiLU(),
            nn.Linear(attn_dim*4, attn_dim*2),
            nn.SiLU(),
            nn.Linear(attn_dim*2, attn_dim),
        )

        # t/s (각 3축)
        self.t_token_embed = nn.Embedding(self.t_dim, attn_dim)
        self.t_agg = nn.Sequential(
            Rearrange('b n t d -> b n (t d)'),
            nn.SiLU(),
            nn.Linear(attn_dim*3, attn_dim),
        )
        self.s_token_embed = nn.Embedding(self.s_dim, attn_dim)
        self.s_agg = nn.Sequential(
            Rearrange('b n s d -> b n (s d)'),
            nn.SiLU(),
            nn.Linear(attn_dim*3, attn_dim),
        )

        # r (1 token)
        self.r_token_embed = nn.Embedding(self.r_dim, attn_dim)
        self.r_proj = nn.Sequential(nn.SiLU(), nn.Linear(attn_dim, attn_dim))

        self.transformer = MTransformer(
            dim=attn_dim,
            context_dim=text_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_obj_num=max_obj_num,
            scene_dec_layers=scene_dec_layers,
            triplet_decoder_layers=triplet_decoder_layers,
            max_num_rel=max_num_rel,
            n_predicate_types=n_predicate_types,
            obj_class_num=node_dim,
            triplet_context=triplet_context
        )

        # === 출력 디코딩 ===
        # Output normalizations
        self.objfeat_out_norm1 = nn.LayerNorm(attn_dim)
        self.objfeat_out_norm2 = nn.LayerNorm(attn_dim)
        
        self.o_slot_embed = nn.Parameter(torch.randn(4, attn_dim))
        nn.init.normal_(self.o_slot_embed, mean=0.0, std=0.02)

        self.out_objfeat_transformer = nn.ModuleList([
            BasicTransformerBlock(  # self-attn + cross-attn + ff
                attn_dim, attn_dim, attn_dim,
                n_heads, True, dropout
            ) for _ in range(2)
        ])

        self.noise_schedule = cosine_schedule

        self.masking_probs = masking_probs
        self.remasking_probs = remasking_probs
        
        self.loss_weights = loss_weights
        
        # Triplet Decoder를 위한 Matcher와 Criterion
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_s=1.0,
            cost_o=1.0
        )
        
        self.criterion = SetCriterion(
            matcher=self.matcher,
            eos_coef=0.1,
            num_classes={'p': n_predicate_types + 1}
        )

        self.o_mask_id, self.t_mask_id, self.s_mask_id, self.r_mask_id = map(lambda d: d-2, [self.objfeat_dim, self.t_dim, self.s_dim, self.r_dim])
        self.o_pad_id, self.t_pad_id, self.s_pad_id, self.r_pad_id = map(lambda d: d-1, [self.objfeat_dim, self.t_dim, self.s_dim, self.r_dim])

        if not self.predict_pad:
            # Change [PAD] token index += 1 for [MASK] token
            self.x_mask_id = self.node_dim-2
            self.x_pad_id = self.node_dim-1
        else:
            self.x_mask_id = self.node_dim-1
            self.x_pad_id = self.node_dim-2

    def get_pad_ids(self):
        return [self.x_pad_id, self.o_pad_id, self.t_pad_id, self.s_pad_id, self.r_pad_id]
    
    def forward(self,
        sample_params: Dict[str, Tensor],
        text_last_hidden_state: Optional[Tensor]=None,
        text_embeds: Optional[Tensor]=None,
        triple_list: Optional[List[List[Tuple[int, int, int]]]]=None,
        spo_class_list: Optional[List[List[Tuple[int, int, int]]]]=None,
        is_training: Optional[bool]=True,
        predicate_types=None
        ):
        # Unpack sample parameters
        x = sample_params["objs"] #(1,21)
        o = sample_params["objfeat_vq_indices"] #(1, 21, 4)
        device = o.device

        boxes = sample_params["boxes"] #(1, 21, 7)
        t = boxes[..., :3]
        s = boxes[..., 3:6]
        r = boxes[..., 6]

        obj_len = sample_params["lengths"] #(1, 1)
        
        B, N = x.shape[0], x.shape[-1]

        # Prepare mask
        device = o.device
        
        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        non_pad_mask = get_non_pad_mask(B, N, obj_len, device=device)

        # Change [PAD] token index += 1 for [MASK] token
        o, t, s, r = map(lambda t, d: torch.where(t==d-2, d-1, t),[o, t, s, r], [self.objfeat_dim, self.t_dim, self.s_dim, self.r_dim])
        
        if not self.predict_pad:
            # Change [PAD] token index += 1 for [MASK] token
            x = torch.where(x==self.node_dim-2, self.node_dim-1, x)
    
        objects = torch.cat([x.unsqueeze(-1), o, t, s, r.unsqueeze(-1)], dim=-1)
        objects_in, mask, object_mask, object_labels = prepare_masked_input(
            objects, 
            rand_mask_probs, 
            self.x_mask_id, self.o_mask_id, self.t_mask_id, self.s_mask_id, self.r_mask_id,
            self.masking_probs[0], self.remasking_probs[0],
            self.masking_level,
            non_pad_mask=non_pad_mask if not self.predict_pad else None,
        ) # mask = (B, N, 12), object_mask = (B, N, 12)

        x_out, o_out, t_out, s_out, r_out = self._forward(
            objects_in[:,:,0], 
            objects_in[:,:,1:5], 
            objects_in[:,:,5:8], 
            objects_in[:,:,8:11], 
            objects_in[:,:,-1], 
            text_last_hidden_state,
            text_embeds,
            object_mask=~object_mask, # mask된 부분이 1->0
            pad_mask=non_pad_mask,
        )

        x_kl_loss, _, x_acc = cal_performance(x_out, object_labels[:,:,0], mask[:,:,0])
        o_kl_loss, _, o_acc = cal_performance_objfeat(o_out, object_labels[:,:,1:5].reshape(B,-1), mask[:,:,1:5].reshape(-1, N*4))
        t_kl_loss, _, t_dist = cal_performance_dist(t_out, object_labels[:,:,5:8].reshape(B,-1), mask[:,:,5:8].reshape(-1, N*3))
        s_kl_loss, _, s_dist = cal_performance_dist(s_out, object_labels[:,:,8:11].reshape(B,-1), mask[:,:,8:11].reshape(-1, N*3))
        r_kl_loss, _, r_dist = cal_performance_dist(r_out, object_labels[:,:,-1], mask[:,:,-1])
        
        losses, acc = {}, {}
        losses["x_kl"] = x_kl_loss
        losses["o_kl"] = o_kl_loss
        losses["t_kl"] = t_kl_loss
        losses["s_kl"] = s_kl_loss
        losses["r_kl"] = r_kl_loss

        acc["x_acc"] = x_acc
        acc["o_acc_strict"] = o_acc["acc_strict"]
        acc["o_acc_partial"] = o_acc["acc_partial"]
        acc["o_acc_token"] = o_acc["acc_token"]
        acc["t_dist"] = t_dist
        acc["s_dist"] = s_dist
        acc["r_dist"] = r_dist

        assert spo_class_list is not None, "spo_class_list should be provided"
        # triplet_losses = self.criterion(triplet_logits, spo_class_list)
        # for loss_name, loss_value in triplet_losses.items():
            # losses[loss_name] = loss_value
            # try:
            #     StatsLogger.instance()[loss_name].update(loss_value.item() * x.shape[0], x.shape[0])
            # except:
            #     pass
        
        for k, v in losses.items():
            if k in self.loss_weights:
                v = self.loss_weights[k] * v
                losses[k] = v
            try:
                StatsLogger.instance()[k].update(v.item() * x.shape[0], x.shape[0])
            except:  # `StatsLogger` is not initialized
                pass


        return losses, acc

    def _forward(self,
        x: LongTensor, o: LongTensor, t: LongTensor, s: LongTensor, r: LongTensor,
        condition: Tensor,
        global_condition: Tensor,
        object_mask: Optional[LongTensor]=None, 
        pad_mask: Optional[LongTensor]=None,
    ):
        ## 모든 마스크는 가려질 애가 0이어야 함.
        B, N, K = o.shape[0], o.shape[1], o.shape[-1]

        # === 입력 임베딩 (분리된 테이블 사용) ===
        x_emb = self.x_embed_proj(self.x_token_embed(x))        # (B,N,D)

        o_tok = self.objfeat_token_embed(o)                     # (B,N,4,D)
        o_emb = self.objfeat_agg(o_tok)                         # (B,N,D)

        t_tok = self.t_token_embed(t)                           # (B,N,3,D)
        t_emb = self.t_agg(t_tok)                               # (B,N,D)

        s_tok = self.s_token_embed(s)                           # (B,N,3,D)
        s_emb = self.s_agg(s_tok)                               # (B,N,D)

        r_tok = self.r_token_embed(r)                           # (B,N,1,D) 또는 (B,N,D) 입력이면 임베딩 후 squeeze
        if r_tok.dim() == 4:  # (B,N,1,D) -> (B,N,D)
            r_tok = r_tok.squeeze(2)
        r_emb = self.r_proj(r_tok)   

        # === Scene Encoding ===
        token_emb = torch.stack([x_emb, o_emb, t_emb, s_emb, r_emb], dim=2)  # (B,N,5,D)

        condition = torch.cat([
            condition,
            self.global_condition_embed(global_condition).unsqueeze(1)
        ], dim=1)


        xo_feat, t_base, s_base, r_base = self.transformer(
                                token_emb,
                                text_emb=condition,
                                pad_mask=pad_mask,
                            ) # (B, N, 5, attn_dim)

        # === X (node) 출력: weight tying ===
        vocab_x = self.x_token_embed.weight[: self.node_dim - (1 if self.predict_pad else 2)]  # (Vx,D)
        out_x = F.linear(xo_feat, vocab_x)                       # (B,N,Vx)
        out_x = out_x.permute(0, 2, 1)  

        # === O (objfeat 4 tokens) 출력: 입력 o_tok을 질의로 refinement → tying 분류 ===
        q = o_tok + self.o_slot_embed.unsqueeze(0)
        q = self.objfeat_out_norm1(q)
        q = q.reshape(B*N, K, -1)                            # (B*N,4,D)
        for block in self.out_objfeat_transformer:
            q = block(
                q, 
                context=xo_feat.reshape(B*N, 1, -1),
                mask=pad_mask.view(B*N, -1)
            )                            # refine o tokens

        q = self.objfeat_out_norm2(q)                             # (B*N,4,D)
        vocab_o = self.objfeat_token_embed.weight[: self.objfeat_dim - 2]  # (Vo,D)
        out_o = F.linear(q, vocab_o)                             # (B*N,4,Vo)
        out_o = out_o.reshape(B, N*4, -1).permute(0, 2, 1)       # (B,Vo,N*4)

        # === Layout → t/s/r: layout_feat을 3채널로 나눠 헤드 통과 후 tying 분류 ===
        # t: (B,N,3,D)
        t_logits = F.linear(                                     # (B,N,3,Ct)
            t_base.reshape(B*N*3, -1),
            self.t_token_embed.weight[: self.t_dim - 2]
        ).view(B, N, 3, -1)
        out_t = t_logits.permute(0, 3, 1, 2).reshape(B, self.t_dim-2, N*3)  # (B,Ct,N*3)

        # s: (B,N,3,D)
        s_logits = F.linear(
            s_base.reshape(B*N*3, -1),
            self.s_token_embed.weight[: self.s_dim - 2]
        ).view(B, N, 3, -1)
        out_s = s_logits.permute(0, 3, 1, 2).reshape(B, self.s_dim-2, N*3)  # (B,Cs,N*3)

        # r: (B,N,1,D)
        out_r = F.linear(r_base, self.r_token_embed.weight[: self.r_dim - 2])  # (B,N,Cr)
        out_r = out_r.permute(0, 2, 1)                           # (B,Cr,N)

        return out_x, out_o, out_t, out_s, out_r
    
    # default setting: sampling 3_2에 해당
    def generate_samples(self,
                         max_length,
                         text_last_hidden_state: Optional[Tensor]=None,
                         text_embeds: Optional[Tensor]=None,
                         obj_len: Optional[LongTensor]=None,
                         timesteps: int=10,
                         temperature=1,
                         topk_filter_thres=0.9,
                         gsample=False,
                         cfg_scale=1.0,  # classifier-free guidance scale
                         all_timesteps=False,
                         fix_prev=False,
                         att_wise_schedule=True,
                         ):
        device = next(self.parameters()).device
        bs = obj_len.shape[0]

        mask_ids = [self.x_mask_id, self.o_mask_id, self.t_mask_id, self.s_mask_id, self.r_mask_id]
        pad_ids = [self.x_pad_id, self.o_pad_id, self.t_pad_id, self.s_pad_id, self.r_pad_id]

        # Initialize all tokens as [MASK]
        objects_in, token_mask, scores = prepare_all_masked_input(bs, max_length, device, obj_len, mask_ids, pad_ids)
        
        valid_mask = token_mask[:,:,0].clone()
        D = token_mask.shape[-1]
        # ------------------------------------------------------------
        all_timesteps_results = []
        all_timesteps_token_mask = []

        x_mask_counts = token_mask[:,:,0].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        o_mask_counts = token_mask[:,:,1:5].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        t_mask_counts = token_mask[:,:,5:8].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        s_mask_counts = token_mask[:,:,8:11].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        r_mask_counts = token_mask[:,:,-1].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        mask_counts = x_mask_counts + o_mask_counts + t_mask_counts + s_mask_counts + r_mask_counts  # shape: (bs,)

        for t in torch.linspace(0, 1, timesteps, device=device):
            obj_mask = (token_mask.sum(dim=-1) < D)  # (B, L)

            if cfg_scale == 1.0:
                # cfg_scale이 1.0일 때는 conditional prediction만 수행
                x_logits, o_logits, t_logits, s_logits, r_logits, _ = self._forward(
                    objects_in[:,:,0], 
                    objects_in[:,:,1:5], 
                    objects_in[:,:,5:8], 
                    objects_in[:,:,8:11], 
                    objects_in[:,:,-1],
                    text_last_hidden_state,
                    text_embeds,
                    object_mask=obj_mask,
                    pad_mask=valid_mask,
                )
            else:
                # Unconditional prediction
                x_logits_uncond, o_logits_uncond, t_logits_uncond, s_logits_uncond, r_logits_uncond, _ = self._forward(
                    objects_in[:,:,0], 
                    objects_in[:,:,1:5], 
                    objects_in[:,:,5:8], 
                    objects_in[:,:,8:11], 
                    objects_in[:,:,-1], 
                    None,
                    None,
                    object_mask=obj_mask,
                    pad_mask=valid_mask,
                )   
                # Conditional prediction
                x_logits_cond, o_logits_cond, t_logits_cond, s_logits_cond, r_logits_cond, _ = self._forward(
                    objects_in[:,:,0], 
                    objects_in[:,:,1:5], 
                    objects_in[:,:,5:8], 
                    objects_in[:,:,8:11], 
                    objects_in[:,:,-1], 
                    text_last_hidden_state,
                    text_embeds,
                    object_mask=obj_mask,
                    pad_mask=valid_mask,
                )

                # Classifier-free guidance 적용
                x_logits = x_logits_uncond + cfg_scale * (x_logits_cond - x_logits_uncond)
                o_logits = o_logits_uncond + cfg_scale * (o_logits_cond - o_logits_uncond)
                t_logits = t_logits_uncond + cfg_scale * (t_logits_cond - t_logits_uncond)
                s_logits = s_logits_uncond + cfg_scale * (s_logits_cond - s_logits_uncond)
                r_logits = r_logits_uncond + cfg_scale * (r_logits_cond - r_logits_uncond)

            rand_mask_prob = self.noise_schedule(t)
            pred_x_ids, x_score = pred_from_logits_(x_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_o_ids, o_score = pred_from_logits_(o_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_t_ids, t_score = pred_from_logits_(t_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_s_ids, s_score = pred_from_logits_(s_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_r_ids, r_score = pred_from_logits_(r_logits, max_length, gsample, topk_filter_thres, temperature)
            
            pred_ids = torch.cat([pred_x_ids, pred_o_ids, pred_t_ids, pred_s_ids, pred_r_ids], dim=-1)
            new_scores = torch.cat([x_score, o_score, t_score, s_score, r_score], dim=-1)

            objects_in = torch.where(token_mask, pred_ids, objects_in)
            all_timesteps_results.append((
                objects_in[:,:,0].clone(), 
                objects_in[:,:,1:5].clone(), 
                objects_in[:,:,5:8].clone(), 
                objects_in[:,:,8:11].clone(), 
                objects_in[:,:,-1].clone()
            ))
            all_timesteps_token_mask.append(token_mask.clone())

            if fix_prev:
                scores = torch.where(token_mask, new_scores, 1e3)
            else:
                scores = torch.where(token_mask, new_scores, scores)

            if att_wise_schedule:
                x_ids, x_is_mask = remask_w_scores_att_wise(objects_in[:,:,0], scores[:,:,0], rand_mask_prob, x_mask_counts, self.x_mask_id)
                o_ids, o_is_mask = remask_w_scores_att_wise(objects_in[:,:,1:5], scores[:,:,1:5], rand_mask_prob, o_mask_counts, self.o_mask_id)
                t_ids, t_is_mask = remask_w_scores_att_wise(objects_in[:,:,5:8], scores[:,:,5:8], rand_mask_prob, t_mask_counts, self.t_mask_id)
                s_ids, s_is_mask = remask_w_scores_att_wise(objects_in[:,:,8:11], scores[:,:,8:11], rand_mask_prob, s_mask_counts, self.s_mask_id)
                r_ids, r_is_mask = remask_w_scores_att_wise(objects_in[:,:,-1], scores[:,:,-1], rand_mask_prob, r_mask_counts, self.r_mask_id)
                objects_in = torch.cat([x_ids.unsqueeze(-1), o_ids, t_ids, s_ids, r_ids.unsqueeze(-1)], dim=-1)
                token_mask = torch.cat([x_is_mask.unsqueeze(-1), o_is_mask, t_is_mask, s_is_mask, r_is_mask.unsqueeze(-1)], dim=-1)
            else:
                objects_in, token_mask = remask_w_scores_total(objects_in, scores, rand_mask_prob, mask_counts, mask_ids)

        if all_timesteps:
            return all_timesteps_results, all_timesteps_token_mask
        else:
            return [[
                objects_in[:,:,0].clone(), 
                objects_in[:,:,1:5].clone(), 
                objects_in[:,:,5:8].clone(), 
                objects_in[:,:,8:11].clone(), 
                objects_in[:,:,-1].clone()]], all_timesteps_token_mask

# ===========================================================================================================
    def uncond(self,
        max_length,
        obj_len: Optional[LongTensor]=None,
        timesteps: int=10,
        temperature=1,
        topk_filter_thres=0.9,
        gsample=False,
        ):
        device = next(self.parameters()).device
        bs = obj_len.shape[0]

        mask_ids = [self.x_mask_id, self.o_mask_id, self.t_mask_id, self.s_mask_id, self.r_mask_id]
        pad_ids = [self.x_pad_id, self.o_pad_id, self.t_pad_id, self.s_pad_id, self.r_pad_id]

        # Initialize all tokens as [MASK]
        objects_in, token_mask, scores = prepare_all_masked_input(bs, max_length, device, obj_len, mask_ids, pad_ids)
        
        valid_mask = token_mask[:,:,0].clone()
        D = token_mask.shape[-1]

        x_mask_counts = token_mask[:,:,0].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        o_mask_counts = token_mask[:,:,1:5].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        t_mask_counts = token_mask[:,:,5:8].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        s_mask_counts = token_mask[:,:,8:11].reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        r_mask_counts = token_mask[:,:,-1].reshape(bs, -1).sum(dim=1)  # shape: (bs,)

        empty_text_cond = torch.zeros(bs, 77, self.text_dim, device=device)
        empty_text_embeds = torch.zeros(bs, self.text_dim, device=device)

        for t in torch.linspace(0, 1, timesteps, device=device):
            obj_mask = (token_mask.sum(dim=-1) < D)  # (B, L)

            x_logits, o_logits, t_logits, s_logits, r_logits, _ = self._forward(
                objects_in[:,:,0], 
                objects_in[:,:,1:5], 
                objects_in[:,:,5:8], 
                objects_in[:,:,8:11], 
                objects_in[:,:,-1],
                empty_text_cond,
                empty_text_embeds,
                object_mask=obj_mask,
                pad_mask=valid_mask,
            )
            
            rand_mask_prob = self.noise_schedule(t)
            pred_x_ids, x_score = pred_from_logits_(x_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_o_ids, o_score = pred_from_logits_(o_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_t_ids, t_score = pred_from_logits_(t_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_s_ids, s_score = pred_from_logits_(s_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_r_ids, r_score = pred_from_logits_(r_logits, max_length, gsample, topk_filter_thres, temperature)
            
            pred_ids = torch.cat([pred_x_ids, pred_o_ids, pred_t_ids, pred_s_ids, pred_r_ids], dim=-1)
            new_scores = torch.cat([x_score, o_score, t_score, s_score, r_score], dim=-1)

            objects_in = torch.where(token_mask, pred_ids, objects_in)
            scores = torch.where(token_mask, new_scores, scores)

            x_ids, x_is_mask = remask_w_scores_att_wise(objects_in[:,:,0], scores[:,:,0], rand_mask_prob, x_mask_counts, self.x_mask_id)
            o_ids, o_is_mask = remask_w_scores_att_wise(objects_in[:,:,1:5], scores[:,:,1:5], rand_mask_prob, o_mask_counts, self.o_mask_id)
            t_ids, t_is_mask = remask_w_scores_att_wise(objects_in[:,:,5:8], scores[:,:,5:8], rand_mask_prob, t_mask_counts, self.t_mask_id)
            s_ids, s_is_mask = remask_w_scores_att_wise(objects_in[:,:,8:11], scores[:,:,8:11], rand_mask_prob, s_mask_counts, self.s_mask_id)
            r_ids, r_is_mask = remask_w_scores_att_wise(objects_in[:,:,-1], scores[:,:,-1], rand_mask_prob, r_mask_counts, self.r_mask_id)
            objects_in = torch.cat([x_ids.unsqueeze(-1), o_ids, t_ids, s_ids, r_ids.unsqueeze(-1)], dim=-1)
            token_mask = torch.cat([x_is_mask.unsqueeze(-1), o_is_mask, t_is_mask, s_is_mask, r_is_mask.unsqueeze(-1)], dim=-1)

        return [[x_ids, o_ids, t_ids, s_ids, r_ids]]

    def complete_scene(self,
                    batch,
                    mask_object_indices,
                    max_length,
                    text_last_hidden_state: Optional[Tensor]=None,
                    text_embeds: Optional[Tensor]=None,
                    timesteps: int=10,
                    temperature=1,
                    topk_filter_thres=0.9,
                    gsample=False,
                    ):
        device = next(self.parameters()).device
        obj_len=batch["lengths"].to(device)
        bs = obj_len.shape[0]
        boxes = batch["boxes"] #(1, 21, 7)
        valid_mask = get_non_pad_mask(bs, max_length, obj_len, device=device)

        x_ids = batch["objs"]
        o_ids = batch["objfeat_vq_indices"]
        t_ids = boxes[..., :3]
        s_ids = boxes[..., 3:6]
        r_ids = boxes[..., 6]

        x_ids, o_ids, t_ids, s_ids, r_ids = map(lambda t, d: torch.where(t==d-2, d-1, t),[x_ids, o_ids, t_ids, s_ids, r_ids], [self.node_dim, self.objfeat_dim, self.t_dim, self.s_dim, self.r_dim])

        # Force mask positions for specified object indices
        for b in range(bs):
            for idx in mask_object_indices[b]:  # e.g., [3, 5, 7]
                x_ids[b, idx] = self.x_mask_id
                o_ids[b, idx, :] = self.o_mask_id
                t_ids[b, idx, :] = self.t_mask_id
                s_ids[b, idx, :] = self.s_mask_id
                r_ids[b, idx] = self.r_mask_id

        x_is_mask = x_ids == self.x_mask_id
        o_is_mask = o_ids == self.o_mask_id
        t_is_mask = t_ids == self.t_mask_id
        s_is_mask = s_ids == self.s_mask_id
        r_is_mask = r_ids == self.r_mask_id
        #pad 안들어가는거 확인

        x_scores = torch.where(x_is_mask, 0, 1e5)
        o_scores = torch.where(o_is_mask, 0, 1e5).view(bs, -1)
        t_scores = torch.where(t_is_mask, 0, 1e5).view(bs, -1)
        s_scores = torch.where(s_is_mask, 0, 1e5).view(bs, -1)
        r_scores = torch.where(r_is_mask, 0, 1e5)

        x_mask_counts = x_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        o_mask_counts = o_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        t_mask_counts = t_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        s_mask_counts = s_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        r_mask_counts = r_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)

        for t in torch.linspace(0, 1, timesteps, device=device):
            mask = torch.cat([x_is_mask.unsqueeze(-1), o_is_mask, t_is_mask, s_is_mask, r_is_mask.unsqueeze(-1)], dim=-1)
            D = mask.shape[-1]
            obj_mask = (mask.sum(dim=-1) < D)  # (B, L)
            x_logits, o_logits, t_logits, s_logits, r_logits, _ = self._forward(
                x_ids, o_ids, t_ids, s_ids, r_ids, 
                text_last_hidden_state,
                text_embeds,
                object_mask=obj_mask,
                pad_mask=valid_mask,
            )

            rand_mask_prob = self.noise_schedule(t)
            pred_x_ids, x_scores_new = pred_from_logits_(x_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_o_ids, o_scores_new = pred_from_logits_(o_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_t_ids, t_scores_new = pred_from_logits_(t_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_s_ids, s_scores_new = pred_from_logits_(s_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_r_ids, r_scores_new = pred_from_logits_(r_logits, max_length, gsample, topk_filter_thres, temperature)
            
            x_scores = torch.where(x_is_mask.reshape(bs, -1), x_scores_new.reshape(bs, -1), x_scores)
            o_scores = torch.where(o_is_mask.reshape(bs, -1), o_scores_new.reshape(bs, -1), o_scores)
            t_scores = torch.where(t_is_mask.reshape(bs, -1), t_scores_new.reshape(bs, -1), t_scores)
            s_scores = torch.where(s_is_mask.reshape(bs, -1), s_scores_new.reshape(bs, -1), s_scores)
            r_scores = torch.where(r_is_mask.reshape(bs, -1), r_scores_new.reshape(bs, -1), r_scores)

            x_ids = torch.where(x_is_mask, pred_x_ids.squeeze(-1), x_ids)
            o_ids = torch.where(o_is_mask, pred_o_ids, o_ids)
            t_ids = torch.where(t_is_mask, pred_t_ids, t_ids)
            s_ids = torch.where(s_is_mask, pred_s_ids, s_ids)
            r_ids = torch.where(r_is_mask, pred_r_ids.squeeze(-1), r_ids)

            x_ids, x_is_mask = remask_w_scores_att_wise(x_ids, x_scores, rand_mask_prob, x_mask_counts, self.x_mask_id)
            o_ids, o_is_mask = remask_w_scores_att_wise(o_ids, o_scores, rand_mask_prob, o_mask_counts, self.o_mask_id)
            t_ids, t_is_mask = remask_w_scores_att_wise(t_ids, t_scores, rand_mask_prob, t_mask_counts, self.t_mask_id)
            s_ids, s_is_mask = remask_w_scores_att_wise(s_ids, s_scores, rand_mask_prob, s_mask_counts, self.s_mask_id)
            r_ids, r_is_mask = remask_w_scores_att_wise(r_ids, r_scores, rand_mask_prob, r_mask_counts, self.r_mask_id)

        return [[x_ids, o_ids, t_ids, s_ids, r_ids]]

    def layout_to_object(self,
                    batch,
                    mask_object_indices,
                    max_length,
                    text_last_hidden_state: Optional[Tensor]=None,
                    timesteps: int=10,
                    temperature=1,
                    topk_filter_thres=0.9,
                    gsample=False,
                    ):
        device = next(self.parameters()).device
        obj_len=batch["lengths"].to(device)
        bs = obj_len.shape[0]
        boxes = batch["boxes"] #(1, 21, 7)
        valid_mask = get_non_pad_mask(bs, max_length, obj_len, device=device)

        x_ids = batch["objs"]
        o_ids = batch["objfeat_vq_indices"]
        t_ids = boxes[..., :3]
        s_ids = boxes[..., 3:6]
        r_ids = boxes[..., 6]
        
        x_ids, o_ids, t_ids, s_ids, r_ids = map(lambda t, d: torch.where(t==d-2, d-1, t),[x_ids, o_ids, t_ids, s_ids, r_ids], [self.node_dim, self.objfeat_dim, self.t_dim, self.s_dim, self.r_dim])

        for b in range(bs):
            for idx in mask_object_indices[b]:  # e.g., [3, 5, 7]
                x_ids[b, idx] = self.x_mask_id
                o_ids[b, idx, :] = self.o_mask_id

        x_is_mask = x_ids == self.x_mask_id
        o_is_mask = o_ids == self.o_mask_id
        
        x_mask_counts = x_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        o_mask_counts = o_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)

        x_scores = torch.where(x_is_mask, 0, 1e5).view(bs, -1)
        o_scores = torch.where(o_is_mask, 0, 1e5).view(bs, -1)

        obj_mask = torch.ones_like(valid_mask)

        empty_text_cond = torch.zeros(bs, 77, self.text_dim, device=device)
        empty_text_embeds = torch.zeros(bs, self.text_dim, device=device)
        
        for t in torch.linspace(0, 1, timesteps, device=device):
            x_logits, o_logits, _, _, _, _ = self._forward(
                x_ids, o_ids, t_ids, s_ids, r_ids, 
                empty_text_cond,
                empty_text_embeds,
                object_mask=obj_mask,
                pad_mask=valid_mask,
            )

            rand_mask_prob = self.noise_schedule(t)

            pred_x_ids, x_scores_new = pred_from_logits_(x_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_o_ids, o_scores_new = pred_from_logits_(o_logits, max_length, gsample, topk_filter_thres, temperature)
            
            x_ids = torch.where(x_is_mask, pred_x_ids.squeeze(-1), x_ids)
            o_ids = torch.where(o_is_mask, pred_o_ids, o_ids)
            
            x_scores = torch.where(x_is_mask.reshape(bs, -1), x_scores_new.reshape(bs, -1), x_scores)
            o_scores = torch.where(o_is_mask.reshape(bs, -1), o_scores_new.reshape(bs, -1), o_scores)
            
            x_ids, x_is_mask = remask_w_scores_att_wise(x_ids, x_scores, rand_mask_prob, x_mask_counts, self.x_mask_id)
            o_ids, o_is_mask = remask_w_scores_att_wise(o_ids, o_scores, rand_mask_prob, o_mask_counts, self.o_mask_id)

        return [[x_ids, o_ids, t_ids, s_ids, r_ids]]

    def rearrange_scene(self,
                    batch,
                    max_length,
                    text_last_hidden_state: Optional[Tensor]=None,
                    text_embeds: Optional[Tensor]=None,
                    timesteps: int=10,
                    temperature=1,
                    topk_filter_thres=0.9,
                    gsample=False,
                    ):
        print("THIS IS CHANGED VERSION")
        device = next(self.parameters()).device
        obj_len=batch["lengths"].to(device)
        bs = obj_len.shape[0]
        valid_mask = get_non_pad_mask(bs, max_length, obj_len, device=device)

        x_ids = batch["objs"]
        o_ids = batch["objfeat_vq_indices"]
        s_ids = batch["boxes"][..., 3:6]
        
        t_ids, t_is_mask = init_tokens((bs, max_length, 3), self.t_mask_id, self.t_pad_id, device, obj_len)
        r_ids, r_is_mask = init_tokens((bs, max_length), self.r_mask_id, self.r_pad_id, device, obj_len)
        
        x_ids, o_ids, s_ids = map(lambda t, d: torch.where(t==d-2, d-1, t),[x_ids, o_ids, s_ids], [self.node_dim, self.objfeat_dim, self.s_dim])
        
        t_mask_counts = t_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        r_mask_counts = r_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)

        t_scores = torch.where(t_is_mask, 0, 1e5).view(bs, -1)
        r_scores = torch.where(r_is_mask, 0, 1e5)

        obj_mask = torch.ones_like(valid_mask)
        
        for t in torch.linspace(0, 1, timesteps, device=device):
            _, _, t_logits, _, r_logits, _ = self._forward(
                x_ids, o_ids, t_ids, s_ids, r_ids, 
                text_last_hidden_state,
                text_embeds,
                object_mask=obj_mask,
                pad_mask=valid_mask,
            )

            rand_mask_prob = self.noise_schedule(t)
            pred_t_ids, t_scores_new = pred_from_logits_(t_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_r_ids, r_scores_new = pred_from_logits_(r_logits, max_length, gsample, topk_filter_thres, temperature)

            t_ids = torch.where(t_is_mask, pred_t_ids, t_ids)
            r_ids = torch.where(r_is_mask, pred_r_ids.squeeze(-1), r_ids)

            t_scores = torch.where(t_is_mask.reshape(bs, -1), t_scores_new.reshape(bs, -1), t_scores)
            r_scores = torch.where(r_is_mask.reshape(bs, -1), r_scores_new.reshape(bs, -1), r_scores)

            t_ids, t_is_mask = remask_w_scores_att_wise(t_ids, t_scores, rand_mask_prob, t_mask_counts, self.t_mask_id)
            r_ids, r_is_mask = remask_w_scores_att_wise(r_ids, r_scores, rand_mask_prob, r_mask_counts, self.r_mask_id)

        return [[x_ids, o_ids, t_ids, s_ids, r_ids]]
    
    def stylize_scene(self,
                    batch,
                    max_length,
                    text_last_hidden_state: Optional[Tensor]=None,
                    text_embeds: Optional[Tensor]=None,
                    timesteps: int=10,
                    temperature=1,
                    topk_filter_thres=0.9,
                    gsample=False,
                    all_timesteps=False,
                    ):

        print("THIS IS CHANGED VERSION")
        device = next(self.parameters()).device
        obj_len=batch["lengths"].to(device)
        bs = obj_len.shape[0]
        boxes = batch["boxes"] #(1, 21, 7)
        valid_mask = get_non_pad_mask(bs, max_length, obj_len, device=device)

        x_ids = batch["objs"]
        t_ids_gt = boxes[..., :3]
        s_ids_gt = boxes[..., 3:6]
        r_ids_gt = boxes[..., 6]

        o_ids, o_is_mask = init_tokens((bs, max_length, 4), self.o_mask_id, self.o_pad_id, device, obj_len)
        t_ids, t_is_mask = init_tokens((bs, max_length, 3), self.t_mask_id, self.t_pad_id, device, obj_len)
        s_ids, s_is_mask = init_tokens((bs, max_length, 3), self.s_mask_id, self.s_pad_id, device, obj_len)
        r_ids, r_is_mask = init_tokens((bs, max_length), self.r_mask_id, self.r_pad_id, device, obj_len)
        
        x_ids, t_ids_gt, s_ids_gt, r_ids_gt = map(lambda t, d: torch.where(t==d-2, d-1, t),[x_ids, t_ids_gt, s_ids_gt, r_ids_gt], [self.node_dim, self.t_dim, self.s_dim, self.r_dim])
        
        o_mask_counts = o_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        t_mask_counts = t_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        s_mask_counts = s_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)
        r_mask_counts = r_is_mask.reshape(bs, -1).sum(dim=1)  # shape: (bs,)

        o_scores = torch.where(o_is_mask, 0, 1e5).view(bs, -1)
        t_scores = torch.where(t_is_mask, 0, 1e5).view(bs, -1)
        s_scores = torch.where(s_is_mask, 0, 1e5).view(bs, -1)
        r_scores = torch.where(r_is_mask, 0, 1e5)

        obj_mask = torch.ones_like(valid_mask)
        
        x_is_mask = torch.zeros_like(x_ids)

        all_timesteps_token_mask = []
        all_timesteps_results = []

        for t in torch.linspace(0, 1, timesteps, device=device):
            _, o_logits, t_logits, s_logits, r_logits, _ = self._forward(
                x_ids, o_ids, t_ids, s_ids, r_ids, 
                text_last_hidden_state,
                text_embeds,
                object_mask=obj_mask,
                pad_mask=valid_mask,
            )

            rand_mask_prob = self.noise_schedule(t)
            pred_o_ids, o_scores_new = pred_from_logits_(o_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_t_ids, t_scores_new = pred_from_logits_(t_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_s_ids, s_scores_new = pred_from_logits_(s_logits, max_length, gsample, topk_filter_thres, temperature)
            pred_r_ids, r_scores_new = pred_from_logits_(r_logits, max_length, gsample, topk_filter_thres, temperature)
            o_ids = torch.where(o_is_mask, pred_o_ids, o_ids)
            t_ids = torch.where(t_is_mask, pred_t_ids, t_ids)
            s_ids = torch.where(s_is_mask, pred_s_ids, s_ids)
            r_ids = torch.where(r_is_mask, pred_r_ids.squeeze(-1), r_ids)

            mask = torch.cat([x_is_mask.unsqueeze(-1), o_is_mask.clone(), t_is_mask, s_is_mask, r_is_mask.unsqueeze(-1)], dim=-1)
            all_timesteps_token_mask.append(mask.clone())
            if all_timesteps:
                all_timesteps_results.append((x_ids, o_ids.clone(), t_ids, s_ids, r_ids))
            
            o_scores = torch.where(o_is_mask.reshape(bs, -1), o_scores_new.reshape(bs, -1), o_scores)
            t_scores = torch.where(t_is_mask.reshape(bs, -1), t_scores_new.reshape(bs, -1), t_scores)
            s_scores = torch.where(s_is_mask.reshape(bs, -1), s_scores_new.reshape(bs, -1), s_scores)
            r_scores = torch.where(r_is_mask.reshape(bs, -1), r_scores_new.reshape(bs, -1), r_scores)
            o_ids, o_is_mask = remask_w_scores_att_wise(o_ids, o_scores, rand_mask_prob, o_mask_counts, self.o_mask_id)
            t_ids, t_is_mask = remask_w_scores_att_wise(t_ids, t_scores, rand_mask_prob, t_mask_counts, self.t_mask_id)
            s_ids, s_is_mask = remask_w_scores_att_wise(s_ids, s_scores, rand_mask_prob, s_mask_counts, self.s_mask_id)
            r_ids, r_is_mask = remask_w_scores_att_wise(r_ids, r_scores, rand_mask_prob, r_mask_counts, self.r_mask_id)

        if all_timesteps:
            return all_timesteps_results, all_timesteps_token_mask
        else:
            # return [[x_ids, o_ids, t_ids_gt, s_ids, r_ids_gt]], all_timesteps_token_mask
            return [[x_ids, o_ids, t_ids_gt, s_ids_gt, r_ids_gt]], all_timesteps_token_mask
    
 