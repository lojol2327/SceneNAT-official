import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict


class HungarianMatcher(nn.Module):
    """
    Vectorized Hungarian Matcher for bipartite matching between predicted and ground truth triplets
    Removes batch for loops and processes entire batch at once
    """
    def __init__(self, 
                 cost_class: float = 1.0, 
                 cost_s: float = 1.0, 
                 cost_o: float = 1.0):
        super().__init__()
        self.cost_class = cost_class  # predicate cost weight
        self.cost_s = cost_s         # subject cost weight
        self.cost_o = cost_o         # object cost weight
        
    @torch.no_grad()
    def forward(self, pred_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                gt_triplets: List[List[Tuple[int, int, int]]]):
        """
        Args:
            pred_logits: (s_logits, p_logits, o_logits) tuple
                s_logits: (B, N_pred, max_obj_num + 1)
                p_logits: (B, N_pred, n_predicate_types + 1)
                o_logits: (B, N_pred, max_obj_num + 1)
            gt_triplets: List[List[Tuple[int, int, int]]] - GT triplets per batch
        
        Returns:
            indices: List of (pred_idx, gt_idx) pairs for each batch
                    Role separation: unmatched prediction handling is handled by Criterion
        """
        s_logits, p_logits, o_logits = pred_logits
        B, N_pred, _ = p_logits.shape  # B=batch size, N_pred=number of predictions(max_num_rel)
        
        indices = []
        
        for b in range(B):
            target_triplets = gt_triplets[b]  # List of (subj, pred, obj)
            num_targets = len(target_triplets)
            
            if num_targets == 0:
                # If no GT triplets, return empty matching
                indices.append((torch.empty(0, dtype=torch.int64, device=p_logits.device), 
                              torch.empty(0, dtype=torch.int64, device=p_logits.device)))
                continue
            
            # Convert GT triplets to tensor
            s_gt = torch.tensor([t[0] for t in target_triplets], device=p_logits.device)
            p_gt = torch.tensor([t[1] for t in target_triplets], device=p_logits.device)
            o_gt = torch.tensor([t[2] for t in target_triplets], device=p_logits.device)
            
            # Calculate vectorized cost matrix
            # 1. Predicate cost: negative log-likelihood for correct class
            prob_p = p_logits[b].softmax(-1)  # (N_pred, n_predicate_types + 1)
            cost_p = -torch.gather(prob_p, 1, p_gt.unsqueeze(0).expand(N_pred, -1))  # (N_pred, num_targets)
            
            # 2. Subject cost
            prob_s = s_logits[b].softmax(-1)  # (N_pred, max_obj_num + 1)
            cost_s = -torch.gather(prob_s, 1, s_gt.unsqueeze(0).expand(N_pred, -1))  # (N_pred, num_targets)
            
            # 3. Object cost
            prob_o = o_logits[b].softmax(-1)  # (N_pred, max_obj_num + 1)
            cost_o = -torch.gather(prob_o, 1, o_gt.unsqueeze(0).expand(N_pred, -1))  # (N_pred, num_targets)
            
            # 4. Calculate final cost matrix
            C = self.cost_class * cost_p + self.cost_s * cost_s + self.cost_o * cost_o
            
            # 5. Find optimal matching using Hungarian algorithm
            pred_idx, gt_idx = linear_sum_assignment(C.cpu())
            
            # 6. Convert matching results to tensor and return
            #    Role separation: unmatched prediction handling is handled by Criterion
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64, device=p_logits.device),
                torch.as_tensor(gt_idx, dtype=torch.int64, device=p_logits.device)
            ))
        
        return indices