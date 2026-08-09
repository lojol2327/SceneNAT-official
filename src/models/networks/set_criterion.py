import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class SetCriterion(nn.Module):
    """
    Vectorized Set Criterion for computing triplet prediction loss
    Removes batch for loops and processes entire batch at once
    """
    def __init__(self, 
                 matcher,
                 eos_coef: float = 0.1,
                 num_classes: dict = None):
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        
        # Weight tensor for Predicate loss (low weight for no-relation class)
        if num_classes is not None:
            p_weight = torch.ones(num_classes['p'])
            p_weight[-1] = self.eos_coef  # no-relation class
            self.register_buffer('p_weight', p_weight)
        
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
            losses: dict of loss values
        """
        s_logits, p_logits, o_logits = pred_logits
        B, N_pred, _ = p_logits.shape
        
        # 1. Bipartite matching
        indices = self.matcher(pred_logits, gt_triplets)
        
        # 2. Vectorized Loss calculation
        losses = {}
        
        # Predicate loss: calculate for all predictions (matched ones use actual GT, others use no-relation)
        losses['loss_p'] = self._compute_predicate_loss_vectorized(p_logits, gt_triplets, indices)
        
        # Subject/Object loss: calculate only for matched valid triplets
        losses['loss_s'] = self._compute_subject_loss_vectorized(s_logits, gt_triplets, indices)        
        losses['loss_o'] = self._compute_object_loss_vectorized(o_logits, gt_triplets, indices)
        
        return losses
    
    def _compute_predicate_loss_vectorized(self, p_logits, gt_triplets, indices):
        """Calculate vectorized Predicate loss"""
        B, N_pred, _ = p_logits.shape
        
        # 1. Initialize all prediction targets to 'no-relation' class
        target_p = torch.full((B, N_pred), p_logits.shape[-1] - 1, 
                             dtype=torch.int64, device=p_logits.device)
        
        # 2. Fill in actual GT predicate values only at matched positions
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:  # If there are matches
                # Extract predicate from GT triplet
                gt_predicates = torch.tensor([gt_triplets[b][idx][1] for idx in gt_idx], 
                                           device=p_logits.device)
                target_p[b, pred_idx] = gt_predicates
        
        # 3. Calculate CrossEntropy Loss for entire batch at once
        loss_p = F.cross_entropy(p_logits.transpose(1, 2), target_p, weight=self.p_weight)
        return loss_p
    
    def _compute_subject_loss_vectorized(self, s_logits, gt_triplets, indices):
        """Calculate vectorized Subject loss - only for matched valid triplets"""
        # Select only matched predictions and GT
        matched_s_logits, matched_gt_s = self._get_matched_predictions(
            s_logits, gt_triplets, indices, component='s'
        )
        
        if matched_s_logits.shape[0] == 0:
            # Return 0 loss if no matches
            return torch.tensor(0.0, device=s_logits.device, requires_grad=True)
        
        # Calculate CrossEntropy Loss only for matched items
        loss_s = F.cross_entropy(matched_s_logits, matched_gt_s)
        return loss_s
    
    def _compute_object_loss_vectorized(self, o_logits, gt_triplets, indices):
        """Calculate vectorized Object loss - only for matched valid triplets"""
        # Select only matched predictions and GT
        matched_o_logits, matched_gt_o = self._get_matched_predictions(
            o_logits, gt_triplets, indices, component='o'
        )
        
        if matched_o_logits.shape[0] == 0:
            # Return 0 loss if no matches
            return torch.tensor(0.0, device=o_logits.device, requires_grad=True)
        
        # Calculate CrossEntropy Loss only for matched items
        loss_o = F.cross_entropy(matched_o_logits, matched_gt_o)
        return loss_o
    
    def _get_matched_predictions(self, logits, gt_triplets, indices, component):
        """
        Helper function to extract matched predictions and GT
        
        Args:
            logits: (B, N_pred, num_classes)
            gt_triplets: List[List[Tuple[int, int, int]]]
            indices: List[Tuple[torch.Tensor, torch.Tensor]]
            component: 's' or 'o' (subject or object)
        
        Returns:
            matched_logits: (num_matched, num_classes)
            matched_gt: (num_matched,)
        """
        matched_logits_list = []
        matched_gt_list = []
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:  # If there are matches
                # Extract matched prediction logits
                matched_logits_list.append(logits[b, pred_idx])
                
                # Extract corresponding component from GT triplet
                if component == 's':
                    gt_values = torch.tensor([gt_triplets[b][idx][0] for idx in gt_idx], 
                                           device=logits.device)
                elif component == 'o':
                    gt_values = torch.tensor([gt_triplets[b][idx][2] for idx in gt_idx], 
                                           device=logits.device)
                
                matched_gt_list.append(gt_values)
        
        if len(matched_logits_list) == 0:
            # If no matches
            return torch.empty(0, logits.shape[-1], device=logits.device), \
                   torch.empty(0, dtype=torch.int64, device=logits.device)
        
        # Concatenate matched results from all batches
        matched_logits = torch.cat(matched_logits_list, dim=0)
        matched_gt = torch.cat(matched_gt_list, dim=0)
        
        return matched_logits, matched_gt