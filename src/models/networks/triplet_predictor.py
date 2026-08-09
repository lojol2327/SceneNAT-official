import torch
import torch.nn as nn

class TripletPredictor(nn.Module):
    """
    Triplet Prediction Heads for Subject, Predicate, Object
    """
    def __init__(self, 
                 dim: int, 
                 n_class_types: int, 
                 n_predicate_types: int):
        super().__init__()
        
        # Subject prediction head (N objects + 1 no-object)
        self.s_head = nn.Linear(dim, n_class_types + 1)
        
        # Predicate prediction head (10 relations + 1 no-relation)
        self.p_head = nn.Linear(dim, n_predicate_types + 1)
        
        # Object prediction head (N objects + 1 no-object)
        self.o_head = nn.Linear(dim, n_class_types + 1)
        
    def forward(self, triplet_features: torch.Tensor):
        """
        Args:
            triplet_features: (B, max_num_rel, D)
        
        Returns:
            subj_logits: (B, max_num_rel, n_class_types + 1)
            pred_logits: (B, max_num_rel, n_predicate_types + 1)
            obj_logits: (B, max_num_rel, n_class_types + 1)
        """
        subj_logits = self.s_head(triplet_features)
        pred_logits = self.p_head(triplet_features)
        obj_logits = self.o_head(triplet_features)
        
        return subj_logits, pred_logits, obj_logits
