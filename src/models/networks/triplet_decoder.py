import torch
import torch.nn as nn
from .basic_transformer import BasicTransformerBlock

class TripletDecoder(nn.Module):
    """
    DETR-style Triplet Decoder for predicting object relations
    """
    def __init__(self, 
                 dim: int, 
                 attn_dim: int, 
                 context_dim: int, 
                 max_num_rel: int = 4,
                 n_heads: int = 8, 
                 n_layers: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.max_num_rel = max_num_rel
        self.dim = dim
        
        # Triplet Queries: learnable parameters
        self.triplet_queries = nn.Parameter(torch.randn(max_num_rel, dim))
        nn.init.normal_(self.triplet_queries, mean=0.0, std=0.02)
        
        # Decoder layers: stack multiple BasicTransformerBlock layers
        self.decoder_layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=dim,
                attn_dim=attn_dim,
                context_dim=context_dim,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def forward(self, obj_feat: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        Args:
            obj_feat: Scene representation (B, N, D)
            pad_mask: Padding mask (B, N)
        
        Returns:
            triplet_features: (B, max_num_rel, D)
        """
        B = obj_feat.shape[0]
        
        # Replicate triplet queries for batch size
        queries = self.triplet_queries.unsqueeze(0).expand(B, -1, -1)  # (B, max_num_rel, D)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            queries = layer(
                x=queries,
                context=obj_feat,  # use scene representation as context
                mask=None,  # no masking for triplet queries
                context_mask=pad_mask
            )
        
        return queries  # (B, max_num_rel, D)
