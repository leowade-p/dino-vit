# models.py

import torch
import torch.nn as nn

# =================================================================================
# Classification Model (Transformer-based)
#
# This file contains the definitions for the final classifier that takes a
# sequence of RoI features from a patient and predicts the diagnosis.
# The architecture uses a local positional embedding for patches within each RoI.
# =================================================================================

class PatchPositionEmbedding(nn.Module):
    """ 
    Positional embedding for patches within a single RoI feature map.
    It learns the spatial relationship of the 3x3 grid inside an RoI.
    """
    def __init__(self, grid_size=3, dim=512, learnable=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.dim = dim
        
        pos_embed = torch.randn(1, self.num_patches, dim) * 0.02
        if learnable:
            self.pos_embed = nn.Parameter(pos_embed)
        else:
            self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        # x shape: (B * MaxRoIs * num_patches, dim) -> (B * MaxRoIs, num_patches, dim)
        # pos_embed shape: (1, num_patches, dim)
        # Broadcasting adds the positional embedding to each RoI's patch sequence.
        return x 

class ROIBasedTransformerClassifier(nn.Module):
    """ 
    Downstream Transformer classifier that processes a sequence of RoI tokens.
    It uses self-attention to learn the relationships between different RoIs of a patient.
    """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            batch_first=True,
            dropout=0.1, # Added dropout for regularization
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # x shape: (B, L, C_hidden), where L is the total number of patches from all RoIs
        # mask shape: (B, L), indicating which patches are padding
        
        B, L, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            # Create a mask for the CLS token (always attend to it)
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            # TransformerEncoderLayer expects padding mask where True means ignore
            transformer_mask = ~full_mask
        else:
            transformer_mask = None
        
        encoded_x = self.encoder(x, src_key_padding_mask=transformer_mask)
        
        # We take the output of the CLS token for classification
        cls_output = self.norm(encoded_x[:, 0])
        return self.classifier(cls_output)

class FullTumorClassifier(nn.Module):
    """ 
    The complete classification model pipeline.
    It orchestrates the patch embedding, positional encoding, and Transformer classification.
    """
    def __init__(self, pos_embedder, transformer_classifier):
        super().__init__()
        self.pos_embedder = pos_embedder
        self.transformer_classifier = transformer_classifier
        self.tokens_per_roi = pos_embedder.num_patches

    def forward(self, roi_feat_maps, roi_mask=None):
        # roi_feat_maps: (B, MaxRoIs, C, H, W)
        # roi_mask: (B, MaxRoIs), True indicates a valid (non-padding) RoI
        
        B, MaxRoIs, C, H, W = roi_feat_maps.shape

        # 1. Reshape for processing: (B, MaxRoIs, C, H, W) -> (B * MaxRoIs, C, H, W)
        reshaped_maps = roi_feat_maps.reshape(-1, C, H, W)
        
        # 2. Flatten C,H,W to token sequence: (B*MaxRoIs, C, H, W) -> (B*MaxRoIs, H*W, C)
        tokens = reshaped_maps.flatten(2).permute(0, 2, 1)

        # 3. Apply positional embedding to each RoI's patch sequence
        tokens_with_pe = self.pos_embedder(tokens)
        
        # 4. Project to hidden_dim for Transformer
        projected_tokens = self.transformer_classifier.proj(tokens_with_pe)
        
        # 5. Reshape back to patient-level sequences: 
        # (B*MaxRoIs, H*W, C_hidden) -> (B, MaxRoIs*H*W, C_hidden)
        final_tokens = projected_tokens.reshape(B, -1, projected_tokens.shape[-1])

        # 6. Create a token-level mask from the RoI-level mask
        token_mask = None
        if roi_mask is not None:
            # If an RoI is padding (False), all its tokens should also be padding (False)
            token_mask = roi_mask.repeat_interleave(self.tokens_per_roi, dim=1)
        
        # 7. Pass the full sequence of tokens to the transformer for classification
        return self.transformer_classifier(final_tokens, token_mask)