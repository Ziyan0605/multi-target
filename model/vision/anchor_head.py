import torch
import torch.nn as nn

from pipeline.registry import registry


@registry.register_other_model("anchor_head_v1")
class AnchorHeadV1(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, obj_embeds, anchor_ids, anchor_masks):
        """Aggregate anchor-conditioned features.

        Args:
            obj_embeds (torch.Tensor): object embeddings of shape (B, O, C).
            anchor_ids (torch.Tensor): anchor indices of shape (B, A) padded with -1.
            anchor_masks (torch.Tensor): boolean mask of shape (B, A) indicating valid anchors.
        Returns:
            torch.Tensor: aggregated anchor context of shape (B, C).
        """
        if obj_embeds.dim() != 3:
            raise ValueError("obj_embeds should have shape (B, O, C)")

        batch_size, _, hidden_size = obj_embeds.shape

        if anchor_ids.dim() == 1:
            anchor_ids = anchor_ids.unsqueeze(0)
        if anchor_masks.dim() == 1:
            anchor_masks = anchor_masks.unsqueeze(0)

        if anchor_ids.shape[0] != batch_size:
            raise ValueError("anchor_ids batch dimension mismatch with obj_embeds")
        if anchor_masks.shape != anchor_ids.shape:
            raise ValueError("anchor_masks must match anchor_ids shape")

        if anchor_ids.shape[1] == 0:
            return obj_embeds.new_zeros(batch_size, hidden_size)

        valid_mask = anchor_masks.float()
        if valid_mask.sum() == 0:
            return obj_embeds.new_zeros(batch_size, hidden_size)

        max_index = obj_embeds.shape[1] - 1
        gather_index = anchor_ids.clamp(min=0, max=max_index).unsqueeze(-1).expand(-1, -1, hidden_size)
        anchor_feats = torch.gather(obj_embeds, 1, gather_index)
        anchor_feats = anchor_feats * valid_mask.unsqueeze(-1)
        anchor_sum = anchor_feats.sum(dim=1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        anchor_mean = anchor_sum / denom
        anchor_mean = self.dropout(anchor_mean)
        return self.proj(anchor_mean)
