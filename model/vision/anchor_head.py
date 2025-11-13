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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, obj_embeds, obj_masks, anchor_labels=None):
        """Predict anchor logits and aggregate anchor-aware context.

        Args:
            obj_embeds (torch.Tensor): object embeddings of shape (B, O, C).
            obj_masks (torch.Tensor): boolean mask of shape (B, O) for valid objects.
            anchor_labels (torch.Tensor, optional): binary labels of shape (B, O).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - anchor logits of shape (B, O)
                - anchor context of shape (B, C)
        """
        if obj_embeds.dim() != 3:
            raise ValueError("obj_embeds should have shape (B, O, C)")

        if obj_masks.dim() != 2:
            raise ValueError("obj_masks should have shape (B, O)")

        if obj_embeds.shape[:2] != obj_masks.shape:
            raise ValueError("obj_masks must match the first two dimensions of obj_embeds")

        batch_size, num_obj, hidden_size = obj_embeds.shape
        obj_masks_bool = obj_masks.bool()

        anchor_logits = self.classifier(obj_embeds).squeeze(-1)
        anchor_logits = anchor_logits.masked_fill(~obj_masks_bool, -1e4)

        mask_float = obj_masks_bool.float()
        if anchor_labels is not None:
            if anchor_labels.shape != obj_masks.shape:
                raise ValueError("anchor_labels must match obj_masks shape")
            weights = anchor_labels.float() * mask_float
        else:
            weights = torch.sigmoid(anchor_logits) * mask_float

        weights_sum = weights.sum(dim=1, keepdim=True)
        denom = weights_sum.clamp(min=1e-6)
        pooled = torch.bmm(weights.unsqueeze(1), obj_embeds).squeeze(1)
        context = pooled / denom
        context = self.dropout(context)
        context = self.proj(context)
        return anchor_logits, context
