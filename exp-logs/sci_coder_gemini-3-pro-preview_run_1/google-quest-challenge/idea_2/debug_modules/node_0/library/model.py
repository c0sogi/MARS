import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class ContextualDualEncoder(nn.Module):
    """
    Contextual Dual-Encoder with Hybrid Pooling and Explicit Interactions.

    Architecture:
    1. Shared Transformer Backbone (distilroberta-base).
    2. Dual-Branch processing for Question and Answer.
    3. Hybrid Pooling (Masked Average + Max) for both branches.
    4. Explicit Interactions (Product, Abs Diff) on Average Pooled vectors only.
    5. Concatenation of [Q_avg, Q_max, A_avg, A_max, Interaction_Prod, Interaction_Diff].
    6. MLP Head for 30-label regression.
    """

    def __init__(self):
        super(ContextualDualEncoder, self).__init__()

        # 1. Backbone Architecture
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)

        # Dimensions
        self.hidden_size = Config.HIDDEN_SIZE

        # Fusion Dimension Calculation
        # Components: Q_avg (H), Q_max (H), A_avg (H), A_max (H), Prod (H), Diff (H)
        # Total = 6 * H
        self.fusion_size = self.hidden_size * 6

        # 2. MLP Head
        self.head = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.fusion_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, Config.NUM_LABELS),
        )

        # Initialize weights for the head (optional, but good practice)
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Initialize weights for the MLP head."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def freeze_backbone(self):
        """Freezes the parameters of the transformer backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the parameters of the transformer backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _avg_pooling(self, last_hidden_state, attention_mask):
        """
        Performs Masked Global Average Pooling.

        Args:
            last_hidden_state: (Batch, Seq_Len, Hidden_Dim)
            attention_mask: (Batch, Seq_Len)

        Returns:
            (Batch, Hidden_Dim)
        """
        # Expand mask to match hidden state dimensions: (B, L) -> (B, L, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings, ignoring padded tokens
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)

        # Sum mask to get count of valid tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def _max_pooling(self, last_hidden_state, attention_mask):
        """
        Performs Global Max Pooling with Masking.

        Args:
            last_hidden_state: (Batch, Seq_Len, Hidden_Dim)
            attention_mask: (Batch, Seq_Len)

        Returns:
            (Batch, Hidden_Dim)
        """
        # Expand mask: (B, L) -> (B, L, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bool()
        )

        # Clone to avoid in-place modification issues
        embeddings = last_hidden_state.clone()

        # Set padded tokens to a very small number so they aren't selected by max
        embeddings[~mask_expanded] = -1e9

        # Take max over sequence dimension
        return torch.max(embeddings, 1)[0]

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass of the model.

        Args:
            q_input_ids, q_attention_mask: Inputs for Question branch
            a_input_ids, a_attention_mask: Inputs for Answer branch

        Returns:
            logits: (Batch, 30)
        """
        # ---------------------------------------------------------
        # 1. Question Branch
        # ---------------------------------------------------------
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state  # (B, L, H)

        q_avg = self._avg_pooling(q_hidden, q_attention_mask)
        q_max = self._max_pooling(q_hidden, q_attention_mask)

        # ---------------------------------------------------------
        # 2. Answer Branch
        # ---------------------------------------------------------
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state  # (B, L, H)

        a_avg = self._avg_pooling(a_hidden, a_attention_mask)
        a_max = self._max_pooling(a_hidden, a_attention_mask)

        # ---------------------------------------------------------
        # 3. Explicit Interactions (Only on Average Pooled Vectors)
        # ---------------------------------------------------------
        # Element-wise Product
        inter_prod = q_avg * a_avg

        # Absolute Difference
        inter_diff = torch.abs(q_avg - a_avg)

        # ---------------------------------------------------------
        # 4. Fusion
        # ---------------------------------------------------------
        # Concatenate: [Q_avg, Q_max, A_avg, A_max, Prod, Diff]
        fusion_vector = torch.cat(
            [q_avg, q_max, a_avg, a_max, inter_prod, inter_diff], dim=1
        )

        # ---------------------------------------------------------
        # 5. Prediction Head
        # ---------------------------------------------------------
        logits = self.head(fusion_vector)

        return logits
