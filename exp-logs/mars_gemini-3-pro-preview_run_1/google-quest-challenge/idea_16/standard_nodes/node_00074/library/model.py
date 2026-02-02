import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DualDistilRoBERTa(nn.Module):
    """
    DistilRoBERTa Dual-Encoder with Consistency-Regularized Optimization (R-Drop).

    Architecture:
    - Backbone: Shared distilroberta-base.
    - Streams: Independent Question and Answer processing.
    - Pooling: Masked Global Average and Max Pooling.
    - Fusion: Concatenation of [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|].
    - Head: Linear -> ReLU -> Dropout -> Linear.
    """

    def __init__(self, num_labels=30, model_name="distilroberta-base"):
        super(DualDistilRoBERTa, self).__init__()

        # Load Backbone Configuration and Model
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Dimensions
        self.hidden_size = config.hidden_size  # 768 for distilroberta-base

        # Fusion Dimension:
        # 4 direct vectors (u_avg, u_max, v_avg, v_max) + 2 interaction vectors (prod, diff)
        # Total = 6 * hidden_size
        self.fusion_dim = 6 * self.hidden_size

        # Layer Normalization after fusion
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # Monolithic Minimalist Head
        # Structure: Linear -> ReLU -> Dropout(0.1) -> Linear
        # We project to hidden_size first to allow feature mixing before final projection
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels),
        )

        # Initialize Head Weights
        self._init_head_weights()

    def _init_head_weights(self):
        """
        Initialize the head's weights using a Normal distribution (mu=0, sigma=0.02).
        """
        for module in self.head:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _masked_avg_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the average of hidden states, ignoring padded tokens.
        """
        # Expand mask: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum mask (count valid tokens)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

        return sum_embeddings / sum_mask

    def _masked_max_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the max of hidden states, ignoring padded tokens.
        """
        # Expand mask
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Clone to avoid in-place modification issues if used elsewhere
        embeddings = last_hidden_state.clone()

        # Set padding tokens to large negative value so they are not selected by max
        embeddings[input_mask_expanded == 0] = -1e9

        # Max over sequence dimension
        return torch.max(embeddings, 1)[0]

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass for the Dual-Encoder.
        """
        # --- Stream A: Question ---
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        q_hidden = q_outputs.last_hidden_state

        # Pooling
        u_avg = self._masked_avg_pooling(q_hidden, q_attention_mask)
        u_max = self._masked_max_pooling(q_hidden, q_attention_mask)

        # --- Stream B: Answer ---
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        a_hidden = a_outputs.last_hidden_state

        # Pooling
        v_avg = self._masked_avg_pooling(a_hidden, a_attention_mask)
        v_max = self._masked_max_pooling(a_hidden, a_attention_mask)

        # --- Interaction-Aware Fusion ---
        # Interactions computed ONLY on Average Pooled vectors
        uv_prod = u_avg * v_avg
        uv_diff = torch.abs(u_avg - v_avg)

        # Concatenate: [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
        fused = torch.cat([u_avg, u_max, v_avg, v_max, uv_prod, uv_diff], dim=1)

        # Normalize
        fused = self.fusion_norm(fused)

        # --- Prediction Head ---
        logits = self.head(fused)

        return logits
