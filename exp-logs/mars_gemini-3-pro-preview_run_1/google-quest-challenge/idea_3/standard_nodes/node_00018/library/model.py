import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MaskedHybridPooling(nn.Module):
    """
    Performs hybrid pooling (Mean + Max) on transformer outputs,
    correctly handling attention masks.
    """

    def __init__(self):
        super(MaskedHybridPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)

        # Expand mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Max Pooling
        # Use masked_fill to set padding tokens to a large negative value
        # so they are not selected by max pooling.
        bool_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()) == 0
        masked_hidden = last_hidden_state.masked_fill(bool_mask, -1e9)
        max_embeddings = torch.max(masked_hidden, 1)[0]

        # Concatenate [Mean, Max] -> Size: (batch, 2 * hidden_size)
        return torch.cat([mean_embeddings, max_embeddings], 1)


class ContextualDualEncoder(nn.Module):
    """
    Monolithic Dual-Encoder architecture.
    Uses a shared backbone (e.g., DistilRoBERTa) to encode Question and Answer independently.
    Predicts all 30 targets from a fused representation.
    """

    def __init__(self, model_name="distilroberta-base"):
        super(ContextualDualEncoder, self).__init__()

        # Load Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.hidden_size = self.config.hidden_size

        # Pooling Layer
        self.pooler = MaskedHybridPooling()

        # Dimensions
        # Hybrid pooling outputs 2 * hidden_size (Mean + Max)
        self.pooled_dim = self.hidden_size * 2

        # Fusion Dimension calculation:
        # u (Question Pool, 2H) + v (Answer Pool, 2H) +
        # (u_mean * v_mean) (H) + |u_mean - v_mean| (H)
        # Total = 2H + 2H + H + H = 6H
        self.fusion_dim = 6 * self.hidden_size

        # Normalization for the heterogeneous fused vector (Cite solution_lesson_node_00009)
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # Monolithic Head: Predicts all 30 targets (Cite solution_lesson_node_00016)
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 30),
        )

        # Initialize custom head
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Initialize weights for the custom heads."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass for the dual-encoder model.
        """
        # 1. Encode Question
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        # u shape: (Batch, 2 * Hidden) -> [Mean, Max]
        u = self.pooler(q_out.last_hidden_state, q_attention_mask)

        # 2. Encode Answer
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        # v shape: (Batch, 2 * Hidden) -> [Mean, Max]
        v = self.pooler(a_out.last_hidden_state, a_attention_mask)

        # 3. Feature Fusion
        # Extract Mean components (first half) for interaction (Cite solution_lesson_node_00007)
        u_mean = u[:, : self.hidden_size]
        v_mean = v[:, : self.hidden_size]

        # Compute Interaction Terms
        prod = u_mean * v_mean
        diff = torch.abs(u_mean - v_mean)

        # Concatenate: u (2H), v (2H), prod (H), diff (H) -> 6H
        fused = torch.cat([u, v, prod, diff], dim=1)

        # Normalize
        fused = self.fusion_norm(fused)

        # 4. Prediction
        logits = self.head(fused)

        return logits
