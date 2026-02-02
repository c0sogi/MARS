import torch
import torch.nn as nn
import copy
from transformers import AutoModel
from library.config import Config


class ResidualProjectionHead(nn.Module):
    """
    Residual Interaction Head:
    Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
    Preserves raw multi-level signals via skip connection while enabling non-linear mixing.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.final = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x):
        # Path 1: Non-linear transformation
        h = self.project(x)
        h = self.relu(h)
        h = self.dropout(h)

        # Path 2: Skip connection (Concat)
        combined = torch.cat([x, h], dim=1)

        # Final projection
        logits = self.final(combined)
        return logits


class DualRoBERTa(nn.Module):
    """
    Independent Dual-Encoder RoBERTa.
    Uses two separate RoBERTa backbones for Question and Answer streams.
    Implements Lesson 00076 (Decouple Encoders) and Lesson 00077 (Capacity Scaling).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Independent Backbones (Cite 00076)
        self.q_backbone = AutoModel.from_pretrained(config.BACKBONE)
        self.a_backbone = AutoModel.from_pretrained(config.BACKBONE)

        # Fusion Dimension:
        # u_avg, v_avg, u_max, v_max, u*v, |u-v|
        # 6 * 768 = 4608
        self.fusion_dim = config.HIDDEN_SIZE * 6

        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        self.head = ResidualProjectionHead(
            input_dim=self.fusion_dim,
            hidden_dim=config.HIDDEN_SIZE,
            output_dim=config.NUM_TARGETS,
        )

        self._init_head_weights()

    def _init_head_weights(self):
        # Initialize head specific layers
        # Use Normal initialization to match backbone (Cite 00051)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def masked_avg_pool(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def masked_max_pool(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).bool()
        out = hidden_states.clone()
        out[~mask_expanded] = -1e9
        return torch.max(out, 1)[0]

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        # Independent Forward Passes
        out_q = self.q_backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)
        out_a = self.a_backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)

        # Last Hidden States
        last_hidden_q = out_q.last_hidden_state
        last_hidden_a = out_a.last_hidden_state

        # Pooling
        u_avg = self.masked_avg_pool(last_hidden_q, attention_mask_q)
        v_avg = self.masked_avg_pool(last_hidden_a, attention_mask_a)
        u_max = self.masked_max_pool(last_hidden_q, attention_mask_q)
        v_max = self.masked_max_pool(last_hidden_a, attention_mask_a)

        # Interactions (Cite 00057: Simple interactions preferred)
        i_prod = u_avg * v_avg
        i_diff = torch.abs(u_avg - v_avg)

        # Fusion
        features = torch.cat(
            [u_avg, v_avg, u_max, v_max, i_prod, i_diff],
            dim=1,
        )

        features = self.fusion_norm(features)
        logits = self.head(features)

        return logits
