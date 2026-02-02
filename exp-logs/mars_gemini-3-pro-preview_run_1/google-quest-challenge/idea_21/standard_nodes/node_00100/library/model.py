import torch
import torch.nn as nn
import copy
from transformers import AutoModel
from library.config import Config


class DualRoBERTa(nn.Module):
    """
    Independent Dual-Encoder RoBERTa.
    - Two independent roberta-base backbones (one for Q, one for A).
    - Allows specialization for disjoint target labels.
    - Simple MLP head with backbone-aligned initialization.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Independent Backbones (Cite solution_lesson_node_00076)
        self.q_backbone = AutoModel.from_pretrained(config.BACKBONE)
        self.a_backbone = AutoModel.from_pretrained(config.BACKBONE)

        # Fusion Dimension
        # u_mean, v_mean, u_max, v_max, u_mean*v_mean, |u_mean-v_mean|
        # 6 * 768
        self.fusion_dim = config.HIDDEN_SIZE * 6

        # Minimalist Head (Cite solution_lesson_node_00054)
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_SIZE, config.NUM_TARGETS),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        # Initialize head to match backbone scale (Cite solution_lesson_node_00051)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

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
        out_q = self.q_backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)[
            0
        ]
        out_a = self.a_backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)[
            0
        ]

        # Pooling
        u_mean = self.masked_avg_pool(out_q, attention_mask_q)
        v_mean = self.masked_avg_pool(out_a, attention_mask_a)
        u_max = self.masked_max_pool(out_q, attention_mask_q)
        v_max = self.masked_max_pool(out_a, attention_mask_a)

        # Geometric Interactions (Cite solution_lesson_node_00085)
        # Computed directly on pooled outputs
        prod = u_mean * v_mean
        diff = torch.abs(u_mean - v_mean)

        # Concatenation
        features = torch.cat([u_mean, v_mean, u_max, v_max, prod, diff], dim=1)

        # Prediction
        logits = self.head(features)

        return logits
