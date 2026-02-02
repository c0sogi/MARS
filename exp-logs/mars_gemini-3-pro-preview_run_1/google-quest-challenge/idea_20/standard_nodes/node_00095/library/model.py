import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class SiameseDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Configuration and Base Model
        # Cite Lesson 77: Scaling to roberta-base improves robustness and performance
        self.base_model = AutoModel.from_pretrained(Config.MODEL_NAME)

        self.hidden_size = Config.HIDDEN_SIZE

        # Fused Vector F Components:
        # u_avg, u_max, v_avg, v_max (4 vectors)
        # u_avg * v_avg (1 vector)
        # |u_avg - v_avg| (1 vector)
        # Total = 6 vectors
        self.fusion_dim = self.hidden_size * 6

        # Simple MLP Head (Cite Lesson 54: Minimalist Projection Heads Preferable)
        # Linear -> ReLU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, Config.NUM_TARGETS),
        )

        # Initialize Head Weights
        self._init_weights(self.head[0])
        self._init_weights(self.head[3])

    def _init_weights(self, module):
        """Initialize weights with Normal distribution as per strategy."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=Config.INIT_MEAN, std=Config.INIT_STD)
            if module.bias is not None:
                module.bias.data.zero_()

    def masked_mean_pooling(self, hidden_states, attention_mask):
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def masked_max_pooling(self, hidden_states, attention_mask):
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        # Set padded tokens to large negative value so they aren't picked as max
        hidden_states = hidden_states.clone()
        hidden_states[mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(hidden_states, 1)
        return max_embeddings

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        # 1. Encoding (Siamese - Shared Weights)
        # Cite Lesson 80: Symmetric backbones required for geometric interactions
        q_out = self.base_model(input_ids=input_ids_q, attention_mask=attention_mask_q)[
            0
        ]
        a_out = self.base_model(input_ids=input_ids_a, attention_mask=attention_mask_a)[
            0
        ]

        # 2. Pooling
        q_avg = self.masked_mean_pooling(q_out, attention_mask_q)
        q_max = self.masked_max_pooling(q_out, attention_mask_q)

        a_avg = self.masked_mean_pooling(a_out, attention_mask_a)
        a_max = self.masked_max_pooling(a_out, attention_mask_a)

        # 3. Interaction (Cite Lesson 57: Simple interaction features)
        prod = q_avg * a_avg
        diff = torch.abs(q_avg - a_avg)

        # 4. Fusion
        F_vec = torch.cat([q_avg, q_max, a_avg, a_max, prod, diff], dim=1)

        # 5. Head
        logits = self.head(F_vec)

        return logits
