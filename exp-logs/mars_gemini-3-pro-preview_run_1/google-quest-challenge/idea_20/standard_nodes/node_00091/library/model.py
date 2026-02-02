import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class SiameseDualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Configuration and Base Model
        # Cite Lesson 00077: Capacity Scaling in Dual-Encoders (RoBERTa-base > DistilRoBERTa)
        # Cite Lesson 00085: Avoid Uninitialized Projections Before Geometric Interactions
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.hidden_size = Config.HIDDEN_SIZE

        # Fused Vector F Components:
        # u_avg, u_max, v_avg, v_max (4 vectors)
        # u_avg * v_avg (1 vector)
        # |u_avg - v_avg| (1 vector)
        # Total = 6 vectors
        self.fusion_dim = self.hidden_size * 6

        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # Residual Interaction Head
        # Structure: Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
        self.head_proj = nn.Linear(self.fusion_dim, self.hidden_size)
        self.head_dropout = nn.Dropout(Config.DROPOUT)
        self.head_final = nn.Linear(
            self.fusion_dim + self.hidden_size, Config.NUM_TARGETS
        )

        # Initialize Head Weights
        self._init_weights(self.head_proj)
        self._init_weights(self.head_final)

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
        # 1. Siamese Encoding
        q_out = self.backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)[0]
        a_out = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)[0]

        # 2. Pooling
        q_avg = self.masked_mean_pooling(q_out, attention_mask_q)
        q_max = self.masked_max_pooling(q_out, attention_mask_q)

        a_avg = self.masked_mean_pooling(a_out, attention_mask_a)
        a_max = self.masked_max_pooling(a_out, attention_mask_a)

        # 3. Interaction (Only on Average Pooled vectors)
        # Cite Lesson 00057: Simplicity in Feature Fusion
        prod = q_avg * a_avg
        diff = torch.abs(q_avg - a_avg)

        # 4. Fusion
        F_vec = torch.cat([q_avg, q_max, a_avg, a_max, prod, diff], dim=1)
        F_vec = self.layer_norm(F_vec)

        # 5. Residual Interaction Head
        # Path A: Non-linear projection
        proj = self.head_proj(F_vec)
        proj = F.relu(proj)
        proj = self.head_dropout(proj)

        # Path B: Skip connection (Concatenation)
        concat = torch.cat([F_vec, proj], dim=1)

        # Final Logits
        logits = self.head_final(concat)

        return logits
