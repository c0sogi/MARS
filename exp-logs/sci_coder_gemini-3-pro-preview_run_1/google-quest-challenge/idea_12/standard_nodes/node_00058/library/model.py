import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class DistilRobertaDualEncoder(nn.Module):
    """
    Dual-Encoder architecture using DistilRoBERTa.
    Features:
    - Independent processing of Question and Answer.
    - Hybrid Pooling (Mean + Max).
    - Interaction features (Product, Difference).
    - Monolithic output head for 30 targets.
    """

    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        # Freeze embeddings if needed, but usually fine-tuning updates them.
        # We rely on differential learning rates in the training loop instead of freezing here.

        self.hidden_size = Config.HIDDEN_SIZE

        # Feature Dimension Calculation
        # 1. q_avg
        # 2. q_max
        # 3. a_avg
        # 4. a_max
        # 5. q_avg * a_avg
        # 6. |q_avg - a_avg|
        # Total = 6 * hidden_size
        self.fusion_dim = 6 * self.hidden_size

        # Head Architecture
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_size, Config.NUM_TARGETS),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize custom head weights.
        """
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

        # LayerNorm usually initializes weight to 1 and bias to 0 by default, which is fine.

    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        )
        # Set padding tokens to large negative value
        hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(hidden_state, 1)[0]
        return max_embeddings

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # --- Question Branch ---
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state  # (B, L_q, H)

        q_avg = self.mean_pooling(q_hidden, q_attention_mask)  # (B, H)
        q_max = self.max_pooling(q_hidden, q_attention_mask)  # (B, H)

        # --- Answer Branch ---
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state  # (B, L_a, H)

        a_avg = self.mean_pooling(a_hidden, a_attention_mask)  # (B, H)
        a_max = self.max_pooling(a_hidden, a_attention_mask)  # (B, H)

        # --- Feature Fusion ---
        # Explicit Interactions (Global)
        inter_prod_global = q_avg * a_avg
        inter_diff_global = torch.abs(q_avg - a_avg)

        # Concatenate all features
        features = torch.cat(
            [
                q_avg,  # 1
                q_max,  # 2
                a_avg,  # 3
                a_max,  # 4
                inter_prod_global,  # 5
                inter_diff_global,  # 6
            ],
            dim=1,
        )

        # --- Classification Head ---
        norm_features = self.layer_norm(features)
        logits = self.classifier(norm_features)

        return logits
