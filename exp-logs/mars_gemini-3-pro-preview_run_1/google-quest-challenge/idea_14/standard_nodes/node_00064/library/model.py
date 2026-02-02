import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class ContextualizedDualEncoder(nn.Module):
    """
    DistilRoBERTa Dual-Encoder.

    Architecture:
    - Shared Backbone: distilroberta-base
    - Branch 1 Input: [CLS] Title [SEP] Body [SEP]
    - Branch 2 Input: [CLS] Answer [SEP] (Cite solution_lesson_node_00063)
    - Pooling:
        - Branch 1: Global Average + Max Pooling
        - Branch 2: Target-Masked Pooling (Average + Max) restricted to Answer tokens
    - Fusion: Concatenation of [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
    - Head: LayerNorm -> Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self):
        super(ContextualizedDualEncoder, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)

        self.hidden_size = self.config.hidden_size

        # Fusion Dimension Calculation
        # Vectors: u_avg, u_max, v_avg, v_max, prod, diff
        # Count: 6 vectors of size hidden_size
        self.fusion_dim = self.hidden_size * 6

        # Monolithic Simple Head
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_TARGETS),
        )

        # Initialize weights for the head (optional, but good practice)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights of the head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids_q,
        attention_mask_q,
        input_ids_a,
        attention_mask_a,
        pooling_mask_a,
    ):
        """
        Args:
            input_ids_q: Input IDs for Question stream (Title + Body)
            attention_mask_q: Attention mask for Question stream
            input_ids_a: Input IDs for Answer stream (Title + Answer)
            attention_mask_a: Attention mask for Answer stream (used for backbone)
            pooling_mask_a: Specific mask for pooling Answer tokens from Branch 2

        Returns:
            logits: Unnormalized output scores (B, 30)
        """
        # --- Branch 1: Question (Title + Body) ---
        out_q = self.backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)
        last_hidden_q = out_q.last_hidden_state  # (B, L, H)

        # --- Branch 2: Contextualized Answer (Title + Answer) ---
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        last_hidden_a = out_a.last_hidden_state  # (B, L, H)

        # --- Pooling Branch 1 (Standard Global) ---
        # Expand mask for broadcasting: (B, L) -> (B, L, 1)
        mask_q = attention_mask_q.unsqueeze(-1).float()

        # Average Pooling
        sum_embeddings_q = torch.sum(last_hidden_q * mask_q, dim=1)
        sum_mask_q = torch.clamp(mask_q.sum(dim=1), min=1e-9)
        u_avg = sum_embeddings_q / sum_mask_q

        # Max Pooling
        # Set padded tokens to a very small number so they aren't selected by max
        last_hidden_q_masked = last_hidden_q.clone()
        last_hidden_q_masked[mask_q.expand_as(last_hidden_q) == 0] = -1e9
        u_max = torch.max(last_hidden_q_masked, dim=1)[0]

        # --- Pooling Branch 2 (Target-Masked: Answer tokens only) ---
        # pooling_mask_a is 1.0 for Answer tokens, 0.0 for Title/Special tokens
        mask_a = pooling_mask_a.unsqueeze(-1)  # (B, L, 1)

        # Average Pooling
        sum_embeddings_a = torch.sum(last_hidden_a * mask_a, dim=1)
        sum_mask_a = torch.clamp(mask_a.sum(dim=1), min=1e-9)
        v_avg = sum_embeddings_a / sum_mask_a

        # Max Pooling
        last_hidden_a_masked = last_hidden_a.clone()
        last_hidden_a_masked[mask_a.expand_as(last_hidden_a) == 0] = -1e9
        v_max = torch.max(last_hidden_a_masked, dim=1)[0]

        # --- Interaction-Aware Fusion ---
        # Compute interactions on the average vectors (representing the core semantic content)
        diff = torch.abs(u_avg - v_avg)
        prod = u_avg * v_avg

        # Concatenate all features
        # [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
        concat_features = torch.cat([u_avg, u_max, v_avg, v_max, prod, diff], dim=1)

        # --- Head ---
        x = self.layer_norm(concat_features)
        logits = self.classifier(x)

        return logits
