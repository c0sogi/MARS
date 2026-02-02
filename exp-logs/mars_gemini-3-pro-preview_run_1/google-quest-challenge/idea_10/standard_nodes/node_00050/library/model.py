import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class DistilRobertaDualEncoder(nn.Module):
    """
    Text-Augmented DistilRoBERTa Dual-Encoder with Wide-Bottleneck Fusion.

    Architecture:
    1. Shared DistilRoBERTa backbone for Question and Answer.
    2. Masked Global Average and Max Pooling.
    3. Interaction-Aware Fusion: Concatenates avg/max vectors plus product and difference of averages.
    4. Wide-Bottleneck Projection Head: High-dim fusion -> 1024 -> 30 targets.
    """

    def __init__(self):
        super(DistilRobertaDualEncoder, self).__init__()

        # Load Configuration and Backbone
        # We use the config to get hidden sizes dynamically
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        self.hidden_size = config.hidden_size

        # ==========================================
        # Fusion Layer Definition
        # ==========================================
        # Components:
        # 1. u_avg (Question Average)
        # 2. u_max (Question Max)
        # 3. v_avg (Answer Average)
        # 4. v_max (Answer Max)
        # 5. u_avg * v_avg (Element-wise Product)
        # 6. |u_avg - v_avg| (Absolute Difference)
        # Total Dimension = hidden_size * 6
        self.fusion_dim = self.hidden_size * 6

        # Layer Normalization for the fused vector before the head
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # ==========================================
        # Wide-Bottleneck Projection Head
        # ==========================================
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, Config.BOTTLENECK_DIM),
            # Cite solution_lesson_node_00027: Use ReLU for the non-linear projection head.
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.BOTTLENECK_DIM, Config.NUM_TARGETS),
        )

        # Initialize weights for the new layers
        self._init_weights(self.head)
        self._init_weights(self.fusion_norm)

    def _init_weights(self, module):
        """
        Initialize weights for specific modules (Linear, LayerNorm).
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def global_average_pooling(self, hidden_states, attention_mask):
        """
        Computes Global Average Pooling masking out padding tokens.
        """
        # hidden_states: (Batch, Seq_Len, Hidden)
        # attention_mask: (Batch, Seq_Len)

        # Expand mask to match hidden dimensions: (Batch, Seq_Len, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )

        # Sum hidden states (masked)
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)

        # Sum mask (clamp to avoid division by zero)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def global_max_pooling(self, hidden_states, attention_mask):
        """
        Computes Global Max Pooling masking out padding tokens.
        """
        # hidden_states: (Batch, Seq_Len, Hidden)
        # attention_mask: (Batch, Seq_Len)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )

        # Clone hidden states to avoid modifying original tensor in place
        embeddings = hidden_states.clone()

        # Set padding tokens to a very small number so they aren't picked as max
        embeddings[input_mask_expanded == 0] = -1e9

        max_embeddings, _ = torch.max(embeddings, 1)

        return max_embeddings

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        """
        Forward pass of the dual encoder.
        """
        # ==========================================
        # Backbone Processing
        # ==========================================
        # Branch 1: Question
        outputs_q = self.backbone(
            input_ids=input_ids_q, attention_mask=attention_mask_q
        )
        last_hidden_state_q = outputs_q.last_hidden_state  # (B, L, H)

        # Branch 2: Answer
        outputs_a = self.backbone(
            input_ids=input_ids_a, attention_mask=attention_mask_a
        )
        last_hidden_state_a = outputs_a.last_hidden_state  # (B, L, H)

        # ==========================================
        # Pooling
        # ==========================================
        u_avg = self.global_average_pooling(last_hidden_state_q, attention_mask_q)
        u_max = self.global_max_pooling(last_hidden_state_q, attention_mask_q)

        v_avg = self.global_average_pooling(last_hidden_state_a, attention_mask_a)
        v_max = self.global_max_pooling(last_hidden_state_a, attention_mask_a)

        # ==========================================
        # Interaction-Aware Fusion
        # ==========================================
        # Compute explicit interactions on Average Pooled vectors
        interaction_prod = u_avg * v_avg
        interaction_diff = torch.abs(u_avg - v_avg)

        # Concatenate all components
        # Dimensions: H + H + H + H + H + H = 6H
        fused_vector = torch.cat(
            [u_avg, u_max, v_avg, v_max, interaction_prod, interaction_diff], dim=1
        )

        # Apply Layer Normalization
        fused_vector = self.fusion_norm(fused_vector)

        # ==========================================
        # Wide-Bottleneck Projection
        # ==========================================
        logits = self.head(fused_vector)

        return logits
