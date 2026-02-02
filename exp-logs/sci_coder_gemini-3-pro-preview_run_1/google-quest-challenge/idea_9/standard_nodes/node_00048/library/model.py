import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class BottleneckHead(nn.Module):
    """
    MLP Head with Dimensionality Reduction Bottleneck.
    Projects high-dim fused features to a lower dimension before classification.
    Cite Lesson 00046: Dimensionality Bottlenecks in Feature Fusion Heads.
    """

    def __init__(self, input_dim, hidden_dim=512, dropout_p=0.1, num_labels=30):
        super(BottleneckHead, self).__init__()

        # Bottleneck projection: 4608 -> 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        # Final classification: 512 -> 30
        self.fc2 = nn.Linear(hidden_dim, num_labels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DualDistilRoBERTa(nn.Module):
    """
    Text-Augmented DistilRoBERTa Dual-Encoder with Gated Feature Fusion.
    """

    def __init__(self):
        super(DualDistilRoBERTa, self).__init__()

        # Load Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Dimensions
        self.hidden_size = self.config.hidden_size

        # Fusion Dimension Calculation:
        # 1. u_avg (Question Avg)
        # 2. u_max (Question Max)
        # 3. v_avg (Answer Avg)
        # 4. v_max (Answer Max)
        # 5. u_avg * v_avg (Interaction Product)
        # 6. |u_avg - v_avg| (Interaction Diff)
        # Total = 6 * hidden_size
        self.fusion_dim = 6 * self.hidden_size

        # Normalization before the head
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # Bottleneck Head (Cite Lesson 00046)
        self.head = BottleneckHead(
            input_dim=self.fusion_dim,
            hidden_dim=512,
            dropout_p=Config.DROPOUT_PROB,
            num_labels=Config.NUM_LABELS,
        )

    def _masked_avg_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the average of hidden states, ignoring padding tokens.
        """
        # Expand mask: [Batch, SeqLen] -> [Batch, SeqLen, Hidden]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum hidden states masked
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum valid tokens
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

        return sum_embeddings / sum_mask

    def _masked_max_pooling(self, last_hidden_state, attention_mask):
        """
        Computes the max of hidden states, ignoring padding tokens.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Replace padding tokens with a very small number so they aren't picked as max
        last_hidden_state = (
            last_hidden_state.clone()
        )  # Clone to avoid modifying in-place if needed
        last_hidden_state[input_mask_expanded == 0] = -1e9

        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # --- Question Branch ---
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state  # [Batch, SeqLen, Hidden]

        u_avg = self._masked_avg_pooling(q_hidden, q_attention_mask)
        u_max = self._masked_max_pooling(q_hidden, q_attention_mask)

        # --- Answer Branch ---
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state  # [Batch, SeqLen, Hidden]

        v_avg = self._masked_avg_pooling(a_hidden, a_attention_mask)
        v_max = self._masked_max_pooling(a_hidden, a_attention_mask)

        # --- Interaction-Aware Fusion ---
        # Interactions computed ONLY on Average Pooled vectors
        interaction_prod = u_avg * v_avg
        interaction_diff = torch.abs(u_avg - v_avg)

        # Concatenate: [u_avg, u_max, v_avg, v_max, prod, diff]
        fused_features = torch.cat(
            [u_avg, u_max, v_avg, v_max, interaction_prod, interaction_diff], dim=1
        )

        # Normalize
        fused_features = self.fusion_norm(fused_features)

        # --- Bottleneck Head ---
        logits = self.head(fused_features)

        return logits
