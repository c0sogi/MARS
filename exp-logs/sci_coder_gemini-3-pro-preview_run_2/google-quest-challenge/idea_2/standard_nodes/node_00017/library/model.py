import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CausalDeberta(nn.Module):
    """
    Causal-Aware Siamese DeBERTa network.

    Implements decoupled prediction heads (Cite solution_lesson_node_00014):
    1. Question Head: Predicts 21 question-related targets using only Question embeddings.
    2. Answer/Interaction Head: Predicts 9 answer-related targets using [u, v, |u-v|, u*v].

    Uses Full Fine-Tuning (Cite solution_lesson_node_00008).
    """

    def __init__(self):
        super(CausalDeberta, self).__init__()

        # Load configuration and pre-trained backbone
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        # Identify target indices
        # Config.TARGET_COLS is ordered: 21 question cols then 9 answer cols
        self.q_indices = [
            i for i, c in enumerate(Config.TARGET_COLS) if c.startswith("question_")
        ]
        self.a_indices = [
            i for i, c in enumerate(Config.TARGET_COLS) if not c.startswith("question_")
        ]

        # Head 1: Question Only (Input: u)
        self.q_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(config.hidden_size, len(self.q_indices)),
            nn.Sigmoid(),
        )

        # Head 2: Interaction (Input: [u, v, |u-v|, u*v])
        self.fusion_dim = 4 * config.hidden_size
        self.a_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(config.hidden_size, len(self.a_indices)),
            nn.Sigmoid(),
        )

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        # --- Question Stream ---
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        u = self._mean_pooling(q_outputs.last_hidden_state, q_attention_mask)

        # --- Answer Stream ---
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        v = self._mean_pooling(a_outputs.last_hidden_state, a_attention_mask)

        # --- Head 1: Question Targets ---
        q_probs = self.q_classifier(u)

        # --- Head 2: Answer Targets ---
        diff = torch.abs(u - v)
        prod = u * v
        fused_features = torch.cat([u, v, diff, prod], dim=1)
        a_probs = self.a_classifier(fused_features)

        # --- Reassemble ---
        # Concatenate in order (Question cols first, then Answer cols)
        return torch.cat([q_probs, a_probs], dim=1)

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Applies mean pooling to the token embeddings, ignoring padded tokens.

        Args:
            last_hidden_state (Tensor): Sequence of hidden states (Batch, SeqLen, Hidden).
            attention_mask (Tensor): Mask indicating valid tokens (Batch, SeqLen).

        Returns:
            Tensor: Pooled representation (Batch, Hidden).
        """
        # Expand mask to match hidden state dimensions: (Batch, SeqLen, Hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings of valid tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Count valid tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Compute mean
        return sum_embeddings / sum_mask
