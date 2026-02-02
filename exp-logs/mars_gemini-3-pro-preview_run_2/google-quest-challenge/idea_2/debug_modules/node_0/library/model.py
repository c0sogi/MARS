import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SiameseRoBERTa(nn.Module):
    """
    Siamese DistilRoBERTa network with an explicit interaction head.

    Architecture:
    1. Shared DistilRoBERTa backbone for Question and Answer streams.
    2. Mean Pooling to generate fixed-size sentence embeddings (u, v).
    3. Interaction Fusion: [u, v, |u-v|, u*v].
    4. MLP Classification Head mapping to 30 target probabilities.
    """

    def __init__(self):
        super(SiameseRoBERTa, self).__init__()

        # Load configuration and pre-trained backbone
        config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=config)

        # Dimension of the fused interaction vector
        # Concatenation of u, v, |u-v|, u*v results in 4x the hidden size
        self.fusion_dim = 4 * config.hidden_size

        # Classification Head (MLP)
        # Maps the fused vector to the 30 target labels
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(config.hidden_size, Config.NUM_LABELS),
            nn.Sigmoid(),  # Output probabilities in range [0, 1]
        )

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass of the Siamese network.

        Args:
            q_input_ids (Tensor): Input IDs for the question stream (Batch, SeqLen).
            q_attention_mask (Tensor): Attention mask for the question stream (Batch, SeqLen).
            a_input_ids (Tensor): Input IDs for the answer stream (Batch, SeqLen).
            a_attention_mask (Tensor): Attention mask for the answer stream (Batch, SeqLen).

        Returns:
            Tensor: Probabilities for the 30 target labels (Batch, 30).
        """
        # --- Question Stream ---
        # Pass through backbone
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        # Apply Mean Pooling to get embedding 'u'
        u = self._mean_pooling(q_outputs.last_hidden_state, q_attention_mask)

        # --- Answer Stream ---
        # Pass through backbone (shared weights)
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        # Apply Mean Pooling to get embedding 'v'
        v = self._mean_pooling(a_outputs.last_hidden_state, a_attention_mask)

        # --- Interaction Fusion ---
        # Construct features: [u, v, |u-v|, u*v]
        diff = torch.abs(u - v)
        prod = u * v
        fused_features = torch.cat([u, v, diff, prod], dim=1)

        # --- Classification ---
        probs = self.classifier(fused_features)

        return probs

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
