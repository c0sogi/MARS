import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import MODEL_NAME, IDENTITY_COLUMNS, DROPOUT


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels (features) across the temporal dimension.
    This promotes independence between feature maps.
    """

    def __init__(self, p=0.2):
        super().__init__()
        # Dropout1d expects (batch, channels, length)
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # Permute to (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        # Permute back to (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)
        return x


class AttentionPooling(nn.Module):
    """
    Attention Pooling mechanism that computes a weighted sum of hidden states.
    It learns which tokens are important for the classification task.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - 1 for token, 0 for pad
        """
        # Calculate raw attention scores
        # w shape: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the average
        # Expand mask to match w dimensions: (batch_size, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1)

        # Set scores of padding tokens to a very large negative number
        # so that softmax approaches 0
        w = w.masked_fill(mask_expanded == 0, -1e9)

        # Calculate softmax weights
        scores = torch.softmax(w, dim=1)

        # Compute weighted sum of hidden states
        # (batch, seq, 1) * (batch, seq, hidden) -> (batch, seq, hidden)
        # Sum over seq_len dimension -> (batch, hidden)
        pooled_output = torch.sum(scores * last_hidden_state, dim=1)

        return pooled_output


class MultiTaskRoBERTa(nn.Module):
    """
    RoBERTa-based model with Multi-Task learning heads for Toxicity and Identity detection.
    Uses Spatial Dropout and Attention Pooling for better generalization.
    """

    def __init__(
        self, model_name=MODEL_NAME, dropout_rate=DROPOUT, num_identities=None
    ):
        super().__init__()

        if num_identities is None:
            num_identities = len(IDENTITY_COLUMNS)

        # Load Transformer Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Custom Layers
        self.spatial_dropout = SpatialDropout(p=dropout_rate)
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Multi-Task Heads
        # 1. Toxicity Head (Binary Classification)
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # 2. Identity Head (Multi-label Classification)
        self.identity_head = nn.Linear(self.config.hidden_size, num_identities)

    def forward(self, input_ids, attention_mask):
        # Pass through backbone
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Spatial Dropout
        embeddings = self.spatial_dropout(last_hidden_state)

        # Apply Attention Pooling
        pooled_output = self.pooling(embeddings, attention_mask)

        # Prediction Heads
        tox_logits = self.toxicity_head(pooled_output)
        identity_logits = self.identity_head(pooled_output)

        return tox_logits, identity_logits
