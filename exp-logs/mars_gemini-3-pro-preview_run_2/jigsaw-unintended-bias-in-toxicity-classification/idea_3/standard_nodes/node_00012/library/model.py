import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire feature channels across the sequence dimension.
    Input shape: (batch_size, seq_len, hidden_dim)
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        # Permute to (batch_size, hidden_dim, seq_len) for Dropout2d
        # Dropout2d expects (N, C, L) or (N, C, H, W). We use (N, C, L).
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of sequence hidden states.
    Weights are learned dynamically via a small MLP.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # last_hidden_state: (batch, seq_len, hidden_dim)
        # attention_mask: (batch, seq_len)

        # Compute attention scores
        # w shape: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens (set score to -inf so softmax -> 0)
        # attention_mask is 1 for valid, 0 for pad
        mask = attention_mask.unsqueeze(-1).float()
        w = w.masked_fill(mask == 0, -1e9)

        # Compute weights
        weights = torch.softmax(w, dim=1)

        # Weighted sum of hidden states
        # (batch, seq, 1) * (batch, seq, hidden) -> (batch, seq, hidden) -> sum -> (batch, hidden)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)

        return context_vector


class MultiTaskRoberta(nn.Module):
    """
    RoBERTa-based model with Multi-Task Learning for Toxicity and Identity detection.
    Features:
    - RoBERTa backbone
    - Spatial Dropout on hidden states
    - Attention Pooling aggregation
    - Separate heads for Toxicity (target) and Identity attributes
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        num_identities: int = len(Config.IDENTITY_COLUMNS),
        dropout: float = Config.DROPOUT,
        spatial_dropout: float = Config.SPATIAL_DROPOUT,
    ):
        super().__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_dim = self.config.hidden_size

        # Custom Layers
        self.spatial_dropout = SpatialDropout(p=spatial_dropout)
        self.attention_pooling = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Prediction Heads
        # Head 1: Main Toxicity Target
        self.toxicity_head = nn.Linear(hidden_dim, 1)

        # Head 2: Auxiliary Identity Attributes
        self.identity_head = nn.Linear(hidden_dim, num_identities)

        # Initialize custom layers
        self._init_weights(self.attention_pooling)
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)

    def _init_weights(self, module):
        """Initialize weights for custom layers similar to RoBERTa."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 1. Backbone Feature Extraction
        # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Regularization (Spatial Dropout)
        embeddings = self.spatial_dropout(last_hidden_state)

        # 3. Aggregation (Attention Pooling)
        # Discards [CLS], aggregates all tokens weighted by relevance
        pooled_output = self.attention_pooling(embeddings, attention_mask)

        # 4. Final Dropout
        pooled_output = self.dropout(pooled_output)

        # 5. Multi-Task Prediction
        toxicity_logits = self.toxicity_head(pooled_output)
        identity_logits = self.identity_head(pooled_output)

        return toxicity_logits, identity_logits
