import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling to the last hidden state of a transformer.
    Computes a weighted average of token embeddings, where weights are learned.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Compute attention scores: (B, L, H) -> (B, L, 1) -> (B, L)
        w = self.attention(last_hidden_state).squeeze(-1)

        # Mask padding tokens so they don't contribute to the average
        if attention_mask is not None:
            # Set score of padding tokens to a very small number
            # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
            w = w.masked_fill(attention_mask == 0, -1e4)

        # Normalize scores to probabilities
        weights = torch.softmax(w, dim=-1)

        # Compute weighted sum: (B, L, 1) * (B, L, H) -> sum over L -> (B, H)
        feature = torch.sum(weights.unsqueeze(-1) * last_hidden_state, dim=1)
        return feature


class HybridModel(nn.Module):
    """
    A hybrid architecture combining a Transformer backbone with explicit structural features.

    Structure:
    1. Transformer Backbone (DeBERTa/RoBERTa)
    2. Attention Pooling on Last Hidden State
    3. Concatenation with Structural Features (Levenshtein, Jaccard, etc.)
    4. Classification Head
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = Config.num_classes,
        num_structural_features: int = len(Config.structural_features),
        pretrained: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Classification Head
        # Input dimension = Transformer Embedding Size + Number of Structural Features
        input_dim = self.config.hidden_size + num_structural_features
        self.fc = nn.Linear(input_dim, num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom pooling and head layers.
        Follows standard transformer initialization patterns.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, structural_features=None):
        """
        Forward pass of the model.

        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            structural_features: (B, num_structural_features) or None

        Returns:
            logits: (B, num_classes)
        """
        # 1. Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, L, H)

        # 2. Pooling
        feature = self.pooler(last_hidden_state, attention_mask)  # (B, H)

        # 3. Feature Fusion
        if structural_features is not None:
            # Concatenate dense embedding with structural scalars
            feature = torch.cat([feature, structural_features], dim=1)

        # 4. Classification Head
        logits = self.fc(feature)

        return logits
