import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Implementation of Attention Pooling (also known as Weighted Average Pooling).
    It learns a weight distribution over the sequence tokens to produce a single
    vector representation.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Calculate attention scores
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens with -inf so they don't contribute to softmax
        v = torch.tensor(float("-inf")).to(w.device)
        mask = attention_mask.unsqueeze(-1) > 0.5
        w = torch.where(mask, w, v)

        # Softmax to get normalized weights
        w = F.softmax(w, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum of hidden states
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_size] -> sum over seq_len
        return torch.sum(w * last_hidden_state, dim=1)


class DebertaV3WithFeatures(nn.Module):
    """
    Hybrid model architecture combining a DeBERTa-v3-large backbone with
    explicit structural features.

    Structure:
    1. Backbone (DeBERTa) -> Contextual Embeddings
    2. Attention Pooling -> Global Semantic Vector
    3. Concatenation (Semantic Vector + Structural Features)
    4. Classifier Head -> 5-class Logits
    """

    def __init__(
        self, num_features=6, num_classes=5, pretrained_model_name=Config.MODEL_NAME
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Enable Gradient Checkpointing to save memory with the Large model
        self.backbone.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooling = AttentionPooling(self.config.hidden_size)

        # Classification Head
        # Input dimension is the sum of the transformer hidden size and the number of manual features
        input_dim = self.config.hidden_size + num_features
        self.fc = nn.Linear(input_dim, num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.fc)
        self._init_weights(self.pooling.attention)

    def _init_weights(self, module):
        """
        Standard initialization for linear and normalization layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                self._init_weights(submodule)

    def forward(self, input_ids, attention_mask, structural_features, label=None):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Token indices [batch, seq_len].
            attention_mask (torch.Tensor): Attention mask [batch, seq_len].
            structural_features (torch.Tensor): Manual features [batch, num_features].
            label (torch.Tensor, optional): Ground truth labels (unused in forward, but kept for API consistency).

        Returns:
            torch.Tensor: Logits for the 5 classes [batch, num_classes].
        """
        # 1. Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pooling
        feature_vector = self.pooling(last_hidden_state, attention_mask)

        # 3. Feature Fusion
        # Concatenate the semantic embedding with the structural features
        # Ensure structural_features is on the correct device and dtype
        combined_vector = torch.cat([feature_vector, structural_features], dim=1)

        # 4. Classification
        logits = self.fc(combined_vector)

        return logits
