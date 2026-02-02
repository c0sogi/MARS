import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to aggregate sequence of hidden states into a single embedding.
    Applies a learned attention mechanism to weight informative tokens more heavily.
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
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            feature: (batch_size, hidden_size)
        """
        # Calculate attention scores
        w = self.attention(last_hidden_state).squeeze(-1)  # (batch, seq_len)

        # Mask padding tokens so they don't contribute to the pooling
        if attention_mask is not None:
            w = w.masked_fill(attention_mask == 0, -1e9)

        # Normalize scores to probabilities
        weights = torch.softmax(w, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum of hidden states
        feature = torch.sum(last_hidden_state * weights, dim=1)  # (batch, hidden_size)
        return feature


class RegressionModel(nn.Module):
    """
    DeBERTa-v3 based model for direct score regression.
    Architecture: Backbone -> AttentionPooling -> Linear(1)
    """

    def __init__(self, model_path=None, pretrained=True):
        super().__init__()
        if model_path is None:
            model_path = Config.MODEL_PATH

        config = AutoConfig.from_pretrained(model_path)
        # Disable internal dropout for deterministic behavior and potentially better convergence
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        # Enable Gradient Checkpointing to save memory with Large models
        self.backbone.gradient_checkpointing_enable()

        self.pool = AttentionPooling(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, 1)

        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize custom layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Returns raw logits (scores).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        embeddings = self.pool(last_hidden_state, attention_mask)
        logits = self.fc(embeddings)

        return logits


class OrdinalModel(nn.Module):
    """
    DeBERTa-v3 based model for ordinal classification.
    Architecture: Backbone -> AttentionPooling -> Linear(5)
    Outputs 5 logits corresponding to P(y>1), P(y>2), ..., P(y>5).
    """

    def __init__(self, model_path=None, pretrained=True):
        super().__init__()
        if model_path is None:
            model_path = Config.MODEL_PATH

        config = AutoConfig.from_pretrained(model_path)
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.backbone = AutoModel.from_config(config)

        # Enable Gradient Checkpointing
        self.backbone.gradient_checkpointing_enable()

        self.pool = AttentionPooling(config.hidden_size)
        # 5 output neurons for the 5 thresholds between 6 classes
        self.fc = nn.Linear(config.hidden_size, 5)

        self._init_weights(self.pool)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize custom layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        Returns 5 logits for ordinal binary classification.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        embeddings = self.pool(last_hidden_state, attention_mask)
        logits = self.fc(embeddings)

        return logits
