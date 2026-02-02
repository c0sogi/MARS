import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Implementation of Attention Pooling.
    Aggregates the sequence of hidden states into a single vector using a learnable attention mechanism.
    This allows the model to dynamically weight informative tokens over padding or noise.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            pooled_output: Tensor of shape (batch_size, hidden_size)
        """
        # Calculate attention scores
        w = self.attention(last_hidden_state)  # (batch_size, seq_len, 1)

        # Mask padding tokens by setting their weights to a very large negative value
        # attention_mask is 1 for valid tokens, 0 for padding
        min_value = torch.finfo(w.dtype).min
        w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, min_value)

        # Apply softmax to get normalized attention weights
        weights = torch.softmax(w, dim=1)  # (batch_size, seq_len, 1)

        # Compute weighted sum of hidden states
        # Broadcasting weights across the hidden_size dimension
        pooled_output = torch.sum(
            weights * last_hidden_state, dim=1
        )  # (batch_size, hidden_size)

        return pooled_output


class EssayModel(nn.Module):
    """
    DeBERTa-v3-large based model for Essay Scoring.

    Architecture:
    1. Backbone: DeBERTa-v3-large (Feature Extractor)
    2. Pooling: Attention Pooling (Aggregation)
    3. Head: Linear Regression (Score Prediction)

    Returns both the score logits and the pooled embeddings for downstream stacking.
    """

    def __init__(self, config_path=None, pretrained=True):
        super().__init__()
        self.config = Config

        # Load Configuration
        if config_path is None:
            model_config = AutoConfig.from_pretrained(self.config.model_name)
        else:
            model_config = torch.load(config_path)

        # Update config for regression task
        # We disable dropout in the backbone for deterministic regression fine-tuning
        # and to prevent noise during the embedding generation phase.
        model_config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "num_labels": 1,
            }
        )

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.config.model_name, config=model_config
            )
        else:
            self.backbone = AutoModel.from_config(model_config)

        # Enable Gradient Checkpointing if configured (saves VRAM for Large models)
        if self.config.use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Initialize Custom Layers
        self.pooling = AttentionPooling(model_config.hidden_size)
        self.fc = nn.Linear(model_config.hidden_size, 1)

        # Initialize weights for the new layers to match backbone statistics
        self._init_weights(self.pooling.attention)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.backbone.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def forward(self, input_ids, attention_mask, meta_features=None, **kwargs):
        """
        Forward pass of the model.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            meta_features: (batch_size, num_features)
                           Note: Meta-features are NOT used in the backbone forward pass.
                           They are reserved for the LightGBM stacking head to ensure
                           the backbone focuses purely on semantic text features.
            **kwargs: Catch-all for other arguments (e.g., labels, essay_id)

        Returns:
            dict: {
                "logits": Tensor of shape (batch_size),
                "embeddings": Tensor of shape (batch_size, hidden_size)
            }
        """
        # 1. Backbone Extraction
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Attention Pooling
        embeddings = self.pooling(last_hidden_state, attention_mask)

        # 3. Regression Head
        logits = self.fc(embeddings)

        return {
            "logits": logits.squeeze(-1),  # Flatten to (batch_size)
            "embeddings": embeddings,
        }
