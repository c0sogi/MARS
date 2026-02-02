import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling mechanism to weight token embeddings dynamically.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Calculate attention weights
        # w: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Squeeze to (batch_size, seq_len)
        w = w.squeeze(-1)

        # Mask padding tokens by setting their weights to a large negative value
        if attention_mask is not None:
            # attention_mask is 1 for tokens, 0 for padding
            # We want to make padding positions -inf so softmax makes them 0
            w = w.masked_fill(attention_mask == 0, -1e4)

        # Apply softmax to get normalized weights
        weights = torch.softmax(w, dim=-1)

        # Compute weighted sum
        # weights: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # result: (batch_size, hidden_size)
        weighted_embeddings = torch.sum(
            last_hidden_state * weights.unsqueeze(-1), dim=1
        )

        return weighted_embeddings


class MeanPooling(nn.Module):
    """
    Standard Mean Pooling (Average of non-padding tokens).
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class EssayScorerModel(nn.Module):
    """
    Essay Scoring Model using DeBERTa-v3-Large backbone with Attention Pooling.
    """

    def __init__(self, cfg=Config, pretrained=True):
        super(EssayScorerModel, self).__init__()
        self.cfg = cfg

        # Load configuration from the backbone
        self.model_config = AutoConfig.from_pretrained(cfg.model_name)
        self.hidden_size = self.model_config.hidden_size

        # Initialize Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(cfg.model_name)
        else:
            self.backbone = AutoModel.from_config(self.model_config)

        # Enable Gradient Checkpointing to save memory with Large models + Long sequences
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            print(f"Enabling gradient checkpointing for {cfg.model_name}")
            self.backbone.gradient_checkpointing_enable()

        # Initialize Pooling Strategy
        if cfg.pooling_type == "attention":
            self.pooler = AttentionPooling(self.hidden_size)
        elif cfg.pooling_type == "mean":
            self.pooler = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling type: {cfg.pooling_type}")

        # Regression Head
        self.fc = nn.Linear(self.hidden_size, cfg.target_size)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, target_size)
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Apply Pooling
        feature = self.pooler(last_hidden_state, attention_mask)

        # Apply Regression Head
        output = self.fc(feature)

        return output
