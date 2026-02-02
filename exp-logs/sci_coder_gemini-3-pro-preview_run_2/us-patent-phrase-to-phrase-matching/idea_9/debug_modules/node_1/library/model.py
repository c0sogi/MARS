import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling to the last hidden state of the transformer.
    Computes a weighted average of token embeddings where weights are learned.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # Calculate attention scores: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens (set to -inf so softmax becomes 0)
        # attention_mask is (batch, seq_len), unsqueeze to (batch, seq_len, 1)
        w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

        # Normalize weights: (batch, seq_len, 1)
        weights = torch.softmax(w, dim=1)

        # Weighted sum: (batch, hidden_size)
        # weights * last_hidden_state broadcasts to (batch, seq_len, hidden_size)
        # Sum over seq_len dimension
        return torch.sum(weights * last_hidden_state, dim=1)


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa model with Attention Pooling and Internal Feature Fusion.
    """

    def __init__(self, pretrained_model_name=CFG.model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Initialize backbone
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Enable Gradient Checkpointing if configured
        if CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Pooling Layer
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Feature Fusion Dimension
        # We concatenate the pooled text embedding with the structural features
        self.fc_input_dim = self.config.hidden_size + CFG.num_structural_features

        # Classification Head
        self.fc = nn.Linear(self.fc_input_dim, CFG.num_classes)
        self.dropout = nn.Dropout(CFG.fc_dropout)

        # Initialize weights for custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, structural_features):
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of token ids (batch, seq_len)
            attention_mask: Tensor of attention masks (batch, seq_len)
            structural_features: Tensor of explicit features (batch, num_structural_features)

        Returns:
            logits: Tensor of class logits (batch, num_classes)
        """
        # Get transformer outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        feature = self.pooler(last_hidden_state, attention_mask)

        # Internal Fusion: Concatenate text embedding with structural features
        combined_feature = torch.cat([feature, structural_features], dim=1)

        # Apply Dropout and Classification Head
        output = self.dropout(combined_feature)
        logits = self.fc(output)

        return logits
