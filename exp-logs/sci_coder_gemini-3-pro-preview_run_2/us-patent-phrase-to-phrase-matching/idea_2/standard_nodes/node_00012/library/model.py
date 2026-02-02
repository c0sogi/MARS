import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to dynamically weight hidden states.
    Computes a weighted average of the token embeddings based on their importance.
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # Calculate attention weights
        # w shape: (batch_size, seq_len, 1)
        w = self.attention(last_hidden_state)

        # Mask padding tokens so they don't contribute to the softmax
        # attention_mask shape: (batch_size, seq_len)
        # We expand mask to match w dimensions
        mask = attention_mask.unsqueeze(-1).expand(w.shape)
        w = w.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get normalized weights
        w = torch.softmax(w, dim=1)

        # Compute weighted sum of hidden states
        # weighted_avg shape: (batch_size, hidden_size)
        weighted_avg = torch.sum(w * last_hidden_state, dim=1)

        return weighted_avg


class CustomDeberta(nn.Module):
    """
    Context-Enriched DeBERTa-v3-Large Cross-Encoder.
    Uses Attention Pooling and a 5-class Classification Head.
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=True,
    ):
        super(CustomDeberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Enable Gradient Checkpointing for memory efficiency with Large models
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Initialize Attention Pooling
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout (MSD)
        # Acts as an internal ensemble to improve generalization
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(5)])

        # Classification Head
        # Maps pooled embedding to 5 logits corresponding to scores {0.0, 0.25, 0.5, 0.75, 1.0}
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """Initialize weights for the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Pass through DeBERTa backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state sequence
        # shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Apply Attention Pooling
        # shape: (batch_size, hidden_size)
        feature = self.pooler(last_hidden_state, attention_mask)

        # Multi-Sample Dropout Forward Pass
        # Average the logits from multiple dropout masks
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.fc(dropout(feature))
            else:
                logits += self.fc(dropout(feature))

        logits /= len(self.dropouts)

        return logits
