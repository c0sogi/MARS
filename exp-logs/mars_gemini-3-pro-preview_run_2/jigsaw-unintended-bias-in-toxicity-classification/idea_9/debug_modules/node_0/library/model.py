import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling to the last hidden state of the transformer.
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
        # Calculate attention weights
        w = self.attention(last_hidden_state).float()

        # Mask padding tokens so they don't contribute to the softmax
        w[attention_mask == 0] = float("-inf")

        # Normalize weights
        w = torch.softmax(w, dim=1)

        # Compute weighted sum
        c = torch.sum(last_hidden_state * w, dim=1)
        return c


class ToxicityModel(nn.Module):
    """
    Multi-Task RoBERTa-Large model with Attention Pooling and Multi-Sample Dropout.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Backbone
        self.roberta = AutoModel.from_pretrained(config.MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size

        # Pooling
        self.pooler = AttentionPooling(hidden_size)

        # Multi-Sample Dropout
        # We create multiple dropout layers to be applied in parallel
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.DROPOUT_RATE) for _ in range(config.DROPOUT_SAMPLES)]
        )

        # Heads
        self.toxicity_head = nn.Linear(hidden_size, 1)
        self.identity_head = nn.Linear(hidden_size, len(config.IDENTITY_COLUMNS))

        # Initialize weights for custom layers
        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.pooler)

    def _init_weights(self, module):
        """
        Initialize weights for Linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.roberta.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Feature Extraction
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state

        # Pooling
        pooled_output = self.pooler(last_hidden_state, attention_mask)

        # Multi-Sample Dropout for Toxicity Head
        # Pass the pooled output through multiple dropout masks and average the logits
        tox_logits = 0
        for dropout in self.dropouts:
            tox_logits += self.toxicity_head(dropout(pooled_output))
        tox_logits /= len(self.dropouts)

        # Identity Head (Auxiliary)
        # Use the first dropout layer for the auxiliary task
        ident_logits = self.identity_head(self.dropouts[0](pooled_output))

        return tox_logits, ident_logits
