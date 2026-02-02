import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa-v3-large architecture for Phrase Similarity.

    Features:
    - Backbone: microsoft/deberta-v3-large
    - Multi-Layer Fusion: Concatenates [CLS] tokens from the last 4 layers.
    - Weighted Projection: Fuses the concatenated features via a Linear layer.
    - Head: 5-class classification output.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True

        # Initialize Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        self.hidden_size = self.config.hidden_size

        # --- Classification Head ---
        self.dropout = nn.Dropout(Config.fc_dropout)
        self.fc = nn.Linear(self.hidden_size, Config.target_size)

        # Initialize weights for custom layers
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initializes weights for the custom linear layers using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits of shape (batch_size, 5).
        """
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token from the last hidden state
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]

        # Apply Classification Head
        x = self.dropout(cls_token)
        logits = self.fc(x)

        return logits
