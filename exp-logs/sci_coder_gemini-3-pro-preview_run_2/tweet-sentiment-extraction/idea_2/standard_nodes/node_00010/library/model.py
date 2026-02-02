import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class TweetModel(nn.Module):
    """
    Transformer-based model for Span Prediction in Tweet Sentiment Extraction.
    Uses a pre-trained backbone (e.g., RoBERTa) followed by a linear head
    to predict start and end token indices.
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing model parameters.
        """
        super(TweetModel, self).__init__()
        self.config = config

        # Load Transformer Configuration and Backbone
        transformer_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(
            config.MODEL_NAME, config=transformer_config
        )

        # Prediction Head
        # Standard dropout for transformers
        self.drop = nn.Dropout(0.1)
        # Output layer: projects hidden state to 2 logits (start, end)
        self.out = nn.Linear(transformer_config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.out)

    def _init_weights(self, module):
        """
        Initialize weights for the linear layer.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]

        Returns:
            start_logits (torch.Tensor): Logits for start index [batch_size, seq_len]
            end_logits (torch.Tensor): Logits for end index [batch_size, seq_len]
        """
        # Get hidden states from the backbone
        # outputs[0] corresponds to the last_hidden_state
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]

        # Apply dropout and linear projection
        x = self.drop(last_hidden_state)
        logits = self.out(x)  # Shape: [batch_size, seq_len, 2]

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension
        start_logits = start_logits.squeeze(-1)  # Shape: [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)  # Shape: [batch_size, seq_len]

        return start_logits, end_logits
