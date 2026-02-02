import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Model class for Sentiment Extraction using DeBERTa-v3-Large backbone.

    Architecture:
    - Backbone: DeBERTa-v3-Large (pretrained)
    - Head: Linear layer (hidden_size -> 2) for start/end logits
    - Dropout applied before the head.
    """

    def __init__(self, conf=Config):
        """
        Initialize the model.

        Args:
            conf: Configuration object containing model settings.
                  Defaults to the imported Config class.
        """
        super(TweetModel, self).__init__()
        self.conf = conf

        # Load Configuration
        self.config = AutoConfig.from_pretrained(
            conf.model_name, output_hidden_states=True
        )

        # Load Backbone
        self.model = AutoModel.from_pretrained(conf.model_name, config=self.config)

        # Head Components
        self.dropout = nn.Dropout(conf.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        # Initialize Head Weights
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head.
        Uses the initializer range from the backbone config.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask. Shape: (batch_size, seq_len)

        Returns:
            start_logits (torch.Tensor): Logits for start position. Shape: (batch_size, seq_len)
            end_logits (torch.Tensor): Logits for end position. Shape: (batch_size, seq_len)
        """
        # Backbone Forward
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # We use the last hidden state for token classification
        sequence_output = outputs.last_hidden_state

        # Apply Dropout
        sequence_output = self.dropout(sequence_output)

        # Linear Projection to (batch_size, seq_len, 2)
        logits = self.classifier(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension -> (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
