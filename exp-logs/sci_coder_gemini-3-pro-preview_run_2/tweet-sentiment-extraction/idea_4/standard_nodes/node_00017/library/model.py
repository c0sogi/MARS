import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural network model for Tweet Sentiment Extraction.
    Uses a DeBERTa-v3-large backbone with a token-level prediction head.
    """

    def __init__(self, conf=Config):
        """
        Initializes the model architecture.

        Args:
            conf (Config): Configuration class containing model parameters.
        """
        super(TweetModel, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(
            conf.MODEL_NAME, output_hidden_states=True
        )

        # Load pre-trained backbone
        self.model = AutoModel.from_pretrained(conf.MODEL_NAME, config=self.config)

        # Regularization
        self.drop = nn.Dropout(0.1)

        # Prediction Head: Projects hidden state to 2 logits (start, end) per token
        self.l0 = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.l0)

    def _init_weights(self, module):
        """
        Initializes weights for the linear layer.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.

        Returns:
            start_logits (torch.Tensor): Logits for the start token index (batch_size, seq_len).
            end_logits (torch.Tensor): Logits for the end token index (batch_size, seq_len).
        """
        # Pass inputs through the backbone
        # DeBERTa-v3 handles token_type_ids to distinguish between sentiment and text segments
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Retrieve the sequence of hidden states from the last layer
        # Shape: (batch_size, sequence_length, hidden_size)
        out = outputs.last_hidden_state

        # Apply dropout
        out = self.drop(out)

        # Project to start/end logits
        # Shape: (batch_size, sequence_length, 2)
        logits = self.l0(out)

        # Split into separate tensors for start and end logits
        # Each shape: (batch_size, sequence_length, 1)
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
