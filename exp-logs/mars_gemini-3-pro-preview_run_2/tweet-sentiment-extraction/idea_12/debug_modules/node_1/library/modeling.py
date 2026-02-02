import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TweetModel(nn.Module):
    """
    A wrapper class for Hugging Face transformer models (DeBERTa, RoBERTa, etc.)
    with a simple linear head for span extraction.
    """

    def __init__(self, model_name):
        super(TweetModel, self).__init__()
        # Load configuration and backbone model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        # Simple Linear Head: Dropout + Linear Projection
        # We project the hidden state to 2 values: start_logit and end_logit
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.out)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
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
            start_logits (torch.Tensor): Logits for the start position of the span.
            end_logits (torch.Tensor): Logits for the end position of the span.
        """
        # Pass inputs to the backbone
        # We pass token_type_ids explicitly as DeBERTa uses them.
        # RoBERTa accepts them (if vocab size matches) or ignores them.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the last hidden state: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout for regularization
        sequence_output = self.drop(sequence_output)

        # Project to logits: (batch_size, sequence_length, 2)
        logits = self.out(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Remove the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
