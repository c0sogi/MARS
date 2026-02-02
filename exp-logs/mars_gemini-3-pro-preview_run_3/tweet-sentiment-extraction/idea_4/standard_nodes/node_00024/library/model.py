import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural Network architecture for Sentiment Extraction.

    Backbone: microsoft/deberta-v3-large
    Head: Simple Linear Layer (Hidden Size -> 2)

    This model predicts the start and end indices of the selected text span
    directly from the last hidden state of the transformer.
    """

    def __init__(self):
        super().__init__()
        # Load configuration from the pre-trained model
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)

        # Initialize the backbone
        # We only need the last hidden state, so output_hidden_states can be False (default)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Multi-Sample Dropout for regularization
        # Creates an implicit ensemble to improve generalization
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(5)])

        # Simple Linear Head: Projects hidden_size to 2 outputs (start_logit, end_logit)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """
        Initializes weights for the linear layer using the backbone's initializer range.
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
            start_logits (torch.Tensor): Logits for the start index of the span. Shape: (batch_size, seq_len)
            end_logits (torch.Tensor): Logits for the end index of the span. Shape: (batch_size, seq_len)
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the sequence of hidden states from the last layer
        # Shape: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply Multi-Sample Dropout
        # Average the logits across different dropout masks
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(sequence_output))
            else:
                logits += self.classifier(dropout(sequence_output))

        logits /= len(self.dropouts)

        # Split the last dimension to separate start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
