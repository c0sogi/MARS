import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    Neural network model for Tweet Sentiment Extraction.
    Uses a DeBERTa-v3-large backbone with a simple linear prediction head.
    """

    def __init__(self):
        super(TweetModel, self).__init__()

        # Load configuration and backbone model
        self.config = AutoConfig.from_pretrained(
            Config.model_name, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Simple Linear Head: Projects hidden state to 2 logits (start and end)
        # We strictly avoid complex heads (CNN, Pooling) as per the strategy.
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        """
        Initializes weights for the linear head using the backbone's initializer range.
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
            start_logits (torch.Tensor): Logits for the start position of the selected text.
            end_logits (torch.Tensor): Logits for the end position of the selected text.
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the last hidden state: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Project to logits: [batch_size, seq_len, 2]
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        # shape: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
