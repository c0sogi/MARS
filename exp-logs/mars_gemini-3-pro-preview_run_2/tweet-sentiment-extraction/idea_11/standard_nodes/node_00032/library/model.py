import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TweetModel(nn.Module):
    """
    A PyTorch module for Tweet Sentiment Extraction.
    Wraps a HuggingFace Transformer backbone (e.g., DeBERTa-v3, RoBERTa)
    and adds a simple linear head for span prediction.
    """

    def __init__(self, model_name):
        """
        Initializes the TweetModel.

        Args:
            model_name (str): The name or path of the pre-trained transformer model.
        """
        super(TweetModel, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        # Disable hidden states output to save memory
        self.config.output_hidden_states = False

        # Load the pre-trained backbone
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        # Determine dropout probability from config (default to 0.1 if not present)
        dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.drop = nn.Dropout(dropout_prob)

        # Simple Linear Head: Project hidden size to 2 (start_logit, end_logit)
        # We avoid complex heads (CNN, RNN) to preserve token-level signal
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        """
        Initialize the weights of the linear head.
        Standard transformers initialization: Normal(0, 0.02) for weights, 0 for bias.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): Mask to avoid attention on padding. Shape: (batch_size, seq_len)
            token_type_ids (torch.Tensor, optional): Segment token indices. Shape: (batch_size, seq_len)

        Returns:
            start_logits (torch.Tensor): Logits for the start position. Shape: (batch_size, seq_len)
            end_logits (torch.Tensor): Logits for the end position. Shape: (batch_size, seq_len)
        """
        # Pass inputs through the backbone
        # AutoModel handles the specific logic for DeBERTa vs RoBERTa
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the last hidden state: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout for regularization
        sequence_output = self.drop(sequence_output)

        # Pass through the linear head -> (batch_size, sequence_length, 2)
        logits = self.qa_outputs(sequence_output)

        # Split the last dimension into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
