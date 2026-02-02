import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TweetModel(nn.Module):
    """
    TweetModel wraps a pre-trained transformer backbone (DeBERTa-v3) with a
    span prediction head (Pointer Network formulation).
    """

    def __init__(self, model_name, dropout=0.1):
        """
        Initializes the model architecture.

        Args:
            model_name (str): The name or path of the pre-trained model (e.g., 'microsoft/deberta-v3-base').
            dropout (float): The dropout probability.
        """
        super(TweetModel, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Span Prediction Head: Projects hidden size to 2 (Start Logit, End Logit)
        self.head = nn.Linear(self.config.hidden_size, 2)

        # Loss Function: Cross Entropy with Label Smoothing as per strategy
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Initialize weights for the custom head
        self._init_weights(self.head)

    def _init_weights(self, module):
        """
        Initializes weights for the custom linear head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            start_positions (torch.Tensor, optional): Ground truth start indices for loss calculation.
            end_positions (torch.Tensor, optional): Ground truth end indices for loss calculation.

        Returns:
            tuple: (loss, start_logits, end_logits) if targets are provided, otherwise (start_logits, end_logits).
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the last hidden state: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Apply dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # Project to logits: (batch_size, seq_len, 2)
        logits = self.head(last_hidden_state)

        # Split into start and end logits: (batch_size, seq_len, 1) each
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate loss if targets are provided
        if start_positions is not None and end_positions is not None:
            start_loss = self.loss_fn(start_logits, start_positions)
            end_loss = self.loss_fn(end_logits, end_positions)
            total_loss = start_loss + end_loss
            return total_loss, start_logits, end_logits

        return start_logits, end_logits
