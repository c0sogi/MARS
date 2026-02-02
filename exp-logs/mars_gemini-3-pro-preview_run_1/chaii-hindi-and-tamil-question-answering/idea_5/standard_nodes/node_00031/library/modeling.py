import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiTaskQAModel(nn.Module):
    """
    Multi-Task Question Answering Model.

    Architecture:
        - Backbone: XLM-Roberta-Large (pre-trained)
        - Head 1: Span Prediction (Linear layer projecting to 2 logits: start, end)
        - Head 2: Relevance Classification (Linear layer on [CLS] token projecting to 1 logit)

    This architecture allows the model to simultaneously predict the answer span
    and determine if the current sliding window is relevant (contains the answer).
    """

    def __init__(self, config):
        """
        Initializes the model architecture.

        Args:
            config: Configuration object containing 'model_name' (e.g., 'xlm-roberta-large').
        """
        super(MultiTaskQAModel, self).__init__()
        self.config = config

        # Load the configuration for the backbone
        self.model_config = AutoConfig.from_pretrained(config.model_name)

        # Initialize the backbone model
        # We use AutoModel to get the raw hidden states
        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=self.model_config
        )

        # Hidden size of the transformer output
        hidden_size = self.model_config.hidden_size

        # =====================================================================
        # HEAD 1: SPAN PREDICTION
        # =====================================================================
        # Projects hidden states to 2 values per token: (start_logit, end_logit)
        self.qa_outputs = nn.Linear(hidden_size, 2)

        # =====================================================================
        # HEAD 2: RELEVANCE CLASSIFICATION
        # =====================================================================
        # Projects the [CLS] token representation to a single logit (binary classification)
        # This helps filter out windows that do not contain the answer.
        self.relevance_classifier = nn.Linear(hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
                                           Shape: (batch_size, sequence_length)
            token_type_ids (torch.Tensor, optional): Segment token indices.
                                                     Shape: (batch_size, sequence_length)

        Returns:
            start_logits (torch.Tensor): Logits for the start position. Shape: (batch_size, sequence_length)
            end_logits (torch.Tensor): Logits for the end position. Shape: (batch_size, sequence_length)
            relevance_logits (torch.Tensor): Logits for window relevance. Shape: (batch_size,)
        """
        # Pass inputs through the backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get the last hidden state: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # ---------------------------------------------------------------------
        # 1. Compute Span Logits
        # ---------------------------------------------------------------------
        # (batch_size, sequence_length, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # ---------------------------------------------------------------------
        # 2. Compute Relevance Logits
        # ---------------------------------------------------------------------
        # Extract the [CLS] token representation (index 0)
        # (batch_size, hidden_size)
        cls_output = sequence_output[:, 0, :]

        # Project to scalar logit
        # (batch_size, 1) -> (batch_size,)
        relevance_logits = self.relevance_classifier(cls_output).squeeze(-1)

        return start_logits, end_logits, relevance_logits
