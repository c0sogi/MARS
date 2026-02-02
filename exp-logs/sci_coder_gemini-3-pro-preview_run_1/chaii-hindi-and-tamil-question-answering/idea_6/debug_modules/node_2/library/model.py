import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomXLMRoberta(nn.Module):
    """
    Multi-Task XLM-R Architecture for Question Answering.

    Backbone: XLM-Roberta Large
    Heads:
        1. Span Head: Predicts start and end logits for answer extraction.
        2. Relevance Head: Binary classification on [CLS] token to predict answer presence.
    """

    def __init__(self, model_name):
        """
        Args:
            model_name (str): The name or path of the pre-trained model (e.g., 'xlm-roberta-large').
        """
        super(CustomXLMRoberta, self).__init__()

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load backbone model
        # add_pooling_layer=False because we manually extract CLS for relevance
        # and use sequence output for spans.
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

        # Span Head: Maps hidden_size -> 2 (start_logit, end_logit)
        self.span_head = nn.Linear(self.config.hidden_size, 2)

        # Relevance Head: Maps hidden_size -> 1 (logit for "answer exists")
        self.relevance_head = nn.Linear(self.config.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Initialize weights for the new heads
        self._init_weights(self.span_head)
        self._init_weights(self.relevance_head)

    def _init_weights(self, module):
        """
        Initializes weights for the task-specific heads using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start position of the answer.
            end_logits (torch.Tensor): Logits for the end position of the answer.
            relevance_logits (torch.Tensor): Logits indicating if the answer is present in the window.
        """
        # Pass through the backbone
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Shape: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout to the features
        sequence_output = self.dropout(sequence_output)

        # --- 1. Span Prediction ---
        # Project to 2 dimensions (start, end)
        span_logits = self.span_head(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = span_logits.split(1, dim=-1)

        # Squeeze to remove the last dimension: (batch_size, seq_len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # --- 2. Relevance Prediction ---
        # Extract [CLS] token representation (Index 0 for XLM-R)
        cls_token_state = sequence_output[:, 0, :]

        # Project to 1 dimension: (batch_size, 1)
        relevance_logits = self.relevance_head(cls_token_state)

        # Squeeze to (batch_size)
        relevance_logits = relevance_logits.squeeze(-1)

        return start_logits, end_logits, relevance_logits
