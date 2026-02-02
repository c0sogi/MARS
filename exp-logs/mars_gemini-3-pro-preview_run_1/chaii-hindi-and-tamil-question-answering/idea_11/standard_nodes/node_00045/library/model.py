import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomXLMRoberta(nn.Module):
    """
    Custom XLM-Roberta model for Question Answering with Multi-Task Heads.

    This architecture uses a shared backbone (XLM-R Large) and splits into:
    1. SpanHead: Predicts start and end logits for answer extraction.
    2. RelevanceHead: Predicts a binary logit indicating if the answer is present in the window.
    """

    def __init__(self, model_name=Config.MODEL_NAME):
        """
        Initializes the model architecture.

        Args:
            model_name (str): The name or path of the pre-trained model to load.
                              Defaults to the value in Config.MODEL_NAME.
        """
        super(CustomXLMRoberta, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load Backbone Model
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)

        # Span Head: Predicts start and end scores for each token (Hidden -> 2)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Relevance Head: Predicts if the answer is in the window (Hidden -> 1)
        # Applied to the [CLS] token representation (index 0)
        self.relevance_classifier = nn.Linear(self.config.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Initialize weights for the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.relevance_classifier)

    def _init_weights(self, module):
        """
        Initializes weights for the custom heads using the backbone's initializer range.

        Args:
            module (nn.Module): The module to initialize.
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
            start_logits (torch.Tensor): Logits for the start position of the answer. Shape (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for the end position of the answer. Shape (Batch, Seq_Len)
            relevance_logits (torch.Tensor): Logits for the relevance of the window. Shape (Batch, 1)
        """
        # Pass through the backbone
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Sequence output: (Batch, Seq_Len, Hidden)
        sequence_output = outputs.last_hidden_state

        # CLS token output: (Batch, Hidden) - typically the first token in XLM-R
        cls_output = sequence_output[:, 0, :]

        # Apply Dropout
        sequence_output = self.dropout(sequence_output)
        cls_output = self.dropout(cls_output)

        # 1. Span Prediction
        # Project to 2 dimensions (start, end)
        logits = self.qa_outputs(sequence_output)  # (Batch, Seq_Len, 2)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the last dimension to get (Batch, Seq_Len)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 2. Relevance Prediction
        # Project CLS token to 1 dimension (Batch, 1)
        relevance_logits = self.relevance_classifier(cls_output)

        return start_logits, end_logits, relevance_logits
