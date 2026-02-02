import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class XLMRobertaForQA(nn.Module):
    """
    XLM-Roberta model for Question Answering with an auxiliary relevance head.

    Architecture:
    - Backbone: XLM-Roberta Large
    - Head 1 (Span): Linear layer predicting start and end logits for each token.
    - Head 2 (Relevance): Linear layer on the [CLS] token predicting if the answer exists in the context.
    """

    def __init__(self, model_name: str):
        """
        Initializes the model architecture.

        Args:
            model_name (str): The name or path of the pre-trained model (e.g., 'xlm-roberta-large').
        """
        super(XLMRobertaForQA, self).__init__()

        # Load configuration and backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)

        # Span Prediction Head: Predicts (start_logit, end_logit) for each token
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Relevance Head: Binary classification on CLS token (Answerable vs Not Answerable)
        self.relevance_head = nn.Linear(self.config.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Initialize weights for the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.relevance_head)

    def _init_weights(self, module):
        """
        Initialize weights for the specific module using the backbone's initializer range.
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
            dict: A dictionary containing:
                - 'start_logits': Logits for the start position of the answer.
                - 'end_logits': Logits for the end position of the answer.
                - 'relevance_logits': Logit for the relevance classification (answer presence).
        """
        # Pass through backbone
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Sequence output: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # --- Head 1: Span Prediction ---
        # (batch_size, seq_len, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch_size, seq_len)

        # --- Head 2: Relevance Prediction ---
        # Extract [CLS] token representation (index 0)
        cls_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)

        # Predict relevance logit
        relevance_logits = self.relevance_head(cls_output).squeeze(-1)  # (batch_size,)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "relevance_logits": relevance_logits,
        }
