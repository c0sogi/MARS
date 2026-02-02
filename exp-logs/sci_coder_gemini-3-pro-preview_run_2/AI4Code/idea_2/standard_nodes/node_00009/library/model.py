import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import MODEL_NAME


class ContextAwareRanker(nn.Module):
    """
    A Transformer-based regression model for ranking notebook cells.

    Architecture:
    - Backbone: Pre-trained Transformer (e.g., DistilRoBERTa)
    - Head: Dropout + Linear layer on top of the [CLS] token

    The model predicts a scalar score (rank) for a given input sequence
    (Markdown content + Code context).
    """

    def __init__(self, model_name=MODEL_NAME, dropout_rate=0.1):
        super(ContextAwareRanker, self).__init__()

        # Load configuration and pre-trained backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Regression Head
        # The backbone output dimension is usually 768 for base models
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.
            labels (torch.Tensor, optional): Target ranks for training.

        Returns:
            dict: Contains 'logits' (predicted ranks) and 'loss' (if labels are provided).
        """
        # Pass inputs through the Transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token representation (index 0)
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply the regression head
        x = self.dropout(cls_embedding)
        logits = self.regressor(x)

        # Squeeze to shape (batch_size,) for scalar regression
        logits = logits.squeeze(-1)

        result = {"logits": logits}

        # Calculate loss if labels are provided (MSE for regression)
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result
