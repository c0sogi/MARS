import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from library.config import Config


class CrossEncoder(nn.Module):
    """
    Cite solution_lesson_node_00002: Cross-Encoder Architecture.
    Encodes the concatenated pair (Context+Anchor, Target) jointly.
    """

    def __init__(self, model_name=None):
        super(CrossEncoder, self).__init__()
        if model_name is None:
            model_name = Config.model_name

        # Load model with a classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the Cross-Encoder.

        Args:
            input_ids (torch.Tensor): Input IDs for concatenated sequences.
            attention_mask (torch.Tensor): Attention masks.

        Returns:
            torch.Tensor: Predicted similarity scores [Batch].
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply sigmoid to map logits to [0, 1] range
        return torch.sigmoid(outputs.logits).squeeze()
