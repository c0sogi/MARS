import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from library.config import Config


class CrossEncoder(nn.Module):
    """
    A Cross-Encoder model that processes the context, anchor, and target jointly.
    Cite solution_lesson_node_00001: Addresses the limitation of Bi-Encoders by allowing
    full token-level interaction.
    """

    def __init__(self, model_name=None):
        super(CrossEncoder, self).__init__()

        if model_name is None:
            model_name = Config.model_name

        # Cite solution_lesson_node_00005: Leverage Task-Specific Transformer Heads
        # Use AutoModelForSequenceClassification to include standard pooling and dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the Cross-Encoder.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract logits from the sequence classification head
        logits = outputs.logits

        # Apply sigmoid to map to [0, 1] range and squeeze to match label shape
        score = self.sigmoid(logits).squeeze(1)

        return score
