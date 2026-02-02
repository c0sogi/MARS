import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM
from library.config import Config


class LocatorNetwork(nn.Module):
    """
    A token-level classification model to identify where a word is missing.
    Uses a Transformer backbone with a custom linear head.
    """

    def __init__(self):
        super(LocatorNetwork, self).__init__()
        # Load pre-trained backbone (e.g., DistilBERT)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_BACKBONE)

        # Hidden size of the transformer (e.g., 768 for base models)
        self.hidden_size = self.backbone.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Pointer Head:
        # Projects hidden state to 1 scalar score per token.
        # Cite solution_lesson_node_00001: Use Pointer Head for Sparse Localization.
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Shape (batch_size, seq_len)

        Returns:
            logits (torch.Tensor): Shape (batch_size, seq_len)
        """
        # Get hidden states from backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # Project to logits (Batch, Seq_Len, 1)
        logits = self.classifier(sequence_output)

        # Squeeze to (Batch, Seq_Len)
        logits = logits.squeeze(-1)

        # Mask padding tokens to -inf so they don't affect Softmax/Loss
        if attention_mask is not None:
            # attention_mask is 1 for keep, 0 for pad.
            # We want pads to be -inf.
            logits = logits + (1.0 - attention_mask) * -10000.0

        return logits


class FillerNetwork(nn.Module):
    """
    A Masked Language Model to predict the missing word.
    Wraps a pre-trained model with an MLM head.
    """

    def __init__(self):
        super(FillerNetwork, self).__init__()
        # Load pre-trained MLM model (backbone + MLM head)
        self.backbone = AutoModelForMaskedLM.from_pretrained(Config.MODEL_BACKBONE)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Shape (batch_size, seq_len)

        Returns:
            logits (torch.Tensor): Shape (batch_size, seq_len, vocab_size)
        """
        # The AutoModelForMaskedLM handles the projection to vocabulary size internally
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Return logits over the vocabulary
        return outputs.logits
