import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TweetLoss(nn.Module):
    """
    Loss function for Tweet Sentiment Extraction.
    Uses CrossEntropyLoss with Label Smoothing to mitigate boundary noise.
    Cite solution_lesson_node_00010
    """

    def __init__(self):
        super(TweetLoss, self).__init__()
        # Use label smoothing of 0.1 to prevent overconfidence on fuzzy boundaries
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        """
        Calculates the loss.
        """
        start_loss = self.ce_loss(start_logits, start_positions)
        end_loss = self.ce_loss(end_logits, end_positions)
        return (start_loss + end_loss) / 2
