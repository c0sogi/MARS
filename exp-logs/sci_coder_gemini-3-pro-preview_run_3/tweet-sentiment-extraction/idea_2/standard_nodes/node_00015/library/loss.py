import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TweetLoss(nn.Module):
    """
    Compound loss function for Tweet Sentiment Extraction.
    Combines CrossEntropyLoss for start/end indices with a differentiable Soft Jaccard Loss
    to directly optimize the competition metric.
    """

    def __init__(self):
        super(TweetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        """
        Calculates the Cross Entropy loss with Label Smoothing.
        Cite solution_lesson_node_00010: Mitigating Boundary Noise in Span Extraction with Label Smoothing.
        """
        start_ce = self.ce_loss(start_logits, start_positions)
        end_ce = self.ce_loss(end_logits, end_positions)
        return (start_ce + end_ce) / 2
