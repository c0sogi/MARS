import torch
import torch.nn as nn
from library.config import config


class MultiTaskLoss(nn.Module):
    """
    Computes the weighted sum of losses for the Kernel-Pooling Network tasks:
    1. Long Answer Ranking (Binary Classification)
    2. Short Answer Span Prediction (Start/End Index Classification)
    3. Yes/No Answer Classification (Multi-class Classification)
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()

        # Weights for each task
        self.w_ranking = config.LOSS_WEIGHT_RANKING
        self.w_span = config.LOSS_WEIGHT_SPAN
        self.w_yesno = config.LOSS_WEIGHT_YESNO

        # Loss functions
        # Ranking: Binary classification (Is this candidate the answer or not?)
        self.ranking_loss_fn = nn.BCEWithLogitsLoss()

        # Span: Multi-class classification over sequence length (Which token is start/end?)
        # We assume targets are class indices (0 to max_len-1)
        self.span_loss_fn = nn.CrossEntropyLoss()

        # Yes/No: Multi-class classification (NONE, YES, NO)
        self.yesno_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Compute the total loss.

        Args:
            outputs (dict): Dictionary containing model predictions:
                - 'long_score': Tensor [Batch, 1]
                - 'start_logits': Tensor [Batch, C_Len]
                - 'end_logits': Tensor [Batch, C_Len]
                - 'yesno_logits': Tensor [Batch, Num_Classes]
            targets (dict): Dictionary containing ground truth labels:
                - 'label_long': Tensor [Batch] (Float, 0.0 or 1.0)
                - 'label_span_start': Tensor [Batch] (Long, indices)
                - 'label_span_end': Tensor [Batch] (Long, indices)
                - 'label_yesno': Tensor [Batch] (Long, class indices)

        Returns:
            loss (Tensor): Scalar total loss.
            metrics (dict): Dictionary of individual loss components (detached).
        """
        # 1. Ranking Loss
        # Squeeze the model output to match target shape [Batch] if necessary,
        # or unsqueeze target to [Batch, 1]. BCEWithLogitsLoss expects float targets.
        long_scores = outputs["long_score"].squeeze(-1)  # [Batch]
        long_labels = targets["label_long"]  # [Batch]
        loss_ranking = self.ranking_loss_fn(long_scores, long_labels)

        # 2. Span Prediction Loss
        # CrossEntropyLoss expects logits [Batch, C] and targets [Batch]
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        start_labels = targets["label_span_start"]
        end_labels = targets["label_span_end"]

        loss_start = self.span_loss_fn(start_logits, start_labels)
        loss_end = self.span_loss_fn(end_logits, end_labels)
        loss_span = (loss_start + loss_end) / 2.0

        # 3. Yes/No Classification Loss
        yesno_logits = outputs["yesno_logits"]
        yesno_labels = targets["label_yesno"]
        loss_yesno = self.yesno_loss_fn(yesno_logits, yesno_labels)

        # 4. Weighted Sum
        total_loss = (
            (self.w_ranking * loss_ranking)
            + (self.w_span * loss_span)
            + (self.w_yesno * loss_yesno)
        )

        # Metrics for logging
        metrics = {
            "loss_total": total_loss.item(),
            "loss_ranking": loss_ranking.item(),
            "loss_span": loss_span.item(),
            "loss_yesno": loss_yesno.item(),
        }

        return total_loss, metrics
