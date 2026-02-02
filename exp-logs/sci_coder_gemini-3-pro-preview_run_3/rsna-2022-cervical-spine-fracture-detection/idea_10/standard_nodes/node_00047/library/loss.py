import torch
import torch.nn as nn


class HierarchicalCompoundLoss(nn.Module):
    """
    Decoupled Multi-Task Loss.
    Computes loss for 7 vertebral classes and 1 patient class separately.
    Cite solution_lesson_node_00045: Prefer Multi-Task Learning over Logical Aggregation.
    Cite solution_lesson_node_00008: Implicit Weighting via Loss Reduction.
    """

    def __init__(self):
        super(HierarchicalCompoundLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets, patient_target):
        # logits: (B, 8)
        # targets: (B, 7)
        # patient_target: (B,)

        # Split logits
        logits_c = logits[:, :7]
        logits_p = logits[:, 7]

        # 1. Vertebral Loss (Mean over 7 classes)
        loss_c = self.bce(logits_c, targets)

        # 2. Patient Loss (Scalar)
        loss_p = self.bce(logits_p, patient_target)

        # Total Loss = Mean(C1-C7) + Patient
        return loss_c + loss_p
