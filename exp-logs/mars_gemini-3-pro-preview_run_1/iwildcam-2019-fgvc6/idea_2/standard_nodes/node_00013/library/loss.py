import torch
import torch.nn as nn
from library.utils import FocalLoss


class ClassBalancedFocalLoss(FocalLoss):
    """
    Implements the Class Balanced Focal Loss for addressing class imbalance.

    This module computes the focal loss formulated as:
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    It inherits the implementation from library.utils.FocalLoss, which ensures that
    alpha (class weights) is applied as an external multiplier to the loss term
    rather than inside the cross-entropy function. This guarantees proper gradient
    scaling for all classes, including those with high or low frequencies.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (torch.Tensor, optional): Pre-computed class weights of shape (num_classes,).
                                          These weights are used to down-weight dominant classes
                                          and up-weight rare classes.
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                           Default is 'mean'.
        """
        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction)
