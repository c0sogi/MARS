import torch
import torch.nn as nn
from library.config import Config
from library.model import ArcMarginProduct


class ArcFaceLoss(nn.Module):
    """
    ArcFaceLoss module that encapsulates the ArcMarginProduct layer and CrossEntropyLoss.

    This module holds the learnable class centers (via ArcMarginProduct) and computes
    the categorical cross-entropy loss on the margin-modified logits.
    """

    def __init__(
        self,
        in_features=Config.embedding_size,
        out_features=Config.n_classes,
        s=Config.arcface_s,
        m=Config.arcface_m,
        easy_margin=False,
    ):
        """
        Args:
            in_features (int): Size of the input embedding vectors.
            out_features (int): Number of classes (identities).
            s (float): Norm of input feature (scale).
            m (float): Angular margin.
            easy_margin (bool): Whether to use easy margin.
        """
        super(ArcFaceLoss, self).__init__()
        # The ArcMarginProduct layer handles the learnable weights (centers)
        # and the calculation of logits with the additive angular margin.
        self.arc_product = ArcMarginProduct(
            in_features, out_features, s, m, easy_margin
        )

        # Standard Cross Entropy Loss to be applied on the margin logits
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (B, in_features).
            labels (torch.Tensor): Ground truth labels of shape (B).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute logits with angular margin
        # The ArcMarginProduct expects normalized embeddings if not done internally,
        # but based on the provided library/model.py, ArcMarginProduct handles normalization
        # of both weights and input (F.normalize(input)).
        logits = self.arc_product(embeddings, labels)

        # Compute Cross Entropy Loss
        loss = self.criterion(logits, labels)

        return loss
