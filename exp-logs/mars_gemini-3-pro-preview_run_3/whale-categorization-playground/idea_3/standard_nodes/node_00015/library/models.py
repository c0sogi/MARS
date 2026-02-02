import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class CurricularFaceHead(nn.Module):
    """
    CurricularFace Head layer.
    Computes the cosine similarity between L2-normalized embeddings and L2-normalized class centers.
    This layer does not compute the loss itself but provides the cosine logits required by CurricularFaceLoss.
    """

    def __init__(self, in_features, out_features):
        super(CurricularFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable class centers (weights)
        # Shape: (num_classes, embedding_size)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, embedding_size).

        Returns:
            torch.Tensor: Cosine similarity logits of shape (batch_size, num_classes).
        """
        # Normalize input embeddings (L2 norm)
        x = F.normalize(x, p=2, dim=1)
        # Normalize class centers (L2 norm)
        w = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: x . w^T
        logits = F.linear(x, w)
        return logits


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B2 backbone with a CurricularFace head for Whale Species Identification.
    """

    def __init__(self, num_classes):
        super(WhaleEfficientNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # num_classes=0 and global_pool='avg' ensures we get the pooled feature vector
        # without the default classifier.
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=CFG.pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the output feature dimension of the backbone (e.g., 1408 for B2)
        self.in_features = self.backbone.num_features

        # 2. Neck: Dropout -> Linear -> BatchNorm
        # Projects backbone features to the specified embedding size.
        self.dropout = nn.Dropout(p=CFG.drop_rate)
        self.fc = nn.Linear(self.in_features, CFG.embedding_size)
        self.bn = nn.BatchNorm1d(CFG.embedding_size)

        # 3. Head: CurricularFace
        # Computes cosine logits for metric learning.
        self.head = CurricularFaceHead(CFG.embedding_size, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor:
                - If training: Cosine logits (batch_size, num_classes) for loss calculation.
                - If eval: Embeddings (batch_size, embedding_size) for inference/ranking.
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Apply Neck
        features = self.dropout(features)
        embeddings = self.fc(features)
        embeddings = self.bn(embeddings)

        # Conditional return based on mode
        if self.training:
            # During training, pass through the head to get cosine logits
            logits = self.head(embeddings)
            return logits
        else:
            # During inference/validation, return embeddings for re-ranking
            return embeddings
