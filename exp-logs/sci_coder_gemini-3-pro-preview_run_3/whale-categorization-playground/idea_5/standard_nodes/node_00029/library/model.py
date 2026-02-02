import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config


class ArcFace(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    (Cite solution_lesson_node_00008)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Kernel (Weights) for the classes
        self.kernel = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.kernel)

        # Pre-compute constant for margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, label):
        # 1. Normalize features and weights
        kernel_norm = F.normalize(self.kernel, p=2, dim=1)
        emb_norm = F.normalize(embeddings, p=2, dim=1)

        # 2. Compute Cosine Similarity
        cosine = F.linear(emb_norm, kernel_norm)

        # 3. Calculate Target Logit (Angular Margin)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # 4. Create One-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 5. Apply Margin to Target Class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale
        output *= self.s

        return output


class WhaleModel(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout=Config.DROPOUT,
    ):
        super(WhaleModel, self).__init__()

        # 1. Backbone
        # Create EfficientNet-B2
        # num_classes=0 removes the top classifier
        # global_pool='' removes the default pooling, we will add our own
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension (EfficientNet-B2 usually 1408)
        # We can run a dummy pass or check classifier.in_features if it existed
        # For B2, num_features is 1408.
        self.in_features = self.backbone.num_features

        # 2. Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Neck (Embedding Layer)
        # BN -> Dropout -> FC -> BN
        self.neck = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            # PReLU is often helpful in metric learning to preserve negative information
            nn.PReLU(),
        )

        # 4. Head (Metric Learning Loss)
        self.head = ArcFace(
            in_features=embedding_dim,
            out_features=num_classes,
            s=Config.CF_S,
            m=Config.CF_M,
        )

    def forward(self, x, label=None):
        """
        Args:
            x: Input images (B, C, H, W)
            label: Target labels (B). If None, returns embeddings.
        """
        # Feature Extraction
        # (B, C, H, W) -> (B, F, H', W')
        features = self.backbone(x)

        # Pooling
        # (B, F, 1, 1)
        features = self.pooling(features)

        # Flatten
        # (B, F)
        features = features.view(features.size(0), -1)

        # Embedding Projection
        # (B, Emb_Dim)
        embeddings = self.neck(features)

        if label is not None:
            # Training: Return Logits (Scaled and Margined)
            return self.head(embeddings, label)
        else:
            # Inference: Return Normalized Embeddings
            return F.normalize(embeddings, p=2, dim=1)

    def extract_features(self, x):
        """Helper to get embeddings directly."""
        return self.forward(x, label=None)
