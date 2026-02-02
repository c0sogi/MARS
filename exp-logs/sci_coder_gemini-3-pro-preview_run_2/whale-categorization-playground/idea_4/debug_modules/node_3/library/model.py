import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a pooling parameter 'p' to interpolate between Max and Average pooling.
    Reference: Radenovic et al., "Fine-tuning CNN Image Retrieval with No Human Annotation"
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Output: (B, C, 1, 1) -> Flattened later
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace Head (Class Centers).
    Computes cosine similarity between normalized embeddings and normalized class centers.
    This layer stores the learnable weights (centers) for each class.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Xavier Uniform
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        """
        Args:
            features: (B, Emb) input embeddings
        Returns:
            cosine: (B, NumClasses) cosine similarity matrix
        """
        # Normalize features and weights to lie on the hypersphere
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class WhaleModel(nn.Module):
    """
    Whale Species Prediction Model.

    Architecture:
    1. Backbone: EfficientNet-B4 (pretrained on ImageNet)
    2. Pooling: GeM (Generalized Mean Pooling)
    3. Neck: Flatten -> Dropout -> Linear -> BatchNorm
    4. Head: ArcMarginProduct (Training only)
    """

    def __init__(
        self, num_classes, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
    ):
        super().__init__()

        # Load backbone features only (no classifier, no global pool)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        in_features = self.backbone.num_features

        # Pooling and Neck
        self.pooling = GeM()
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fc = nn.Linear(in_features, Config.EMBEDDING_SIZE)
        self.bn = nn.BatchNorm1d(Config.EMBEDDING_SIZE)

        # Metric Learning Head
        self.arc_head = ArcMarginProduct(Config.EMBEDDING_SIZE, num_classes)

        # Initialize Batch Norm
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, images, labels=None):
        """
        Forward pass of the model.

        Args:
            images: (B, 3, H, W) Tensor of input images.
            labels: (B,) Tensor of ground truth labels (optional).

        Returns:
            If labels is provided (Training):
                logits: (B, NumClasses) Cosine similarity scores for ArcFaceLoss.
            If labels is None (Inference):
                embeddings: (B, EmbeddingSize) Feature embeddings.
        """
        # Feature Extraction
        features = self.backbone(images)

        # Pooling (B, C, H, W) -> (B, C, 1, 1)
        features = self.pooling(features)

        # Flatten (B, C, 1, 1) -> (B, C)
        features = features.flatten(1)

        # Dropout
        features = self.dropout(features)

        # Projection to Embedding Space
        embeddings = self.fc(features)
        embeddings = self.bn(embeddings)

        if labels is not None:
            # Training Mode: Pass through ArcFace head to get cosine logits
            return self.arc_head(embeddings)
        else:
            # Inference Mode: Return embeddings for retrieval
            return embeddings
