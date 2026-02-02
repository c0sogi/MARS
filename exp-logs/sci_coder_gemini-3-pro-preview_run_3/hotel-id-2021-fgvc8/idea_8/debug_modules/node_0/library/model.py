import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp for numerical stability
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


class SubCenterArcFaceHead(nn.Module):
    """
    ArcFace Head with Sub-centers for handling high intra-class variance.
    Reference: 'Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces'
    """

    def __init__(self, in_features, out_features, k=3, s=30.0, m=0.5):
        super(SubCenterArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m

        # Shape: (num_classes, k_subcenters, embedding_size)
        # We use a single tensor and reshape logic for efficiency
        self.weight = nn.Parameter(torch.FloatTensor(out_features, k, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        # embeddings: (B, in_features)
        # labels: (B,) or None

        # 1. Normalize Input
        norm_embeddings = F.normalize(embeddings, dim=1)

        # 2. Normalize Weights
        # weight shape: (out, k, in) -> normalize along embedding dim (2)
        norm_weights = F.normalize(self.weight, dim=2)

        # 3. Flatten weights for efficient matrix multiplication
        # (out * k, in)
        flat_weights = norm_weights.view(-1, self.in_features)

        # 4. Compute Cosine Similarity
        # (B, out * k)
        cosine_all = F.linear(norm_embeddings, flat_weights)

        # 5. Reshape and Max Pooling over Sub-centers
        # (B, out, k)
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        # (B, out) - Select the best matching sub-center for each class
        cosine, _ = torch.max(cosine_all, dim=2)

        if labels is None:
            # Inference: return scaled logits
            return cosine * self.s

        # 6. ArcFace Margin Logic (Training)
        # We only apply margin to the ground truth class

        # Create a one-hot mask for ground truth
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Calculate margin term: cos(theta + m)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)

        # Clamp for numerical stability in acos/sqrt
        cosine_clamped = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))

        phi = cosine_clamped * self.cos_m - sine * self.sin_m

        # Handle cases where theta + m > pi to ensure monotonicity
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to ground truth indices
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale by s
        output *= self.s

        return output


class HotelModel(nn.Module):
    """
    Main model class for Hotel Identification.
    Backbone: EfficientNet-V2-S
    Pooling: GeM
    Neck: Linear + BN
    Head: Sub-center ArcFace
    """

    def __init__(
        self,
        num_classes,
        model_name="tf_efficientnetv2_s",
        embedding_size=512,
        scale=30.0,
        margin=0.5,
        k_subcenters=3,
        pretrained=True,
    ):
        super(HotelModel, self).__init__()

        # 1. Backbone
        # Create model with no classifier (num_classes=0)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features

        # 2. Pooling
        self.pooling = GeM()

        # 3. Neck (Embedding Projection)
        # Projects backbone features to the desired embedding dimension
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        # 4. Head (Metric Learning)
        self.head = SubCenterArcFaceHead(
            in_features=embedding_size,
            out_features=num_classes,
            k=k_subcenters,
            s=scale,
            m=margin,
        )

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            labels (torch.Tensor, optional): Ground truth labels (B,).

        Returns:
            If labels is None: Returns embeddings (B, embedding_size)
            If labels is not None: Returns logits (B, num_classes)
        """
        # Feature extraction
        x = self.backbone.forward_features(x)  # (B, C, H, W)
        x = self.pooling(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)

        # Embedding projection
        embeddings = self.neck(x)  # (B, embedding_size)

        if labels is not None:
            # Training: Return logits with margin penalty
            return self.head(embeddings, labels)
        else:
            # Inference/Validation: Return embeddings for retrieval
            return embeddings
