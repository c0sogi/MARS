import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the p-th power mean of the feature map.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN with negative values (e.g., from SiLU/Swish activations)
        # and to ensure numerical stability for the power operation.
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


class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace Head.
    Handles high intra-class variance by allowing multiple centers (sub-centers) per class.
    """

    def __init__(self, in_features, out_features, k=3, s=30.0, m=0.50):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m

        # Weights shape: (Class_Centers * K, Embedding_Dim)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for ArcFace margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        # features: (B, D)
        # labels: (B,)

        # Normalize weights and features
        weights = F.normalize(self.weight)
        features = F.normalize(features)

        # Compute cosine similarity: (B, C*K)
        cosine = F.linear(features, weights)

        # Reshape to (B, C, K) to isolate sub-centers per class
        cosine = cosine.view(-1, self.out_features, self.k)

        # Take the maximum similarity across sub-centers -> (B, C)
        cosine, _ = torch.max(cosine, dim=2)

        if labels is None:
            # During inference (if used), return scaled cosine similarities
            return cosine * self.s

        # --- ArcFace Margin Application ---

        # Gather the cosine similarity for the ground truth class
        target_cosine = cosine.gather(1, labels.view(-1, 1))  # (B, 1)

        # Calculate cos(theta + m)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_cosine, 2))
        phi = target_cosine * self.cos_m - sin_theta * self.sin_m

        # Stability check: ensure we don't violate monotonicity
        phi = torch.where(target_cosine > self.th, phi, target_cosine - self.mm)

        # Create one-hot encoding for targets
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        return output


class HotelModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        n_classes,
        embedding_dim=512,
        pretrained=True,
        use_gem_pooling=True,
        use_bn_neck=True,
        arcface_scale=30.0,
        arcface_margin=0.50,
        sub_centers_k=3,
    ):
        """
        Hotel ID Recognition Model.

        Args:
            backbone_name (str): Name of the timm backbone.
            n_classes (int): Number of target classes (hotels).
            embedding_dim (int): Dimension of the embedding space.
            pretrained (bool): Whether to load pretrained weights.
            use_gem_pooling (bool): Whether to use GeM pooling.
            use_bn_neck (bool): Whether to use Batch Norm neck.
            arcface_scale (float): ArcFace scale factor (s).
            arcface_margin (float): ArcFace margin (m).
            sub_centers_k (int): Number of sub-centers per class.
        """
        super(HotelModel, self).__init__()

        # Initialize Backbone
        # num_classes=0 removes the classification head
        # global_pool='' removes the default pooling, keeping spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of output features from the backbone
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        # Pooling Layer
        if use_gem_pooling:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()

        # Projection Layer
        self.linear = nn.Linear(in_features, embedding_dim)

        # BN Neck
        self.use_bn_neck = use_bn_neck
        if use_bn_neck:
            self.bn = nn.BatchNorm1d(embedding_dim)
            # Initialize BN to pass through gradients cleanly
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)

        # Classification Head
        self.head = SubCenterArcFace(
            in_features=embedding_dim,
            out_features=n_classes,
            k=sub_centers_k,
            s=arcface_scale,
            m=arcface_margin,
        )

    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images (B, C, H, W).
            labels (torch.Tensor, optional): Ground truth labels (B,).

        Returns:
            If labels is None (Inference):
                torch.Tensor: Normalized embeddings (B, embedding_dim).
            If labels is not None (Training):
                torch.Tensor: Logits with ArcFace margin applied (B, n_classes).
        """
        # Extract features from backbone
        x = self.backbone(x)

        # Pool and Flatten
        x = self.pool(x)
        x = self.flatten(x)

        # Project to embedding space
        x = self.linear(x)

        # Apply BN Neck
        if self.use_bn_neck:
            x = self.bn(x)

        # Inference: Return normalized embeddings
        if labels is None:
            return F.normalize(x)

        # Training: Return logits from ArcFace head
        return self.head(x, labels)
