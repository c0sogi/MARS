import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN gradients for small values and ensure stability
        x = x.clamp(min=eps)

        # Apply Average Pooling on x^p
        # Kernel size matches the spatial dimensions (H, W) to perform global pooling
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        x = x.pow(1.0 / p)
        return x


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout module.
    Applies multiple dropout masks to the input and averages the predictions
    from the linear layer to reduce variance and improve generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, p=0.5):
        super(MultiSampleDropout, self).__init__()
        # Create multiple dropout instances
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        # A single linear layer is shared across all dropout paths
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (Batch, In_Features)
        outputs = []
        for dropout in self.dropouts:
            # Apply dropout mask then linear projection
            outputs.append(self.linear(dropout(x)))

        # Stack outputs: (Num_Samples, Batch, Out_Features)
        # Average over the sample dimension (dim=0) to get the ensemble prediction
        return torch.mean(torch.stack(outputs), dim=0)


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task EfficientNet-B3 with GeM Pooling and Multi-Sample Dropout.

    Architecture:
    1. Backbone: EfficientNet-B3 (unfrozen)
    2. Pooling: GeM Pooling
    3. Fusion: Concatenation of Visual + Metadata -> Dense Block
    4. Aux Head: Predicts Diagnosis from Fused Features
    5. Primary Head: Predicts Malignancy from Fused Features + Aux Logits
    """

    def __init__(
        self,
        model_name="efficientnet_b3",
        pretrained=True,
        n_meta_features=0,
        n_diagnosis_classes=0,
        num_classes=1,
    ):
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Visual Backbone
        # num_classes=0 removes the default classification head
        # global_pool='' removes the default pooling, returning spatial feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Retrieve the number of channels in the last feature map (e.g., 1536 for B3)
        self.num_features = self.backbone.num_features

        # 2. GeM Pooling
        self.gem = GeM(p=3)

        # 3. Fusion Module
        # Input: Visual Features + Metadata Features
        fusion_in_features = self.num_features + n_meta_features
        fusion_out_features = 512

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_features, fusion_out_features),
            nn.BatchNorm1d(fusion_out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 4. Auxiliary Head (Diagnosis)
        # Uses Multi-Sample Dropout for regularization
        self.aux_head = MultiSampleDropout(
            in_features=fusion_out_features,
            out_features=n_diagnosis_classes,
            num_samples=5,
            p=0.5,
        )

        # 5. Primary Head (Target: Malignancy)
        # Input: Fused Features + Auxiliary Logits (Hierarchical Injection)
        primary_in_features = fusion_out_features + n_diagnosis_classes
        self.primary_head = MultiSampleDropout(
            in_features=primary_in_features,
            out_features=num_classes,
            num_samples=5,
            p=0.5,
        )

    def forward(self, image, meta=None):
        """
        Args:
            image: (B, 3, H, W) Input images
            meta: (B, n_meta_features) Input metadata features
        Returns:
            primary_logits: (B, 1) Logits for malignancy
            aux_logits: (B, n_diagnosis_classes) Logits for diagnosis
        """
        # 1. Feature Extraction
        # Output: (B, C, H', W')
        features = self.backbone(image)

        # 2. GeM Pooling
        # Output: (B, C, 1, 1)
        features = self.gem(features)

        # Flatten: (B, C)
        features = features.view(features.size(0), -1)

        # 3. Metadata Fusion
        if meta is not None:
            # Concatenate visual and metadata features
            features = torch.cat([features, meta], dim=1)

        # Pass through fusion block
        fused_features = self.fusion(features)

        # 4. Auxiliary Task
        # Predict diagnosis
        aux_logits = self.aux_head(fused_features)

        # 5. Primary Task
        # Hierarchical injection: Concatenate fused features with auxiliary predictions
        primary_input = torch.cat([fused_features, aux_logits], dim=1)
        primary_logits = self.primary_head(primary_input)

        return primary_logits, aux_logits
