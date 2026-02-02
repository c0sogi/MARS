import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the feature map, focusing on high-activation regions
    (like soft MaxPool) which is beneficial for detecting localized disease artifacts.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min to eps to avoid NaN in pow gradient
        # Average pool over spatial dimensions (H, W)
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


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model.
    Uses a timm backbone with GeM pooling and Multi-Sample Dropout.
    Outputs 2 binary logits: [Rust, Scab].
    """

    def __init__(
        self,
        model_name,
        pretrained=True,
        num_classes=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_gem=True,
    ):
        """
        Args:
            model_name (str): Name of the timm model backbone.
            pretrained (bool): Whether to load pretrained weights.
            num_classes (int): Number of output classes (logits).
            drop_rate (float): Dropout rate for the classification head.
            drop_path_rate (float): Stochastic depth rate for the backbone.
            use_gem (bool): Whether to use GeM pooling instead of AdaptiveAvgPool.
        """
        super(AppleDiseaseModel, self).__init__()

        # Initialize backbone
        # num_classes=0 and global_pool="" returns the unpooled feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Determine number of input features for the head
        self.in_features = self.backbone.num_features

        # Pooling Layer
        if use_gem:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Multi-Sample Dropout
        # We use 5 dropout samples to improve generalization and convergence
        self.num_dropout_samples = 5
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(self.num_dropout_samples)]
        )

        # Classification Head
        self.fc = nn.Linear(self.in_features, num_classes)

        # Initialize head weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 1. Feature Extraction
        # Output shape: (Batch, Channels, Height, Width)
        x = self.backbone(x)

        # 2. Pooling
        # Output shape: (Batch, Channels, 1, 1)
        x = self.pooling(x)

        # 3. Flatten
        # Output shape: (Batch, Channels)
        x = x.flatten(1)

        # 4. Multi-Sample Dropout & Classification
        # Pass features through multiple dropout masks and the same linear layer
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(x)))

        # Average the logits
        # Shape: (Batch, Num_Classes)
        logits = torch.mean(torch.stack(logits_list), dim=0)

        return logits
