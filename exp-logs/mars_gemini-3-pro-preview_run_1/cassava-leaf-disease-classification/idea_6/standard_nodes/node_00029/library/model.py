import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial features.
    Reference: https://arxiv.org/abs/1711.02512
    """

    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super(GeM, self).__init__()
        # p can be a fixed value or a learnable parameter
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # 1. Clamp to avoid numerical instability with negative values (though usually ReLU features are >= 0)
        # 2. Raise to power p
        # 3. Average pool over spatial dimensions (H, W)
        # 4. Raise to power 1/p
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

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


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Architecture:
        - Backbone: ConvNeXt-Small (pre-trained on ImageNet-21k)
        - Pooling: Generalized Mean Pooling (GeM)
        - Head: Multi-Sample Dropout (MSD) + Linear
    """

    def __init__(self, model_name=CFG.model_name, pretrained=True):
        super(CassavaModel, self).__init__()

        # 1. Load Backbone
        # num_classes=0 and global_pool='' ensures we get spatial features (B, C, H, W)
        # drop_path_rate controls Stochastic Depth
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=CFG.drop_path_rate,
        )

        # Get the number of input features for the classifier
        self.in_features = self.backbone.num_features

        # 2. Pooling Layer
        if CFG.use_gem:
            self.pooling = GeM(p=CFG.gem_p, learnable=CFG.gem_learnable)
        else:
            # Fallback to standard Global Average Pooling if GeM is disabled
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Classification Head (Multi-Sample Dropout)
        self.use_msd = CFG.use_msd
        if self.use_msd:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(CFG.msd_rate) for _ in range(CFG.msd_num)]
            )
        else:
            self.dropout = nn.Dropout(CFG.msd_rate)

        self.fc = nn.Linear(self.in_features, CFG.num_classes)

        # Initialize weights for the fully connected layer
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Extract features from backbone
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Apply Pooling
        # Shape: (B, C, 1, 1)
        features = self.pooling(features)

        # Flatten
        # Shape: (B, C)
        features = features.view(features.size(0), -1)

        # Apply Classification Head
        if self.use_msd:
            # Multi-Sample Dropout: Average the logits from multiple dropout masks
            logits = sum(
                [self.fc(dropout(features)) for dropout in self.dropouts]
            ) / len(self.dropouts)
        else:
            # Standard Dropout
            logits = self.fc(self.dropout(features))

        return logits
