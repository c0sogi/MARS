import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp min value for numerical stability before power operation
        x = x.clamp(min=eps).pow(p)
        # Average pooling over spatial dimensions
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Root p
        return x.pow(1.0 / p)

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


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier.
    Integrates a timm backbone, GeM pooling, and Multi-Sample Dropout head.
    """

    def __init__(self, model_name, pretrained=True, image_size=None):
        super(CassavaClassifier, self).__init__()
        self.model_name = model_name

        # Create Backbone
        # global_pool='' ensures we get the spatial feature map (B, C, H, W) or (B, H, W, C)
        # We explicitly pass img_size to ensure Swin Transformers initialize
        # their windowing and masks correctly for the target resolution.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.drop_path_rate,
            img_size=image_size,
        )

        # Disable strict image size check to allow Progressive Resizing (e.g., 384 -> 512)
        if hasattr(self.backbone, "patch_embed") and hasattr(
            self.backbone.patch_embed, "strict_img_size"
        ):
            self.backbone.patch_embed.strict_img_size = False

        # Determine input features and handle shape differences
        with torch.no_grad():
            # Dummy forward pass to inspect output shape
            # Use the specified image_size if provided, else default
            if image_size is not None:
                dummy_size = (3, image_size, image_size)
            else:
                dummy_size = self.backbone.default_cfg.get("input_size", (3, 224, 224))

            dummy = torch.randn(1, *dummy_size)
            features = self.backbone(dummy)

            # Identify channel dimension
            if features.ndim == 4:
                # Swin Transformers in timm often return (B, H, W, C)
                # ConvNets return (B, C, H, W)
                if "swin" in model_name:
                    self.in_features = features.shape[-1]
                else:
                    self.in_features = features.shape[1]
            else:
                # Fallback if model returns flattened features (shouldn't happen with global_pool='')
                self.in_features = features.shape[1]

        # Pooling Layer
        if Config.use_gem_pooling:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Multi-Sample Dropout Head
        self.use_msd = Config.use_multi_sample_dropout
        if self.use_msd:
            self.dropouts = nn.ModuleList(
                [
                    nn.Dropout(Config.msd_dropout_rate)
                    for _ in range(Config.msd_num_samples)
                ]
            )
        else:
            self.dropout = nn.Dropout(Config.msd_dropout_rate)

        self.fc = nn.Linear(self.in_features, Config.num_classes)

        # Initialize Head Weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.backbone(x)

        # Standardize feature map to (B, C, H, W) for GeM
        # If features are (B, H, W, C), permute to (B, C, H, W)
        if features.ndim == 4:
            if (
                features.shape[-1] == self.in_features
                and features.shape[1] != self.in_features
            ):
                features = features.permute(0, 3, 1, 2)

        # Pooling
        pooled = self.pooling(features)

        # Flatten
        pooled = pooled.flatten(1)

        # Classification Head
        if self.use_msd:
            # Multi-Sample Dropout: Average logits from multiple dropout masks
            logits = torch.mean(
                torch.stack(
                    [self.fc(dropout(pooled)) for dropout in self.dropouts], dim=0
                ),
                dim=0,
            )
        else:
            logits = self.fc(self.dropout(pooled))

        return logits


def get_optimizer_params(model, base_lr, weight_decay, decay_rate=0.8):
    """
    Constructs parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Strategy:
    - Head parameters (Classifier, Pooling): base_lr
    - Backbone parameters: Grouped by stage/depth, decaying by decay_rate.
      Stage 3 (deepest) -> base_lr * decay^1
      ...
      Stem (shallowest) -> base_lr * decay^5
    """

    # Initialize groups
    head_params = []
    backbone_groups = {
        "stem": [],
        "stage0": [],
        "stage1": [],
        "stage2": [],
        "stage3": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" not in name:
            # Parameters not in backbone (fc, pooling, etc.) belong to head
            head_params.append(param)
        else:
            # Heuristic to map parameter names to stages for ConvNeXt and Swin
            # ConvNeXt uses 'stages.X', Swin uses 'layers.X'
            if "stages.0" in name or "layers.0" in name:
                backbone_groups["stage0"].append(param)
            elif "stages.1" in name or "layers.1" in name:
                backbone_groups["stage1"].append(param)
            elif "stages.2" in name or "layers.2" in name:
                backbone_groups["stage2"].append(param)
            elif "stages.3" in name or "layers.3" in name:
                backbone_groups["stage3"].append(param)
            elif "stem" in name or "patch_embed" in name:
                backbone_groups["stem"].append(param)
            else:
                # Fallback: Parameters like final norms usually belong to the deepest stage
                backbone_groups["stage3"].append(param)

    # Create parameter list with specific LRs
    param_groups = []

    # Head
    param_groups.append(
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay}
    )

    # Backbone (Deepest to Shallowest)
    # Stage 3
    param_groups.append(
        {
            "params": backbone_groups["stage3"],
            "lr": base_lr * (decay_rate**1),
            "weight_decay": weight_decay,
        }
    )

    # Stage 2
    param_groups.append(
        {
            "params": backbone_groups["stage2"],
            "lr": base_lr * (decay_rate**2),
            "weight_decay": weight_decay,
        }
    )

    # Stage 1
    param_groups.append(
        {
            "params": backbone_groups["stage1"],
            "lr": base_lr * (decay_rate**3),
            "weight_decay": weight_decay,
        }
    )

    # Stage 0
    param_groups.append(
        {
            "params": backbone_groups["stage0"],
            "lr": base_lr * (decay_rate**4),
            "weight_decay": weight_decay,
        }
    )

    # Stem
    param_groups.append(
        {
            "params": backbone_groups["stem"],
            "lr": base_lr * (decay_rate**5),
            "weight_decay": weight_decay,
        }
    )

    return param_groups
