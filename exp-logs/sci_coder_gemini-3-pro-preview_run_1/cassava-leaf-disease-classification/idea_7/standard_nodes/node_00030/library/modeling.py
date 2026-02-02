import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Learns a parameter 'p' to transition between Average Pooling (p=1) and Max Pooling (p=infinity).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        """
        Applies GeM pooling.
        Handles both 4D (N, C, H, W) tensors from CNNs and 3D (N, L, C) tensors from Transformers.
        """
        # If input is (N, L, C), permute to (N, C, L) to treat L as spatial dimension
        if x.ndim == 3:
            x = x.permute(0, 2, 1)

        # Clamp for numerical stability
        x = x.clamp(min=eps).pow(p)

        # Apply average pooling
        if x.ndim == 4:
            # Global pooling over (H, W)
            x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        elif x.ndim == 3:
            # Global pooling over L
            x = F.avg_pool1d(x, x.size(-1))

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


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.
    Applies multiple dropout masks to the features and averages the predictions.
    Accelerates convergence and improves generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (B, in_features)
        # Compute logits for each dropout mask using the same FC layer
        logits = [self.fc(dropout(x)) for dropout in self.dropouts]
        # Stack and average
        return torch.mean(torch.stack(logits), dim=0)


class CassavaClassifier(nn.Module):
    """
    Main Classifier Class.
    Wraps a timm backbone, GeM pooling, and Multi-Sample Dropout head.
    Automatically handles feature dimension detection for different architectures.
    """

    def __init__(self, model_arch, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(CassavaClassifier, self).__init__()
        self.model_arch = model_arch

        # Load backbone with no head and no global pooling
        # This returns raw features: (B, C, H, W) for CNNs or (B, L, C) for Transformers
        self.backbone = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features for the head by running a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            features = self.backbone.forward_features(dummy)

            # Identify channel dimension
            if features.ndim == 3:
                # (N, L, C) -> C is last
                in_features = features.shape[-1]
            else:
                # (N, C, H, W) -> C is second
                in_features = features.shape[1]

        # Initialize GeM Pooling
        self.gem = GeM(p=3.0)

        # Initialize Multi-Sample Dropout Head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            num_samples=5,
            dropout_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        # Extract features
        x = self.backbone.forward_features(x)

        # Apply GeM Pooling
        # Output shape: (B, C, 1, 1) or (B, C, 1)
        x = self.gem(x)

        # Flatten to (B, C)
        x = x.flatten(1)

        # Classification
        x = self.head(x)

        return x


def get_llrd_params(
    model,
    lr=Config.LR_MAX,
    weight_decay=Config.WEIGHT_DECAY,
    decay_factor=Config.LLRD_DECAY,
):
    """
    Groups model parameters for Layer-wise Learning Rate Decay (LLRD).
    Assigns lower learning rates to earlier layers and higher rates to the head.

    Logic assumes a 4-stage hierarchy common in ConvNeXt and Swin Transformers.
    """

    # Helper to determine layer ID based on parameter name
    # Returns: 0 (stem/embed), 1-4 (stages), 5 (head/gem)
    def get_layer_id(name):
        if "head" in name or "fc" in name or "gem" in name:
            return 5
        elif "stages.3" in name or "layers.3" in name:
            return 4
        elif "stages.2" in name or "layers.2" in name:
            return 3
        elif "stages.1" in name or "layers.1" in name:
            return 2
        elif "stages.0" in name or "layers.0" in name:
            return 1
        else:
            # Stem, patch_embed, norms, etc.
            return 0

    # Dictionary to hold lists of params for each layer ID
    layer_params = {i: [] for i in range(6)}

    # Iterate named parameters and assign to groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        layer_params[layer_id].append(param)

    parameter_groups = []

    # Create optimizer param groups with calculated LRs
    for layer_id in range(6):
        params = layer_params[layer_id]
        if not params:
            continue

        # Calculate LR for this layer
        # Head (ID 5) -> decay^0 = 1.0 * lr
        # Stem (ID 0) -> decay^5 * lr
        exponent = 5 - layer_id
        layer_lr = lr * (decay_factor**exponent)

        parameter_groups.append(
            {"params": params, "lr": layer_lr, "weight_decay": weight_decay}
        )

    return parameter_groups
