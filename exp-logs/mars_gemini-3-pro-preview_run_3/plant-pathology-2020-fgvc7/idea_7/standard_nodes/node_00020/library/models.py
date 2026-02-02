import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a parameter p to interpolate between Average Pooling and Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Force float32 for numerical stability with AMP
        x = x.to(torch.float32)
        # Clamp min value to avoid NaN in power
        x = x.clamp(min=eps)
        # Average pooling on x^p
        # Output size becomes (B, C, 1, 1)
        x_p = x.pow(p)
        avg_pool = F.avg_pool2d(x_p, (x.size(-2), x.size(-1)))
        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AppleEfficientNet(nn.Module):
    """
    EfficientNet-B4 backbone with Multi-Level GeM Pooling.
    Extracts features from strides 8, 16, and 32.
    """

    def __init__(self, model_name=Config.EFFNET_MODEL_NAME, pretrained=True):
        super(AppleEfficientNet, self).__init__()

        # Load backbone with features_only=True
        # indices (2, 3, 4) correspond to strides 8, 16, 32 for EfficientNet
        self.backbone = timm.create_model(
            model_name, features_only=True, out_indices=(2, 3, 4), pretrained=pretrained
        )

        # Get channel counts for the selected indices
        feature_channels = self.backbone.feature_info.channels()

        # Create a GeM pooling layer for each feature level
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_channels))])

        # Calculate total input dimension for the classifier
        total_features = sum(feature_channels)

        # Final Classifier
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling
            pooled = self.gem_pools[i](feat)
            # Flatten: (B, C, 1, 1) -> (B, C)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate features from all levels
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output


class AppleSwin(nn.Module):
    """
    Swin Transformer Small backbone with Multi-Stage GeM Pooling.
    Extracts features from Stages 2, 3, and 4.
    """

    def __init__(self, model_name=Config.SWIN_MODEL_NAME, pretrained=True):
        super(AppleSwin, self).__init__()

        # Load backbone without features_only (Cite debug_lesson_2)
        # timm 1.0.20+ Swin implementation does not support features_only=True
        self.backbone = timm.create_model(
            model_name, features_only=False, pretrained=pretrained
        )

        # Hook storage
        self.features = {}
        self.hooks = []

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        # Register hooks for stages 2, 3, 4 (indices 1, 2, 3 in layers)
        target_layers = [1, 2, 3]
        for i in target_layers:
            self.hooks.append(
                self.backbone.layers[i].register_forward_hook(get_activation(i))
            )

        # Determine channel counts dynamically (Cite debug_lesson_6)
        # Perform dummy forward pass to get output shapes
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.SWIN_IMG_SIZE, Config.SWIN_IMG_SIZE)
            _ = self.backbone(dummy_input)

            feature_channels = []
            for i in target_layers:
                # Swin outputs are (B, H, W, C)
                # We need C for GeM/Linear setup
                shape = self.features[i].shape
                feature_channels.append(shape[-1])

        # Create GeM pooling layers
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_channels))])

        # Calculate total input dimension
        total_features = sum(feature_channels)

        # Final Classifier
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Clear previous features
        self.features = {}

        # Run backbone (hooks will populate self.features)
        _ = self.backbone(x)

        pooled_features = []
        # Process in order 1, 2, 3
        for i, layer_idx in enumerate([1, 2, 3]):
            feat = self.features[layer_idx]

            # Swin outputs flattened (B, L, C), reshape to (B, H, W, C)
            # Cite debug_lesson_8
            if feat.dim() == 3:
                B, L, C = feat.shape
                H = W = int(L**0.5)
                feat = feat.view(B, H, W, C)

            # Convert to (B, C, H, W) for GeM
            feat = feat.permute(0, 3, 1, 2)

            # Apply GeM pooling
            pooled = self.gem_pools[i](feat)
            # Flatten
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        output = self.fc(concat_features)
        return output
