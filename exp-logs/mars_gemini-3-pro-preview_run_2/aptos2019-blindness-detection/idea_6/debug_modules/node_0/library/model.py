import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer with precision safety for Mixed Precision training.
    Computes: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to the config value
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.use_fp32 = Config.USE_FP32_GEM

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        if self.use_fp32:
            return self.gem_fp32(x)
        else:
            return self.gem(x)

    def gem(self, x):
        x = x.clamp(min=self.eps)
        x_pow = x.pow(self.p)
        # Average pooling over spatial dimensions (H, W)
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        return avg.pow(1.0 / self.p)

    def gem_fp32(self, x):
        # Force operations to Float32 to prevent NaN/Overflow in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x = x.clamp(min=self.eps)
            x_pow = x.pow(self.p)
            avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
            return avg.pow(1.0 / self.p)


class DRModel(nn.Module):
    """
    Diabetic Retinopathy Classification Model (Regression Formulation).
    Wraps a timm backbone (CNN or Transformer) with a GeM pooling layer and a regression head.
    """

    def __init__(self, model_name, pretrained=True, checkpoint_path=None):
        super(DRModel, self).__init__()

        # Determine image size from Config based on model name
        # This is helpful for Transformers that might need specific resolutions
        if model_name == Config.MODEL_CNN["name"]:
            img_size = Config.MODEL_CNN["img_size"]
        elif model_name == Config.MODEL_TRANS["name"]:
            img_size = Config.MODEL_TRANS["img_size"]
        else:
            img_size = 256  # Default fallback

        # Create backbone with no classifier and no global pooling
        # This returns the raw feature maps
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=img_size,
        )

        # Dynamic Feature Dimension & Layout Detection
        # We run a dummy forward pass to detect if output is (B, C, H, W) or (B, H, W, C)
        # and to find the number of channels (C).
        self.channels_last = False
        self.in_features = 0

        with torch.no_grad():
            dummy_input = torch.zeros(2, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy_input)

            if features.ndim == 4:
                # Check if channels are at dim 1 or dim 3
                # Safe bet is checking against timm's num_features if available
                if hasattr(self.backbone, "num_features"):
                    c_dim = self.backbone.num_features
                    if features.shape[1] == c_dim:
                        self.in_features = c_dim
                        self.channels_last = False  # (B, C, H, W)
                    elif features.shape[3] == c_dim:
                        self.in_features = c_dim
                        self.channels_last = True  # (B, H, W, C)
                    else:
                        # Fallback logic based on shape heuristic
                        if features.shape[1] < features.shape[3]:
                            self.in_features = features.shape[1]
                            self.channels_last = False
                        else:
                            self.in_features = features.shape[3]
                            self.channels_last = True
                else:
                    # Fallback if num_features not present, assume standard CNN
                    self.in_features = features.shape[1]
            elif features.ndim == 3:
                # (B, L, C) - Transformer sequence output
                self.in_features = features.shape[2]
                self.channels_last = True  # Treated as sequence
            else:
                # Fallback for unexpected shapes
                self.in_features = features.shape[1]

        # Pooling Layer
        self.pool = GeM(p=Config.GEM_P)

        # Regression Head (Linear)
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

        # Load weights from a local checkpoint if provided
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # 1. Extract Features
        x = self.backbone.forward_features(x)

        # 2. Standardize Layout to (B, C, H, W) for GeM
        if self.channels_last:
            if x.ndim == 4:
                # (B, H, W, C) -> (B, C, H, W)
                x = x.permute(0, 3, 1, 2)
            elif x.ndim == 3:
                # (B, L, C) -> Reshape to (B, C, H, W)
                # Assuming square feature map
                B, L, C = x.shape
                H = int(L**0.5)
                if H * H == L:
                    x = x.view(B, H, H, C).permute(0, 3, 1, 2)
                else:
                    # If not square, treat as 1D spatial (B, C, L, 1)
                    x = x.permute(0, 2, 1).unsqueeze(-1)

        # 3. Apply GeM Pooling
        # Output shape: (B, C, 1, 1)
        x = self.pool(x)

        # 4. Flatten
        x = x.flatten(1)

        # 5. Regression Head
        x = self.fc(x)

        return x
