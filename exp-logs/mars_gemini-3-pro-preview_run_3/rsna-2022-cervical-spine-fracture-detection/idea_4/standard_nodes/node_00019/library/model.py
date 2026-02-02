import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning.
    Based on 'Attention-based Deep Multiple Instance Learning' (Ilse et al., 2018).
    """

    def __init__(self, input_dim, attention_dim=128):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Slices, Input_Dim)

        Returns:
            torch.Tensor: Aggregated feature representation of shape (Batch, Input_Dim)
            torch.Tensor: Attention weights of shape (Batch, Slices, 1)
        """
        # x: (B, S, D)

        # Compute attention scores
        # V: (B, S, attention_dim)
        v_out = self.attention_V(x)
        # U: (B, S, attention_dim)
        u_out = self.attention_U(x)

        # Element-wise multiplication (Gating)
        # w: (B, S, 1)
        a_val = self.attention_w(v_out * u_out)

        # Softmax over the slice dimension (dim=1)
        a_weights = F.softmax(a_val, dim=1)

        # Weighted sum of input features
        # (B, S, 1) * (B, S, D) -> (B, S, D) -> sum -> (B, D)
        z = torch.sum(x * a_weights, dim=1)

        return z, a_weights


class FractureMILModel(nn.Module):
    """
    Anatomically-Guided Attention-MIL Network.
    Uses ResNet18 backbone, Positional Injection, and Gated Attention.
    """

    def __init__(self):
        super(FractureMILModel, self).__init__()

        # 1. Backbone: ResNet18
        # We use the pretrained weights if available, or default initialization
        # Note: In this environment, we might not have internet, so we rely on cache or random init.
        # Ideally weights=models.ResNet18_Weights.DEFAULT, but for robustness we use standard call.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final FC layer.
        # ResNet18 structure: ... -> avgpool -> fc
        # We keep everything up to avgpool.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension from ResNet18 is 512
        self.feature_dim = 512

        # 2. Positional Injection
        # We append 1 dimension (normalized depth) to the features
        self.mil_input_dim = self.feature_dim + 1

        # 3. Aggregation: Gated Attention
        self.attention = GatedAttention(input_dim=self.mil_input_dim, attention_dim=128)

        # 4. Classifier Head
        # Maps aggregated features to 7 classes (C1-C7)
        self.classifier = nn.Linear(self.mil_input_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume of shape (Batch, Slices, Channels, Height, Width)
                              Channels should be 3 (2.5D stacking).

        Returns:
            torch.Tensor: Logits for C1-C7 of shape (Batch, 7)
        """
        batch_size, num_slices, channels, height, width = x.shape

        # --- 1. Feature Extraction ---
        # Reshape to process all slices in parallel: (B*S, C, H, W)
        x_flat = x.view(batch_size * num_slices, channels, height, width)

        # Pass through backbone
        features = self.backbone(x_flat)  # (B*S, 512, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B*S, 512)

        # --- 2. Positional Injection ---
        # Generate normalized depth coordinates [0, 1]
        # We assume slices are ordered.
        device = x.device

        # Create a linspace from 0 to 1 for the number of slices
        # Shape: (S,)
        pos_encoding = torch.linspace(0, 1, steps=num_slices, device=device)

        # Reshape to (1, S, 1) and repeat for batch
        # Shape: (B, S, 1)
        pos_encoding = pos_encoding.view(1, num_slices, 1).expand(batch_size, -1, -1)

        # Flatten to match features: (B*S, 1)
        pos_encoding_flat = pos_encoding.reshape(batch_size * num_slices, 1)

        # Concatenate features and positional encoding
        # (B*S, 512) cat (B*S, 1) -> (B*S, 513)
        features_aug = torch.cat([features, pos_encoding_flat], dim=1)

        # --- 3. Sequence Aggregation ---
        # Reshape back to sequence format: (B, S, 513)
        features_seq = features_aug.view(batch_size, num_slices, -1)

        # Apply Gated Attention
        # z: (B, 513), attn_weights: (B, S, 1)
        z, attn_weights = self.attention(features_seq)

        # --- 4. Classification ---
        # Predict logits for C1-C7
        logits = self.classifier(z)  # (B, 7)

        # Note: patient_overall is derived from these logits during loss calculation or inference
        # as max(sigmoid(logits)). The model returns the raw logits for the vertebrae.

        return logits
