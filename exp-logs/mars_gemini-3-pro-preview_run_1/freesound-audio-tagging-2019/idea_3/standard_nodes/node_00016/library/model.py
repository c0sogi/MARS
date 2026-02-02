import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from library.config import Config


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)

        # Calculate attention scores
        # scores shape: (batch_size, time_steps, 1)
        scores = self.linear(x)

        # Normalize scores to weights
        weights = torch.softmax(scores, dim=1)

        # Compute weighted sum of features
        # output shape: (batch_size, input_dim)
        output = torch.sum(x * weights, dim=1)

        return output


class AudioCRNN(nn.Module):
    def __init__(self):
        super(AudioCRNN, self).__init__()

        # 1. Input Adaptation
        # Learnable Batch Normalization to adapt Log-Mel stats to ImageNet stats
        self.bn0 = nn.BatchNorm2d(1)

        # 2. Backbone (EfficientNet B2)
        # Use ImageNet weights if specified in Config
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        self.backbone = efficientnet_b2(weights=weights)

        # Modify first conv layer to accept 1 channel (Cite solution_lesson_node_00015)
        # efficientnet_b2 features[0] is Conv2dNormActivation, features[0][0] is Conv2d
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        # Initialize with sum of RGB weights
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(first_conv.weight, dim=1, keepdim=True)

        self.backbone.features[0][0] = new_conv

        # EfficientNet-B2 output channels at the final feature map is 1408
        self.backbone_channels = 1408

        # 3. Aggregation (Attention Pooling)
        # Direct connection from backbone to pooling (Cite solution_lesson_node_00010)
        self.attention_pooling = AttentionPooling(self.backbone_channels)

        # 4. Classifier
        self.classifier = nn.Linear(self.backbone_channels, Config.NUM_CLASSES)

    def forward(self, x):
        # Input x shape: (batch, 1, n_mels, time)

        # --- Input Adaptation ---
        x = self.bn0(x)

        # --- Backbone Feature Extraction ---
        # Pass through EfficientNet features section
        # Output shape: (batch, 1408, H_freq, W_time)
        x = self.backbone.features(x)

        # --- Frequency Pooling ---
        # Average pool over the frequency dimension (dim 2)
        # Shape becomes: (batch, 1408, W_time)
        x = torch.mean(x, dim=2)

        # --- Prepare for Attention Pooling ---
        # Permute to (batch, time, features)
        # Shape becomes: (batch, W_time, 1408)
        x = x.permute(0, 2, 1)

        # --- Aggregation ---
        # Apply Attention Pooling to get clip-level embedding
        # Shape becomes: (batch, backbone_channels)
        x = self.attention_pooling(x)

        # --- Classification ---
        # Shape becomes: (batch, num_classes)
        logits = self.classifier(x)

        return logits
