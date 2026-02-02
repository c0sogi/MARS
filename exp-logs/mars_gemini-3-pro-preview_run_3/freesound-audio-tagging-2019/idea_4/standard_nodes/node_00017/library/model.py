import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to weigh time frames based on their relevance.
    Structure: Linear -> Tanh -> Linear -> Softmax
    """

    def __init__(self, input_dim, hidden_dim=None):
        super(AttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Channels)
        Returns:
            out: Weighted sum tensor of shape (Batch, Channels)
        """
        # Calculate attention weights: (Batch, Time, 1)
        weights = self.attention_net(x)

        # Apply weights to input and sum over time: (Batch, Channels)
        # Broadcasting weights across channels
        out = torch.sum(x * weights, dim=1)
        return out


class HybridEfficientNet(nn.Module):
    """
    Hybrid Architecture combining EfficientNet-B0 backbone with a
    dual-branch pooling head (Attention + Global Max) to capture
    both polyphonic and sparse sound events.
    """

    def __init__(self):
        super(HybridEfficientNet, self).__init__()

        # 1. Backbone: EfficientNet-B0 (Pretrained)
        # We use the features extractor part of the model
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        self.features = self.backbone.features

        # Modify first conv layer to accept 1 channel (Cite solution_lesson_node_00015)
        # EfficientNet-B0 first conv is named '0' inside 'features'
        first_conv = self.features[0][0]
        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:] = first_conv.weight.sum(dim=1, keepdim=True)
        self.features[0][0] = new_conv

        # EfficientNet-B0 outputs 1280 channels at the final layer
        self.feature_dim = 1280

        # 2. Hybrid Pooling Heads

        # Branch A: Attention Pooling (for polyphonic/continuous events)
        # We do NOT pool frequency dimension here to preserve resolution (Cite solution_lesson_node_00016)
        self.att_pooling = AttentionPooling(self.feature_dim)

        # Branch B: Global Max Pooling (for sparse/transient events)
        # Max pool over entire spatial map (Freq, Time)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # 3. Classifier
        # Input dimension is 2 * feature_dim (concatenation of both branches)
        self.classifier = nn.Linear(self.feature_dim * 2, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 3, Freq, Time)
        Returns:
            logits: Output logits of shape (Batch, Num_Classes)
        """
        # Extract features: (Batch, 1280, F', T')
        x = self.features(x)

        # --- Branch A: Attention Pooling ---
        # Flatten spatial dimensions (Freq, Time) -> Sequence
        # x shape: (Batch, 1280, F', T')
        batch_size, channels, freq, time = x.size()
        x_seq = x.view(batch_size, channels, -1)  # (Batch, 1280, F'*T')
        x_seq = x_seq.permute(0, 2, 1)  # (Batch, F'*T', 1280)

        # Apply Attention
        out_att = self.att_pooling(x_seq)  # (Batch, 1280)

        # --- Branch B: Global Max Pooling ---
        # Max over F' and T': (Batch, 1280, 1, 1)
        out_max = self.global_max_pool(x)
        # Flatten: (Batch, 1280)
        out_max = out_max.flatten(1)

        # --- Fusion ---
        # Concatenate embeddings: (Batch, 2560)
        out_cat = torch.cat([out_att, out_max], dim=1)

        # --- Classification ---
        logits = self.classifier(out_cat)

        return logits
