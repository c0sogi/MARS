import torch
import torch.nn as nn
import timm
from library.config import ModelConfig, AudioConfig


class AttentivePooling(nn.Module):
    """
    Applies attention mechanism to aggregate temporal features.
    Computes a weighted sum of the time steps, allowing the model to focus
    on the most informative parts of the audio clip.
    """

    def __init__(self, input_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        # alpha shape: (Batch, Time, 1)
        alpha = self.attention(x)
        # Weighted sum over the Time dimension
        x = torch.sum(x * alpha, dim=1)
        return x


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet backbone with Attentive Pooling.
    Removes the Transformer module to improve training stability.
    """

    def __init__(self):
        super(DilatedEfficientNet, self).__init__()

        # Configuration
        self.num_classes = ModelConfig.num_classes
        self.dropout = ModelConfig.dropout

        # 1. Backbone: Dilated EfficientNet-B2
        # - pretrained=True: Start with ImageNet weights.
        # - in_chans=1: Automatically adapts first conv layer by averaging RGB weights.
        # - output_stride=16: Replaces the final stride-2 downsampling with dilation=2.
        #   This preserves temporal resolution.
        # - global_pool='': Return the full feature map (B, C, H, W).
        self.backbone = timm.create_model(
            ModelConfig.model_name,
            pretrained=ModelConfig.pretrained,
            in_chans=ModelConfig.in_channels,
            num_classes=0,
            global_pool="",
            output_stride=16,
        )

        # Determine backbone output channels dynamically
        with torch.no_grad():
            # Dummy input: (Batch, Channels, Freq, Time)
            dummy_input = torch.zeros(
                1, ModelConfig.in_channels, AudioConfig.n_mels, 101
            )
            features = self.backbone(dummy_input)
            # features shape: (1, C, H, W)
            self.backbone_channels = features.shape[1]

        # 2. Attentive Pooling Head
        # Operates directly on backbone channels (e.g., 1408 for B2)
        self.att_pooling = AttentivePooling(self.backbone_channels)

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.backbone_channels, self.num_classes),
        )

        # Initialize custom layers
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the added layers (Head).
        Does not re-initialize the pretrained backbone.
        """
        # Init Classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Init Attentive Pooling
        for m in self.att_pooling.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Log-Mel Spectrograms. Shape (Batch, 1, 128, 101).
        Returns:
            logits (torch.Tensor): Class probabilities. Shape (Batch, Num_Classes).
        """
        # 1. Backbone Feature Extraction
        # x: (B, 1, 128, 101) -> (B, C, H', W')
        x = self.backbone(x)

        # 2. Frequency Pooling
        # x: (B, C, H', W') -> (B, C, W')
        x = x.mean(dim=2)

        # 3. Sequence Preparation
        # Permute to (Batch, Time, Channels)
        # x: (B, C, W') -> (B, W', C)
        x = x.permute(0, 2, 1)

        # 4. Attentive Pooling
        # Aggregates time steps into a single vector
        # x: (B, T, C) -> (B, C)
        x = self.att_pooling(x)

        # 5. Classification
        # x: (B, C) -> (B, num_classes)
        logits = self.classifier(x)

        return logits
