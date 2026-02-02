import torch
import torch.nn as nn
import timm
from library.config import ModelConfig


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


class ContextAwareEfficientNet(nn.Module):
    """
    Hybrid architecture combining a Dilated EfficientNet backbone with a
    Transformer-based Temporal Context Module and Attentive Pooling.
    """

    def __init__(self):
        super(ContextAwareEfficientNet, self).__init__()

        # Configuration
        self.num_classes = ModelConfig.num_classes
        self.d_model = ModelConfig.d_model
        self.dropout = ModelConfig.dropout

        # 1. Backbone: Dilated EfficientNet-B2
        # - pretrained=True: Start with ImageNet weights.
        # - in_chans=1: Automatically adapts first conv layer by averaging RGB weights.
        # - output_stride=16: Replaces the final stride-2 downsampling with dilation=2.
        #   This preserves temporal resolution (feature map width ~7 instead of ~3).
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
            # Time = 101 frames (1s @ 16k sr, 160 hop)
            dummy_input = torch.zeros(
                1, ModelConfig.in_channels, ModelConfig.n_mels, 101
            )
            features = self.backbone(dummy_input)
            # features shape: (1, C, H, W)
            self.backbone_channels = features.shape[1]

        # 2. Projection Layer
        # Projects high-dim backbone features (e.g., 1408) to d_model (256)
        self.projection = nn.Linear(self.backbone_channels, self.d_model)

        # 3. Temporal Context Module (Transformer Encoder)
        # Captures sequential dependencies between frames.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=ModelConfig.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=ModelConfig.num_layers
        )

        # 4. Attentive Pooling Head
        self.att_pooling = AttentivePooling(self.d_model)

        # 5. Classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

        # Initialize custom layers
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the added layers (Projection, Transformer, Head).
        Does not re-initialize the pretrained backbone.
        """
        # Init Projection and Classifier
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # Init Transformer parameters
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        # H' is reduced frequency dim, W' is reduced time dim (~7 frames)
        x = self.backbone(x)

        # 2. Frequency Pooling
        # We assume the frequency dimension (H) contains spectral patterns which the CNN
        # has already processed. We average over H to get a vector per time step.
        # x: (B, C, H', W') -> (B, C, W')
        x = x.mean(dim=2)

        # 3. Sequence Preparation
        # Permute to (Batch, Time, Channels) for Transformer
        x = x.permute(0, 2, 1)

        # 4. Projection
        # x: (B, T, C) -> (B, T, d_model)
        x = self.projection(x)

        # 5. Temporal Context
        # x: (B, T, d_model)
        x = self.transformer(x)

        # 6. Attentive Pooling
        # Aggregates time steps into a single vector
        # x: (B, T, d_model) -> (B, d_model)
        x = self.att_pooling(x)

        # 7. Classification
        # x: (B, d_model) -> (B, num_classes)
        logits = self.classifier(x)

        return logits
