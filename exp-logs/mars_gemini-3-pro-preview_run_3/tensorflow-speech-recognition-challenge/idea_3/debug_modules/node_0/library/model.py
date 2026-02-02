import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import MODEL_CONFIG


class AttentionPooling(nn.Module):
    """
    Implements Attention Pooling (weighted average) over the temporal dimension.
    Learns a weight for each time step to focus on the relevant parts of the command.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Calculate weights: (Batch, Time, 1)
        weights = self.attention(x)

        # Weighted sum: (Batch, Features)
        # sum( (Batch, Time, Features) * (Batch, Time, 1) ) over Time
        context = torch.sum(x * weights, dim=1)

        return context


class ResNet34BiGRU(nn.Module):
    """
    Hybrid CRNN architecture: ResNet34 Backbone + BiGRU + Attention Pooling.
    """

    def __init__(self):
        super(ResNet34BiGRU, self).__init__()

        # 1. Load Pretrained Backbone
        weights = ResNet34_Weights.IMAGENET1K_V1 if MODEL_CONFIG["pretrained"] else None
        self.backbone = resnet34(weights=weights)

        # 2. Modify First Conv for 1-Channel Input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Sum weights across the input channel dimension (dim 1) to preserve filters
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # 3. Modify Strides to Preserve Temporal Resolution
        # Standard ResNet34 downsamples by 32 (2^5). We want less downsampling for time.
        # We change layer3 and layer4 strides from 2 to 1.

        # Layer 3
        self.backbone.layer3[0].conv1.stride = (1, 1)
        self.backbone.layer3[0].downsample[0].stride = (1, 1)

        # Layer 4
        self.backbone.layer4[0].conv1.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        # 4. Define RNN and Head
        # ResNet34 layer4 output channels = 512
        self.cnn_out_dim = 512
        self.hidden_size = MODEL_CONFIG["hidden_size"]
        self.num_classes = MODEL_CONFIG["num_classes"]

        self.gru = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=MODEL_CONFIG["dropout"] if MODEL_CONFIG["dropout"] > 0 else 0,
        )

        # Bidirectional GRU outputs 2 * hidden_size
        self.rnn_out_dim = self.hidden_size * 2

        if MODEL_CONFIG.get("use_attention", True):
            self.pooling = AttentionPooling(self.rnn_out_dim)
        else:
            # Fallback to Global Average Pooling if attention is disabled
            self.pooling = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(MODEL_CONFIG["dropout"])
        self.fc = nn.Linear(self.rnn_out_dim, self.num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, n_mels, time)

        # --- CNN Backbone ---
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # Output x: (Batch, 512, F', T')

        # --- Prepare for RNN ---
        # Average over frequency dimension (F')
        x = torch.mean(x, dim=2)  # (Batch, 512, T')

        # Permute to (Batch, T', 512) for RNN
        x = x.permute(0, 2, 1)

        # --- RNN ---
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # Output x: (Batch, T', 2*Hidden)

        # --- Pooling ---
        if isinstance(self.pooling, AttentionPooling):
            x = self.pooling(x)  # (Batch, 2*Hidden)
        else:
            # AdaptiveAvgPool expects (Batch, Channels, Time)
            x = x.permute(0, 2, 1)
            x = self.pooling(x)
            x = x.squeeze(-1)

        # --- Classifier ---
        x = self.dropout(x)
        x = self.fc(x)

        return x
