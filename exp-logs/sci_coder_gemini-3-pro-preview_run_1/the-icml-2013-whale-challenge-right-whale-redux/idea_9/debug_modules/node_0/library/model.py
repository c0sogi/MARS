import torch
import torch.nn as nn
import timm
import library.config as config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted sum of the input sequence based on learned attention weights.
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
        # weights shape: (Batch, Time, 1)
        weights = self.attention(x)
        # context shape: (Batch, Features)
        context = torch.sum(x * weights, dim=1)
        return context


class SKResNetCRNN(nn.Module):
    """
    Time-Preserving Selective Kernel (SK) ResNet-18 CRNN.

    Architecture:
    1. Backbone: skresnet18 (timm) with modified strides for time preservation.
    2. Temporal: Bi-directional GRU.
    3. Aggregation: Attention Pooling.
    4. Head: Linear Classification Layer.
    """

    def __init__(self):
        super(SKResNetCRNN, self).__init__()

        # 1. Load Backbone
        # We load the full model first to easily access layers
        self.backbone = timm.create_model(
            config.MODEL_BACKBONE, pretrained=config.USE_PRETRAINED
        )

        # 2. Modify first convolution for 1-channel input
        # Standard ResNet structure in timm has 'conv1'
        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Initialize new conv1 weights by averaging the original RGB weights
        with torch.no_grad():
            new_conv1.weight[:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)
            if original_conv1.bias is not None:
                new_conv1.bias[:] = original_conv1.bias

        self.backbone.conv1 = new_conv1

        # 3. Time-Preserving Stride Modification
        # Change strides in layer3 and layer4 from (2, 2) to (2, 1)
        # This preserves temporal resolution while downsampling frequency
        layers_to_modify = ["layer3", "layer4"]
        for layer_name in layers_to_modify:
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                for m in layer.modules():
                    if isinstance(m, nn.Conv2d):
                        if m.stride == (2, 2):
                            m.stride = (2, 1)

        # 4. Determine Backbone Output Dimension
        # skresnet18 (like resnet18) layer4 outputs 512 channels
        self.backbone_out_dim = 512

        # 5. Temporal Modeling (Bi-Directional GRU)
        self.gru = nn.GRU(
            input_size=self.backbone_out_dim,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Aggregation (Attention Pooling)
        # Bidirectional GRU outputs 2 * hidden_size
        self.att_pool = AttentionPooling(config.GRU_HIDDEN_SIZE * 2)

        # 7. Classification Head
        self.fc = nn.Linear(config.GRU_HIDDEN_SIZE * 2, config.NUM_CLASSES)

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        # forward_features returns (Batch, Channels, Freq', Time')
        x = self.backbone.forward_features(x)

        # Pool Frequency Dimension
        # Average over the remaining frequency bins
        # x: (Batch, 512, F', T') -> (Batch, 512, T')
        x = torch.mean(x, dim=2)

        # Prepare for GRU
        # Permute to (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Attention Pooling
        # x: (Batch, Time, 2*Hidden) -> (Batch, 2*Hidden)
        x = self.att_pool(x)

        # Classification
        # Output logits (BCEWithLogitsLoss expected)
        logits = self.fc(x)

        return logits
