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


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module.
    Cite solution_lesson_node_00047
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class CABasicBlock(nn.Module):
    """
    Wrapper for BasicBlock to include Coordinate Attention.
    Cite solution_lesson_node_00047
    """

    def __init__(self, block):
        super(CABasicBlock, self).__init__()
        self.block = block
        # Insert CA after the second convolution, before the residual addition
        self.ca = CoordAtt(block.conv2.out_channels)

    def forward(self, x):
        shortcut = x
        if self.block.downsample is not None:
            shortcut = self.block.downsample(x)

        x = self.block.conv1(x)
        x = self.block.bn1(x)
        x = self.block.act1(x)

        x = self.block.conv2(x)
        x = self.block.bn2(x)

        # Apply Coordinate Attention
        x = self.ca(x)

        if self.block.drop_path is not None:
            x = self.block.drop_path(x)

        x = x + shortcut
        x = self.block.act2(x)
        return x


class SKResNetCRNN(nn.Module):
    """
    Time-Preserving ResNet-18 CRNN with Coordinate Attention.
    Cite solution_lesson_node_00047

    Architecture:
    1. Backbone: resnet18 (timm) with Coordinate Attention blocks.
    2. Temporal: Bi-directional GRU.
    3. Aggregation: Attention Pooling.
    4. Head: Linear Classification Layer.
    """

    def __init__(self):
        super(SKResNetCRNN, self).__init__()

        # 1. Load Backbone
        self.backbone = timm.create_model(
            config.MODEL_BACKBONE, pretrained=config.USE_PRETRAINED
        )

        # 2. Modify first convolution for 1-channel input
        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        with torch.no_grad():
            new_conv1.weight[:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)
            if original_conv1.bias is not None:
                new_conv1.bias[:] = original_conv1.bias

        self.backbone.conv1 = new_conv1

        # 3. Replace BasicBlocks with CABasicBlocks
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.backbone, layer_name)
            for i in range(len(layer)):
                layer[i] = CABasicBlock(layer[i])

        # 4. Time-Preserving Stride Modification
        # Change strides in layer3 and layer4 from (2, 2) to (2, 1)
        layers_to_modify = ["layer3", "layer4"]
        for layer_name in layers_to_modify:
            layer = getattr(self.backbone, layer_name)
            # The stride is in the conv1 of the first block of the layer (usually)
            # But since we wrapped it, we need to access the internal block
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    if m.stride == (2, 2):
                        m.stride = (2, 1)

        # 5. Determine Backbone Output Dimension
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
