import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class AttentivePooling(nn.Module):
    """
    Attentive Pooling layer that dynamically weights active speech segments.
    Input: (Batch, Channels, Time)
    Output: (Batch, Channels)
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # Calculate attention weights: (Batch, 1, Time)
        w = self.attention(x)
        # Weighted sum over time: (Batch, Channels)
        x = torch.sum(x * w, dim=2)
        return x


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.

    Modifications:
    1. First Conv layer adapted for 1-channel input (weights averaged).
    2. Final stage modified to use Dilated Convolutions (stride=1, dilation=2)
       to preserve feature map resolution.
    3. Global Average Pooling over Frequency, Attentive Pooling over Time.
    """

    def __init__(self, num_classes):
        super(DilatedEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b2(weights=weights)

        # 2. Adapt First Layer for 1-Channel Input
        # features[0] is Conv2dNormActivation, features[0][0] is the Conv2d
        first_conv_layer = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias is not None,
        )

        # Initialize weights by averaging the original RGB weights
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(
                first_conv_layer.weight, dim=1, keepdim=True
            )

        self.backbone.features[0][0] = new_conv

        # 3. Modify Final Stage for Dilated Convolutions
        # Find the index of the last block that performs downsampling (stride=2)
        last_stride_2_idx = -1
        for idx, m in enumerate(self.backbone.features):
            is_downsample = False
            for sub in m.modules():
                if isinstance(sub, nn.Conv2d) and sub.stride == (2, 2):
                    is_downsample = True
                    break
            if is_downsample:
                last_stride_2_idx = idx

        # Apply dilation to the last downsampling stage and subsequent blocks
        if last_stride_2_idx != -1:

            def modify_block(module, stride=(1, 1), dilation=(2, 2)):
                for name, child in module.named_children():
                    if isinstance(child, nn.Conv2d):
                        # Remove stride
                        if child.stride == (2, 2):
                            child.stride = stride

                        # Apply dilation to spatial convolutions
                        if child.kernel_size[0] > 1:
                            child.dilation = dilation
                            # Adjust padding to maintain 'same' output size
                            p = ((child.kernel_size[0] - 1) * child.dilation[0]) // 2
                            child.padding = (p, p)
                    else:
                        modify_block(child, stride, dilation)

            # Apply modifications from the last downsampling block to the end of the feature extractor
            # (excluding the final 1x1 projection layer if it exists separately, though usually safe to iterate)
            for i in range(last_stride_2_idx, len(self.backbone.features) - 1):
                modify_block(self.backbone.features[i], stride=(1, 1), dilation=(2, 2))

        # 4. Define Head
        # Determine output channels dynamically
        last_layer = self.backbone.features[-1]
        out_channels = 1408  # Default for B2
        for m in last_layer.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.out_channels
                break

        self.pool = AttentivePooling(out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        x = self.backbone.features(x)  # Output: (Batch, C, F', T')

        # Collapse Frequency Dimension (Global Average Pooling)
        # We average over frequency but keep time for the attentive pooling
        x = torch.mean(x, dim=2)  # Output: (Batch, C, T')

        # Attentive Pooling over Time
        x = self.pool(x)  # Output: (Batch, C)

        # Classification
        x = self.classifier(x)  # Output: (Batch, NumClasses)

        return x
