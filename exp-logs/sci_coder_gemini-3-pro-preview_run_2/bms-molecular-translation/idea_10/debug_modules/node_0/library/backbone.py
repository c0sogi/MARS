import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights
from library.config import Config


class AnisotropicResNet(nn.Module):
    """
    Anisotropic ResNet-34 backbone.

    This model is designed to act as a feature extractor for chemical structure recognition.
    It modifies the standard ResNet-34 architecture to decouple vertical and horizontal
    downsampling.

    Modifications:
    1. Input Layer: Accepts 1-channel grayscale images instead of 3-channel RGB.
    2. Strides:
       - Vertical strides are maintained to collapse the height dimension (capturing structure).
       - Horizontal strides are removed in deeper layers to preserve resolution for the
         sequence modeling task (CTC constraint T >= L).

    Target Downsampling:
    - Height: /32 (Standard ResNet)
    - Width:  /4  (Modified)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load pre-trained ResNet-34
        # Using weights="DEFAULT" for the best available pre-trained weights
        weights = ResNet34_Weights.DEFAULT
        self.resnet = models.resnet34(weights=weights)

        # ---------------------------------------------------------
        # 1. Modify Input Layer
        # ---------------------------------------------------------
        # Standard ResNet conv1 is: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We need input_channels -> 64.
        # We keep the original weights for the first channel or average them?
        # A simple approach is to create a new layer. Averaging weights is better for transfer learning.
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Initialize the new conv1 weights by averaging the original RGB weights
        # This preserves some edge detection filters
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        # ---------------------------------------------------------
        # 2. Modify Strides for Anisotropy
        # ---------------------------------------------------------
        # ResNet structure:
        # conv1 (s=2) -> maxpool (s=2) -> layer1 (s=1) -> layer2 (s=2) -> layer3 (s=2) -> layer4 (s=2)
        # Current total width stride: 2 * 2 * 1 * 2 * 2 * 2 = 32
        # Target total width stride: 4
        # We need to change layer2, layer3, layer4 strides from (2, 2) to (2, 1).

        self._modify_stride(self.resnet.layer2)
        self._modify_stride(self.resnet.layer3)
        self._modify_stride(self.resnet.layer4)

        # ---------------------------------------------------------
        # 3. Output Processing
        # ---------------------------------------------------------
        # We remove the classification head (fc) and average pooling (avgpool) from the original resnet
        # We will implement our own pooling in forward()
        del self.resnet.fc
        del self.resnet.avgpool

        # Projection layer to map ResNet output channels (512) to Encoder dimension (e.g., 384)
        self.resnet_out_channels = 512
        self.projection = nn.Linear(self.resnet_out_channels, config.encoder_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.relu = nn.ReLU(inplace=True)

    def _modify_stride(self, layer_block):
        """
        Modifies the first BasicBlock in a ResNet layer to use stride (2, 1) instead of (2, 2).
        This preserves horizontal resolution.
        """
        # The first block in the layer handles the downsampling
        block = layer_block[0]

        # Modify the first convolution in the block
        # BasicBlock definition: conv1 -> bn1 -> relu -> conv2 -> bn2
        if block.conv1.stride == (2, 2):
            block.conv1.stride = (2, 1)

        # Modify the downsample layer (1x1 conv) if it exists
        if block.downsample is not None:
            # downsample is usually Sequential(Conv2d, BatchNorm2d)
            conv_down = block.downsample[0]
            if conv_down.stride == (2, 2):
                conv_down.stride = (2, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Feature sequence of shape (B, Seq_Len, Encoder_Dim)
        """
        # ---------------------------------------------------------
        # Feature Extraction
        # ---------------------------------------------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Current shape: (B, 512, H/32, W/4)
        # Example: Input (192, W) -> Output (6, W/4)

        # ---------------------------------------------------------
        # Vertical Collapse
        # ---------------------------------------------------------
        # We collapse the remaining height dimension to 1 using adaptive average pooling.
        # This aggregates vertical features into a single vector per column.
        x = nn.functional.adaptive_avg_pool2d(x, (1, None))
        # Shape: (B, 512, 1, W/4)

        # Squeeze height dimension
        x = x.squeeze(2)
        # Shape: (B, 512, W/4)

        # ---------------------------------------------------------
        # Sequence Formatting
        # ---------------------------------------------------------
        # Permute to (B, W/4, 512) for sequence modeling (Batch First)
        x = x.permute(0, 2, 1)
        # Shape: (B, Seq_Len, 512)

        # Project to encoder dimension
        x = self.projection(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Shape: (B, Seq_Len, Encoder_Dim)

        return x
