import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import NUM_CLASSES, SEQ_LEN


class ConvLSTMCell(nn.Module):
    """
    Basic Convolutional LSTM Cell.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block with Bilinear Upsampling.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle padding issues if dimensions are not perfectly divisible
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class RecurrentUNet(nn.Module):
    """
    Recurrent U-Net with ResNet-18 Encoder and Bi-Directional ConvLSTM Bottleneck.
    """

    def __init__(self, num_classes=NUM_CLASSES, seq_len=SEQ_LEN, pretrained=True):
        super(RecurrentUNet, self).__init__()
        self.seq_len = seq_len

        # --- Encoder (ResNet18) ---
        # Cite solution_lesson_node_00012: Prefer ResNet18 for efficiency
        self.encoder = models.resnet18(weights="DEFAULT" if pretrained else None)

        # Disable gradients for initial layers if needed (optional, keeping trainable here)

        # --- Bottleneck (BiConvLSTM) ---
        # ResNet34 Layer 4 has 512 channels.
        # We use hidden_dim=256 for each direction, so concat is 512.
        self.lstm_hidden_dim = 256
        self.lstm_fwd = ConvLSTMCell(
            input_dim=512, hidden_dim=self.lstm_hidden_dim, kernel_size=3, bias=True
        )
        self.lstm_bwd = ConvLSTMCell(
            input_dim=512, hidden_dim=self.lstm_hidden_dim, kernel_size=3, bias=True
        )

        # --- Decoder ---
        # Bottleneck output: 256 (fwd) + 256 (bwd) = 512 channels

        # Block 1: In 512, Skip 256 (Layer3), Out 256
        self.dec1 = DecoderBlock(512, 256, 256)

        # Block 2: In 256, Skip 128 (Layer2), Out 128
        self.dec2 = DecoderBlock(256, 128, 128)

        # Block 3: In 128, Skip 64 (Layer1), Out 64
        self.dec3 = DecoderBlock(128, 64, 64)

        # Block 4: In 64, Skip 64 (Layer0/Conv1), Out 32
        self.dec4 = DecoderBlock(64, 64, 32)

        # Final Output
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Channels, H, W)
        b, s, c, h, w = x.size()

        # Flatten batch and sequence for Time-Distributed Encoder
        x = x.view(b * s, c, h, w)

        # --- Encoder Forward ---
        # Extract skip connections
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x0 = x  # Skip 0: (B*S, 64, H/2, W/2)

        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)  # Skip 1: (B*S, 64, H/4, W/4)
        x2 = self.encoder.layer2(x1)  # Skip 2: (B*S, 128, H/8, W/8)
        x3 = self.encoder.layer3(x2)  # Skip 3: (B*S, 256, H/16, W/16)
        x4 = self.encoder.layer4(x3)  # Bottleneck Input: (B*S, 512, H/32, W/32)

        # --- Bi-Directional ConvLSTM ---
        # Reshape for LSTM: (B, S, C, H, W)
        feat_h, feat_w = x4.size(2), x4.size(3)
        lstm_in = x4.view(b, s, 512, feat_h, feat_w)

        # Initialize states
        h_fwd, c_fwd = self.lstm_fwd.init_hidden(b, (feat_h, feat_w), x.device)
        h_bwd, c_bwd = self.lstm_bwd.init_hidden(b, (feat_h, feat_w), x.device)

        # Forward pass
        fwd_outputs = []
        for t in range(s):
            h_fwd, c_fwd = self.lstm_fwd(lstm_in[:, t], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)

        # Backward pass
        bwd_outputs = []
        for t in range(s - 1, -1, -1):
            h_bwd, c_bwd = self.lstm_bwd(lstm_in[:, t], (h_bwd, c_bwd))
            bwd_outputs.insert(0, h_bwd)

        # --- Select Central Slice ---
        center_idx = s // 2

        # Concatenate forward and backward features for the central slice
        # Shape: (B, 512, H/32, W/32)
        lstm_out = torch.cat([fwd_outputs[center_idx], bwd_outputs[center_idx]], dim=1)

        # Get skip connections for the central slice
        # Reshape skips to (B, S, ...) and select center
        skip3 = x3.view(b, s, *x3.shape[1:])[:, center_idx]
        skip2 = x2.view(b, s, *x2.shape[1:])[:, center_idx]
        skip1 = x1.view(b, s, *x1.shape[1:])[:, center_idx]
        skip0 = x0.view(b, s, *x0.shape[1:])[:, center_idx]

        # --- Decoder Forward ---
        d1 = self.dec1(lstm_out, skip3)
        d2 = self.dec2(d1, skip2)
        d3 = self.dec3(d2, skip1)
        d4 = self.dec4(d3, skip0)

        # Final prediction
        out = self.final_conv(d4)

        return out
