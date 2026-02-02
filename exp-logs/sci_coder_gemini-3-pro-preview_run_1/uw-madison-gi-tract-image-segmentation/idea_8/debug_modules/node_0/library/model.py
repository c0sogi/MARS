import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import SEQ_LENGTH, NUM_CLASSES, IN_CHANNELS, IMG_SIZE


class ConvLSTMCell(nn.Module):
    """
    A Convolutional LSTM Cell for processing spatial sequences.
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

        # Concatenate input and hidden state along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split into 4 gates: Input, Forget, Output, Cell Gate
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class BiConvLSTM(nn.Module):
    """
    Bi-Directional ConvLSTM that processes a sequence forward and backward.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(BiConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        self.fwd_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)
        self.bwd_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        b, t, c, h, w = x.size()

        h_fwd, c_fwd = self.fwd_cell.init_hidden(b, (h, w))
        h_bwd, c_bwd = self.bwd_cell.init_hidden(b, (h, w))

        fwd_outputs = []
        bwd_outputs = []

        # Forward Pass
        for i in range(t):
            h_fwd, c_fwd = self.fwd_cell(x[:, i], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)

        # Backward Pass
        for i in range(t - 1, -1, -1):
            h_bwd, c_bwd = self.bwd_cell(x[:, i], (h_bwd, c_bwd))
            bwd_outputs.insert(0, h_bwd)

        fwd_outputs = torch.stack(fwd_outputs, dim=1)
        bwd_outputs = torch.stack(bwd_outputs, dim=1)

        # Concatenate forward and backward outputs along channel dimension
        # Output channels = hidden_dim * 2
        output = torch.cat([fwd_outputs, bwd_outputs], dim=2)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            # Handle potential rounding errors in dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RecurrentUNet(nn.Module):
    """
    Recurrent U-Net with ResNet-34 Backbone and Deep Supervision.
    """

    def __init__(self, backbone="resnet34", pretrained=True):
        super().__init__()

        # --- Encoder (ResNet34) ---
        try:
            self.encoder = models.resnet34(weights="DEFAULT")
        except:
            self.encoder = models.resnet34(pretrained=True)

        # Modify first layer for 1-channel input (Grayscale)
        original_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize with average of RGB weights
        with torch.no_grad():
            self.encoder.conv1.weight[:] = (
                original_conv.weight.sum(dim=1, keepdim=True) / 3.0
            )

        # Encoder Layers
        self.enc0 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )  # 64, H/2
        self.enc1 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)  # 64, H/4
        self.enc2 = self.encoder.layer2  # 128, H/8
        self.enc3 = self.encoder.layer3  # 256, H/16
        self.enc4 = self.encoder.layer4  # 512, H/32

        # --- Bottleneck (Recurrent) ---
        # Input: 512 channels. Output: 256 hidden * 2 directions = 512 channels.
        self.bottleneck = BiConvLSTM(
            input_dim=512, hidden_dim=256, kernel_size=3, num_layers=1
        )

        # --- Decoder ---
        # Dec4: In(512), Skip(256) -> Out(256)
        self.dec4 = DecoderBlock(512, 256, 256)
        # Dec3: In(256), Skip(128) -> Out(128)
        self.dec3 = DecoderBlock(256, 128, 128)
        # Dec2: In(128), Skip(64) -> Out(64)
        self.dec2 = DecoderBlock(128, 64, 64)
        # Dec1: In(64), Skip(64) -> Out(32)
        self.dec1 = DecoderBlock(64, 64, 32)

        # Final Upsample to full resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # --- Heads ---
        self.final_head = nn.Conv2d(32, NUM_CLASSES, kernel_size=1)

        # Deep Supervision Heads
        # Aux1 attached to Dec2 output (H/4)
        self.aux1_head = nn.Conv2d(64, NUM_CLASSES, kernel_size=1)
        # Aux2 attached to Dec3 output (H/8)
        self.aux2_head = nn.Conv2d(128, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 1, T, H, W) -> Permute to (B, T, 1, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        b, t, c, h, w = x.shape

        # Flatten time dimension for Time-Distributed Encoder: (B*T, 1, H, W)
        x_flat = x.reshape(b * t, c, h, w)

        # Encoder Pass
        e0 = self.enc0(x_flat)  # (B*T, 64, H/2, W/2)
        e1 = self.enc1(e0)  # (B*T, 64, H/4, W/4)
        e2 = self.enc2(e1)  # (B*T, 128, H/8, W/8)
        e3 = self.enc3(e2)  # (B*T, 256, H/16, W/16)
        e4 = self.enc4(e3)  # (B*T, 512, H/32, W/32)

        # Reshape e4 for LSTM: (B, T, 512, H/32, W/32)
        _, c4, h4, w4 = e4.shape
        lstm_in = e4.reshape(b, t, c4, h4, w4)

        # Bottleneck (BiConvLSTM)
        lstm_out = self.bottleneck(lstm_in)  # (B, T, 512, H/32, W/32)

        # Select Center Slice for Decoder
        center_idx = t // 2
        bottleneck_feat = lstm_out[:, center_idx]  # (B, 512, H/32, W/32)

        # Helper to extract center skip connection
        def get_center_skip(feat, b, t):
            _, cf, hf, wf = feat.shape
            feat_reshaped = feat.reshape(b, t, cf, hf, wf)
            return feat_reshaped[:, center_idx]

        skip3 = get_center_skip(e3, b, t)
        skip2 = get_center_skip(e2, b, t)
        skip1 = get_center_skip(e1, b, t)
        skip0 = get_center_skip(e0, b, t)

        # Decoder Pass
        d4 = self.dec4(bottleneck_feat, skip3)  # -> 256, H/16

        d3 = self.dec3(d4, skip2)  # -> 128, H/8
        aux2 = self.aux2_head(d3)  # Aux 2 Output

        d2 = self.dec2(d3, skip1)  # -> 64, H/4
        aux1 = self.aux1_head(d2)  # Aux 1 Output

        d1 = self.dec1(d2, skip0)  # -> 32, H/2

        # Final Upsample
        out = self.final_up(d1)  # -> 32, H, W
        final = self.final_head(out)  # -> NumClasses, H, W

        # Upsample Aux heads to target H, W for loss calculation
        if self.training:
            aux1 = F.interpolate(
                aux1, size=(h, w), mode="bilinear", align_corners=False
            )
            aux2 = F.interpolate(
                aux2, size=(h, w), mode="bilinear", align_corners=False
            )
            return final, aux1, aux2
        else:
            return final
