import torch
import torch.nn as nn
import torchvision.models as models
import timm


class SpineLocalizer(nn.Module):
    """
    Stage 1: 2D U-Net for Spine Localization.
    Uses a ResNet18 backbone as the encoder to predict a segmentation mask
    identifying the cervical spine.
    """

    def __init__(self, pretrained=True):
        super(SpineLocalizer, self).__init__()

        # Load ResNet18 backbone
        # We use the standard torchvision model
        weights = "DEFAULT" if pretrained else None
        self.base_model = models.resnet18(weights=weights)

        # Encoder blocks (extracting layers from ResNet18)
        self.encoder0 = nn.Sequential(
            self.base_model.conv1, self.base_model.bn1, self.base_model.relu
        )  # Output: 64 ch, H/2, W/2

        self.encoder1 = self.base_model.maxpool  # Output: 64 ch, H/4, W/4
        self.encoder2 = self.base_model.layer1  # Output: 64 ch, H/4, W/4
        self.encoder3 = self.base_model.layer2  # Output: 128 ch, H/8, W/8
        self.encoder4 = self.base_model.layer3  # Output: 256 ch, H/16, W/16
        self.encoder5 = self.base_model.layer4  # Output: 512 ch, H/32, W/32

        # Decoder blocks
        # Up-sampling + Skip Connection concatenation + ConvBlock
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5 = self._conv_block(256 + 256, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = self._conv_block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = self._conv_block(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = self._conv_block(64 + 64, 64)  # Concatenating with encoder0 output

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)  # Final 1x1 conv for mask

    def _conv_block(self, in_ch, out_ch):
        """Helper to create a standard conv-bn-relu block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (Batch, Channels, Height, Width)
        # If input is 1 channel (grayscale), repeat to 3 for ResNet compatibility
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # --- Encoder ---
        e0 = self.encoder0(x)  # 64, H/2
        e1 = self.encoder1(e0)  # 64, H/4
        e2 = self.encoder2(e1)  # 64, H/4
        e3 = self.encoder3(e2)  # 128, H/8
        e4 = self.encoder4(e3)  # 256, H/16
        e5 = self.encoder5(e4)  # 512, H/32

        # --- Decoder ---
        # Block 5
        d5 = self.up5(e5)
        # Concatenate with e4
        if d5.size() != e4.size():
            d5 = torch.nn.functional.interpolate(d5, size=e4.shape[2:])
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.conv5(d5)

        # Block 4
        d4 = self.up4(d5)
        if d4.size() != e3.size():
            d4 = torch.nn.functional.interpolate(d4, size=e3.shape[2:])
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.conv4(d4)

        # Block 3
        d3 = self.up3(d4)
        if d3.size() != e2.size():
            d3 = torch.nn.functional.interpolate(d3, size=e2.shape[2:])
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.conv3(d3)

        # Block 2
        d2 = self.up2(d3)
        if d2.size() != e0.size():
            d2 = torch.nn.functional.interpolate(d2, size=e0.shape[2:])
        d2 = torch.cat([d2, e0], dim=1)
        d2 = self.conv2(d2)

        # Block 1 (Final upsample to original resolution)
        d1 = self.up1(d2)
        # Interpolate to match input size exactly if necessary
        if d1.size(2) != x.size(2) or d1.size(3) != x.size(3):
            d1 = torch.nn.functional.interpolate(d1, size=(x.size(2), x.size(3)))

        out = self.conv1(d1)
        return out


class SliceEncoder(nn.Module):
    """
    Stage 2: 2.5D CNN Encoder.
    Extracts a feature vector from a stack of slices (e.g., 3 slices).
    Uses a standard backbone like ResNet50.
    """

    def __init__(self, backbone_name="resnet50", pretrained=True):
        super(SliceEncoder, self).__init__()

        # Try creating model with timm first for easy feature extraction
        try:
            self.model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,  # Removes the classification head
                global_pool="avg",  # Applies Global Average Pooling
            )
            self.out_dim = self.model.num_features
        except Exception:
            # Fallback to torchvision
            if "resnet50" in backbone_name:
                weights = "DEFAULT" if pretrained else None
                m = models.resnet50(weights=weights)
                self.out_dim = m.fc.in_features
                m.fc = nn.Identity()  # Remove FC
                self.model = m
            else:
                raise ValueError(
                    f"Backbone {backbone_name} not supported without timm."
                )

    def forward(self, x):
        # x: (Batch, 3, Height, Width)
        # Returns: (Batch, Feature_Dim)
        return self.model(x)


class SequenceAggregator(nn.Module):
    """
    Stage 3: RNN Aggregator.
    Processes a sequence of feature vectors (representing a patient scan)
    to predict fracture probabilities.
    """

    def __init__(
        self, input_dim, hidden_dim=256, num_layers=2, num_classes=8, dropout=0.2
    ):
        super(SequenceAggregator, self).__init__()

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classification Head
        # We aggregate features using both Max and Average pooling
        # Input size = Hidden * 2 (Bidirectional) * 2 (Max + Avg)
        self.head_input_dim = hidden_dim * 2 * 2

        self.fc = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # Pass through GRU
        # out: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.gru(x)

        # Global Pooling over the sequence dimension (dim=1)
        # This aggregates information from the entire spine
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        # Concatenate pooled features
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Predict logits
        logits = self.fc(combined)

        return logits
