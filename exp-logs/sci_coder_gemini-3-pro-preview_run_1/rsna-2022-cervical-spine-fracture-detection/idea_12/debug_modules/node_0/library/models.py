import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import DenseGCNConv
from library.config import Config


# ---------------------------------------------------------
# Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetLocalizer(nn.Module):
    def __init__(self, n_channels=1, n_classes=8):
        super(UNetLocalizer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder: EfficientNet-B0
        # We need to modify the first layer if input is not 3 channels
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Adjust first conv layer for grayscale input
        if n_channels != 3:
            original_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                n_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            # Initialize weights (average of RGB weights to preserve pre-training info)
            with torch.no_grad():
                new_conv.weight[:] = (
                    original_conv.weight.sum(dim=1, keepdim=True) / n_channels
                )
            backbone.features[0][0] = new_conv

        self.encoder = backbone.features

        # EfficientNet-B0 feature maps at different stages:
        # Index 1: stride 2, 16 channels
        # Index 2: stride 4, 24 channels
        # Index 3: stride 8, 40 channels
        # Index 5: stride 16, 112 channels
        # Index 7: stride 32, 1280 channels (after conv_head? No, features[7] is 320, conv_head is 1280)
        # Let's verify standard EfficientNet B0 structure in torchvision:
        # features[0]: Conv3x3 (stride 2) -> 32
        # features[1]: MBConv1 (stride 1) -> 16
        # features[2]: MBConv6 (stride 2) -> 24
        # features[3]: MBConv6 (stride 2) -> 40
        # features[4]: MBConv6 (stride 1) -> 80
        # features[5]: MBConv6 (stride 1) -> 112
        # features[6]: MBConv6 (stride 2) -> 192
        # features[7]: MBConv6 (stride 1) -> 320
        # features[8]: Conv1x1 -> 1280

        # We will use feature maps at strides 4, 8, 16, 32
        self.up1 = nn.ConvTranspose2d(1280, 112, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(112 + 112, 112)  # Skip from features[5] (112ch)

        self.up2 = nn.ConvTranspose2d(112, 40, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(40 + 40, 40)  # Skip from features[3] (40ch)

        self.up3 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(24 + 24, 24)  # Skip from features[2] (24ch)

        self.up4 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(16 + 16, 16)  # Skip from features[1] (16ch)

        # Final upsample to original size (stride 2 to 1)
        self.up5 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, H, W)

        # Encoder
        # features[0] -> stride 2
        x0 = self.encoder[0](x)
        # features[1] -> stride 2 (16 ch) -> Skip 4
        x1 = self.encoder[1](x0)
        # features[2] -> stride 4 (24 ch) -> Skip 3
        x2 = self.encoder[2](x1)
        # features[3] -> stride 8 (40 ch) -> Skip 2
        x3 = self.encoder[3](x2)
        # features[4]
        x4_pre = self.encoder[4](x3)
        # features[5] -> stride 16 (112 ch) -> Skip 1
        x5 = self.encoder[5](x4_pre)
        # features[6]
        x6_pre = self.encoder[6](x5)
        # features[7] -> stride 32 (320 ch)
        x7 = self.encoder[7](x6_pre)
        # features[8] -> stride 32 (1280 ch)
        x8 = self.encoder[8](x7)

        # Decoder
        d1 = self.up1(x8)
        # Resize x5 to match d1 if necessary (due to padding/odd dims)
        if d1.shape != x5.shape:
            d1 = F.interpolate(
                d1, size=x5.shape[2:], mode="bilinear", align_corners=True
            )
        d1 = torch.cat([x5, d1], dim=1)
        d1 = self.conv1(d1)

        d2 = self.up2(d1)
        if d2.shape != x3.shape:
            d2 = F.interpolate(
                d2, size=x3.shape[2:], mode="bilinear", align_corners=True
            )
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.conv2(d2)

        d3 = self.up3(d2)
        if d3.shape != x2.shape:
            d3 = F.interpolate(
                d3, size=x2.shape[2:], mode="bilinear", align_corners=True
            )
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv3(d3)

        d4 = self.up4(d3)
        if d4.shape != x1.shape:
            d4 = F.interpolate(
                d4, size=x1.shape[2:], mode="bilinear", align_corners=True
            )
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.conv4(d4)

        d5 = self.up5(d4)
        if d5.shape != x.shape:
            d5 = F.interpolate(
                d5, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        logits = self.out_conv(d5)
        return logits


# ---------------------------------------------------------
# Stage 2: Dual-Resolution Feature Encoder
# ---------------------------------------------------------
class DualStreamEncoder(nn.Module):
    def __init__(self, feature_dim=1280):
        super(DualStreamEncoder, self).__init__()

        # Branch A: Local Stream (High Res Crop + Mask) -> 2 Channels
        self.local_backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Modify first layer for 2 channels
        orig_conv_local = self.local_backbone.features[0][0]
        new_conv_local = nn.Conv2d(
            2,
            orig_conv_local.out_channels,
            kernel_size=orig_conv_local.kernel_size,
            stride=orig_conv_local.stride,
            padding=orig_conv_local.padding,
            bias=False,
        )
        with torch.no_grad():
            # Copy weights: Channel 0 (Image) gets avg of RGB, Channel 1 (Mask) gets avg of RGB
            new_conv_local.weight[:, 0, :, :] = orig_conv_local.weight.mean(dim=1)
            new_conv_local.weight[:, 1, :, :] = orig_conv_local.weight.mean(dim=1)
        self.local_backbone.features[0][0] = new_conv_local
        self.local_backbone.classifier = nn.Identity()  # Remove classifier

        # Branch B: Global Stream (Resized Full Slice) -> 1 Channel
        self.global_backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Modify first layer for 1 channel
        orig_conv_global = self.global_backbone.features[0][0]
        new_conv_global = nn.Conv2d(
            1,
            orig_conv_global.out_channels,
            kernel_size=orig_conv_global.kernel_size,
            stride=orig_conv_global.stride,
            padding=orig_conv_global.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv_global.weight[:] = (
                orig_conv_global.weight.sum(dim=1, keepdim=True) / 3.0
            )
        self.global_backbone.features[0][0] = new_conv_global
        self.global_backbone.classifier = nn.Identity()  # Remove classifier

        # Output dimension of EfficientNet B0 is 1280
        self.output_dim = 1280 * 2  # Concatenated

        # Optional projection to reduce dim
        self.proj = nn.Sequential(
            nn.Linear(self.output_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x_local, x_global):
        # x_local: (B, 2, H, W)
        # x_global: (B, 1, H, W)

        # Extract features
        # EfficientNet forward returns logits if classifier is present, but we set it to Identity.
        # However, the `forward` method of EfficientNet typically does avgpool + flatten + classifier.
        # Let's check torchvision implementation.
        # It does: features -> avgpool -> flatten -> classifier.
        # Since classifier is Identity, we get the flattened 1280 vector.

        feat_local = self.local_backbone(x_local)
        feat_global = self.global_backbone(x_global)

        # Concatenate
        combined = torch.cat([feat_local, feat_global], dim=1)

        # Project
        out = self.proj(combined)

        return out


# ---------------------------------------------------------
# Stage 3: Graph-Recurrent Aggregator
# ---------------------------------------------------------
class SpinalGraphAggregator(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, gcn_dim=128):
        super(SpinalGraphAggregator, self).__init__()

        # 1. Sequence Modeling
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.gru_out_dim = hidden_dim * 2

        # 2. Graph Convolution
        # 7 nodes (C1-C7)
        self.gcn1 = DenseGCNConv(self.gru_out_dim, gcn_dim)
        self.gcn2 = DenseGCNConv(gcn_dim, gcn_dim)
        self.relu = nn.ReLU()

        # 3. Prediction Heads
        # Per-vertebrae classifiers
        self.node_classifiers = nn.ModuleList([nn.Linear(gcn_dim, 1) for _ in range(7)])

        # Patient overall classifier
        # Input: Pooled graph features (Max pool over 7 nodes)
        self.patient_classifier = nn.Linear(gcn_dim, 1)

    def get_spinal_adjacency(self, device):
        # Create adjacency matrix for linear chain C1-C2-...-C7
        # 7 nodes.
        # A[i, j] = 1 if connected
        adj = torch.zeros((7, 7), device=device)
        # Add self loops
        adj.fill_diagonal_(1)
        # Add connections
        for i in range(6):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        return adj

    def forward(self, x, anatomical_probs):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim) - Sequence of slice features
            anatomical_probs: (Batch, Seq_Len, 8) - Probabilities for [Bg, C1..C7]
        """
        batch_size = x.size(0)

        # 1. Bi-GRU
        # gru_out: (Batch, Seq_Len, Hidden*2)
        gru_out, _ = self.gru(x)

        # 2. Anatomical Pooling
        # We want to aggregate Seq_Len into 7 Nodes based on anatomical_probs.
        # anatomical_probs[:, :, 1:] corresponds to C1..C7
        # Shape: (Batch, Seq_Len, 7)
        weights = anatomical_probs[:, :, 1:]

        # Normalize weights along sequence dimension to sum to 1 (avoid scaling issues)
        # Add epsilon to avoid div by zero
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        weights_norm = weights / weights_sum

        # Weighted Average:
        # (Batch, 7, Seq_Len) @ (Batch, Seq_Len, Hidden) -> (Batch, 7, Hidden)
        # Transpose weights to (B, 7, L)
        node_features = torch.bmm(weights_norm.transpose(1, 2), gru_out)

        # 3. Spinal GCN
        # Prepare Adjacency
        adj = self.get_spinal_adjacency(x.device)
        # Expand for batch: (Batch, 7, 7)
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # GCN Layers
        # Input: (Batch, 7, Hidden)
        g_x = self.gcn1(node_features, adj_batch)
        g_x = self.relu(g_x)
        g_x = self.gcn2(g_x, adj_batch)  # (Batch, 7, GCN_Dim)

        # 4. Predictions

        # A. Vertebrae predictions (C1-C7)
        # Apply specific classifier to each node
        vert_logits = []
        for i in range(7):
            # Node i features: (Batch, GCN_Dim)
            node_feat = g_x[:, i, :]
            logit = self.node_classifiers[i](node_feat)
            vert_logits.append(logit)

        # Stack logits: (Batch, 7)
        vert_logits = torch.cat(vert_logits, dim=1)
        vert_probs = torch.sigmoid(vert_logits)

        # B. Patient Overall prediction
        # Global Max Pooling over nodes
        graph_feat, _ = torch.max(g_x, dim=1)  # (Batch, GCN_Dim)
        patient_logit = self.patient_classifier(graph_feat)
        patient_prob = torch.sigmoid(patient_logit)

        return vert_probs, patient_prob
