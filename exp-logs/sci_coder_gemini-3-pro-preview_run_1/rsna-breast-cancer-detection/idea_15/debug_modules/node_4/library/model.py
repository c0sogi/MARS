import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DeformableAlignmentModule(nn.Module):
    """
    Predicts dense offsets between target and source feature maps and aligns the source
    to the target using bilinear interpolation (grid_sample).
    """

    def __init__(self, channels):
        super().__init__()
        # Predict offsets: input 2*channels -> output 2 (dx, dy)
        # We use a small bottleneck structure to keep parameter count low
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels // 2, 2, kernel_size=3, padding=1, bias=True
            ),  # Output: dx, dy
        )

        # Initialize the final convolution weights/bias to zero so training starts
        # with an identity transform (no warping).
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)

    def forward(self, target_feat, source_feat):
        """
        Args:
            target_feat: [B, C, H, W]
            source_feat: [B, C, H, W] (to be warped)
        Returns:
            aligned_source: [B, C, H, W]
        """
        B, C, H, W = target_feat.shape

        # 1. Predict Flow/Offsets
        combined = torch.cat([target_feat, source_feat], dim=1)
        offsets = self.offset_net(combined)  # [B, 2, H, W]

        # 2. Generate Base Grid
        # Create a meshgrid of pixel coordinates
        # yy: [H, W], xx: [H, W]
        yy, xx = torch.meshgrid(
            torch.arange(H, device=target_feat.device, dtype=torch.float32),
            torch.arange(W, device=target_feat.device, dtype=torch.float32),
            indexing="ij",
        )

        # Stack to [B, 2, H, W]
        base_grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # 3. Apply Offsets
        # sampling_grid = base + offsets
        sampling_grid = base_grid + offsets

        # 4. Normalize to [-1, 1] for grid_sample
        # x_norm = 2 * x / (W-1) - 1
        # y_norm = 2 * y / (H-1) - 1
        # We perform this in-place or via separate tensor operations
        # Note: We divide by max(dim-1, 1) to avoid div by zero if dim=1

        norm_grid = torch.empty_like(sampling_grid)
        norm_grid[:, 0, :, :] = 2.0 * sampling_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
        norm_grid[:, 1, :, :] = 2.0 * sampling_grid[:, 1, :, :] / max(H - 1, 1) - 1.0

        # Permute to [B, H, W, 2] required by grid_sample
        norm_grid = norm_grid.permute(0, 2, 3, 1)

        # 5. Resample
        # align_corners=True is standard for geometric consistency
        aligned_source = F.grid_sample(
            source_feat,
            norm_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return aligned_source


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Takes a list of feature maps from the backbone and returns a list of feature maps
    with the same channel dimension, enriched with top-down context.
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, inputs):
        """
        Args:
            inputs: List of feature maps [P3, P4, P5] (low res to high res or vice versa?
                    Usually backbone returns [stride 8, stride 16, stride 32])
        Returns:
            outputs: List of FPN features [P3_out, P4_out, P5_out]
        """
        # inputs are usually ordered by stride: [stride 8, stride 16, stride 32]
        # Top-down pathway starts from the last element (highest stride, lowest res)

        # 1. Lateral connections (1x1 convs)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # 2. Top-down path
        # Start from the top (last element)
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            # Upsample the higher level (i) and add to lower level (i-1)
            # laterals[i] is smaller spatial size than laterals[i-1]
            top_down = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + top_down

        # 3. FPN output convs (3x3) to reduce aliasing
        outputs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]

        return outputs


class SiameseFPNEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Load EfficientNet-B2, extracting features at strides 8, 16, 32
        # indices=(2, 3, 4) typically correspond to these strides in EfficientNet
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.IN_CHANNELS,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Get channel counts dynamically
        feature_info = self.backbone.feature_info
        in_channels_list = feature_info.channels()

        # 2. FPN
        self.fpn_channels = 128
        self.fpn = FPN(in_channels_list, self.fpn_channels)

        # 3. Alignment Modules
        # One aligner per FPN level
        self.aligners = nn.ModuleList(
            [
                DeformableAlignmentModule(self.fpn_channels)
                for _ in range(len(in_channels_list))
            ]
        )

        # 4. Classification Head
        # We concat (Target_GAP + Diff_GAP) for each level
        # Total features = Num_Levels * (FPN_Channels + FPN_Channels)
        num_levels = len(in_channels_list)
        total_features = num_levels * (self.fpn_channels * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE), nn.Linear(total_features, 1)
        )

    def forward_one_branch(self, x):
        """Extracts FPN features for a single image."""
        # Backbone features: [P3, P4, P5]
        features = self.backbone(x)
        # FPN features
        fpn_features = self.fpn(features)
        return fpn_features

    def forward(self, image, image_contra):
        # 1. Extract Features (Siamese)
        # Shared weights for backbone and FPN
        feats_target = self.forward_one_branch(image)
        feats_contra = self.forward_one_branch(image_contra)

        global_descriptors = []

        # 2. Process each level
        for i in range(len(feats_target)):
            f_t = feats_target[i]
            f_c = feats_contra[i]

            # A. Align Contralateral to Target
            f_c_aligned = self.aligners[i](f_t, f_c)

            # B. Compute Difference
            # This suppresses symmetric tissue and demographic embeddings
            diff = f_t - f_c_aligned

            # C. Global Average Pooling
            # We keep both the target context and the difference signal
            gap_t = F.adaptive_avg_pool2d(f_t, 1).flatten(1)
            gap_d = F.adaptive_avg_pool2d(diff, 1).flatten(1)

            global_descriptors.append(gap_t)
            global_descriptors.append(gap_d)

        # 3. Fusion
        # Concatenate all descriptors from all levels
        embedding = torch.cat(global_descriptors, dim=1)

        # 4. Classification
        logits = self.classifier(embedding)

        return logits
