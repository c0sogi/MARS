import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SpatialAlignmentModule(nn.Module):
    """
    Estimates a dense displacement field (flow) between two feature maps.
    Used to align the contralateral features to the target features.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Reduce channels and estimate flow
        # Input: Concatenation of Target and Contra features (C + C)
        self.conv1 = nn.Conv2d(
            in_channels * 2, 256, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU(inplace=True)

        # Output: 2 channels for (dx, dy) in normalized coordinates [-1, 1]
        self.conv3 = nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=True)

        # Initialize the final layer to zero so training starts with identity alignment
        nn.init.constant_(self.conv3.weight, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, target_feat, contra_feat):
        x = torch.cat([target_feat, contra_feat], dim=1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        flow = self.conv3(x)
        return flow


class FlowAlignedSiameseNet(nn.Module):
    """
    Flow-Aligned Pyramid Siamese Network using EfficientNet-B2 backbone.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # efficientnet_b2, pretrained, features_only to get pyramid
        # out_indices (2, 3, 4) corresponds to P3 (stride 8), P4 (stride 16), P5 (stride 32)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=3,
        )

        # Get channel counts for P3, P4, P5
        self.feature_channels = self.backbone.feature_info.channels()

        # 2. Spatial Alignment Module
        # We estimate flow at the coarsest level (P5)
        self.sam = SpatialAlignmentModule(self.feature_channels[-1])

        # 3. Classification Head
        # We concatenate GAP(Target) and GAP(Diff) for all 3 levels.
        # Total input dim = sum(channels * 2)
        total_features = sum(c * 2 for c in self.feature_channels)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def _warp_features(self, x, flow):
        """
        Warps feature map x using the provided flow field.

        Args:
            x: Feature map (B, C, H, W)
            flow: Flow field (B, 2, H_flow, W_flow)

        Returns:
            Warped feature map (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Upsample flow to match feature map resolution
        # flow is (B, 2, Hf, Wf) -> (B, 2, H, W)
        if flow.shape[2:] != (H, W):
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)

        # Generate base grid in [-1, 1]
        xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        # Stack to (B, H, W, 2)
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Permute flow for addition: (B, 2, H, W) -> (B, H, W, 2)
        flow_perm = flow.permute(0, 2, 3, 1)

        # Apply flow
        # The flow is learned as an offset in normalized coordinates
        sampling_grid = base_grid + flow_perm

        # Grid sample
        warped = F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped

    def forward(self, target, contra):
        """
        Args:
            target: (B, 3, H, W)
            contra: (B, 3, H, W)
        Returns:
            logits: (B, 1)
        """
        # 1. Extract Features
        # Returns list [P3, P4, P5]
        feats_t = self.backbone(target)
        feats_c = self.backbone(contra)

        # 2. Estimate Flow at P5 (Coarsest level)
        p5_t = feats_t[-1]
        p5_c = feats_c[-1]

        # Flow shape: (B, 2, H_p5, W_p5)
        flow = self.sam(p5_t, p5_c)

        # 3. Process Pyramid Levels
        embeddings = []

        for i in range(len(feats_t)):
            ft = feats_t[i]
            fc = feats_c[i]

            # Warp contralateral features to align with target
            fc_warped = self._warp_features(fc, flow)

            # Compute Difference (Symmetry-Difference)
            # The Age/Implant channels are constant spatially, so warping them doesn't
            # change values much (except at borders), so subtraction still cancels demographic bias.
            diff = ft - fc_warped

            # Global Average Pooling
            # (B, C, H, W) -> (B, C)
            gap_t = ft.mean(dim=(2, 3))
            gap_d = diff.mean(dim=(2, 3))

            embeddings.append(gap_t)
            embeddings.append(gap_d)

        # 4. Concatenate and Classify
        # (B, Total_Features)
        combined = torch.cat(embeddings, dim=1)

        logits = self.head(combined)

        return logits
