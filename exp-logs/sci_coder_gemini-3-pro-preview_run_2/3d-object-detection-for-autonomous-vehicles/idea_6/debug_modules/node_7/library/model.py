import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PillarFeatureNet(nn.Module):
    def __init__(self, num_input_features, num_filters, voxel_size, pc_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range

        # Input features: x, y, z, i, t (5)
        # Augmented features:
        # + (x - x_mean, y - y_mean, z - z_mean) -> 3
        # + (x - x_center, y - y_center, z - z_center) -> 3
        # Total = 5 + 3 + 3 = 11
        self.num_input = num_input_features + 6

        self.conv = nn.Linear(self.num_input, num_filters, bias=False)
        self.bn = nn.BatchNorm1d(num_filters)

    def forward(self, voxels, num_points, coords):
        # voxels: (P, N, 5)
        # num_points: (P,)
        # coords: (P, 4) -> (batch_idx, z_idx, y_idx, x_idx)

        device = voxels.device
        P, N, C = voxels.shape

        # 1. Feature Decoration
        # ---------------------

        # Create a mask for valid points: (P, N, 1)
        # arange(N) < num_points
        mask = (
            torch.arange(N, device=device).unsqueeze(0) < num_points.unsqueeze(1)
        ).unsqueeze(2)

        # Calculate Arithmetic Mean (Cluster Center)
        # Sum valid points
        points_sum = (voxels[..., :3] * mask).sum(dim=1, keepdim=True)
        # Divide by count (clamp to avoid div by 0)
        points_mean = points_sum / torch.clamp(
            num_points.view(-1, 1, 1).float(), min=1.0
        )

        # Offset from mean
        f_cluster = (voxels[..., :3] - points_mean) * mask

        # Calculate Geometric Center (Pillar Center)
        # x_c = min_x + x_idx * v_x + v_x/2
        x_idx = coords[:, 3].float().unsqueeze(1)
        y_idx = coords[:, 2].float().unsqueeze(1)
        z_idx = coords[:, 1].float().unsqueeze(1)

        center_x = (
            self.pc_range[0] + x_idx * self.voxel_size[0] + self.voxel_size[0] / 2.0
        )
        center_y = (
            self.pc_range[1] + y_idx * self.voxel_size[1] + self.voxel_size[1] / 2.0
        )
        center_z = (
            self.pc_range[2] + z_idx * self.voxel_size[2] + self.voxel_size[2] / 2.0
        )

        # Expand centers to (P, N, 3)
        centers = (
            torch.cat([center_x, center_y, center_z], dim=1)
            .unsqueeze(1)
            .repeat(1, N, 1)
        )

        # Offset from center
        f_center = (voxels[..., :3] - centers) * mask

        # Combine all features
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)  # (P, N, 11)

        # 2. Linear Encoding
        # ------------------
        # Flatten: (P*N, 11)
        x = features.view(-1, self.num_input)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Reshape back: (P, N, C_out)
        x = x.view(P, N, self.conv.out_features)

        # 3. Max Pooling
        # --------------
        # Mask invalid points again (ReLU output is >= 0, so 0 padding is safe floor)
        x = x * mask

        # Max over points dimension
        x_max = torch.max(x, dim=1)[0]  # (P, C_out)

        return x_max


class PointPillarsScatter(nn.Module):
    def __init__(self, num_features, grid_size):
        super().__init__()
        self.num_features = num_features
        # grid_size is [x, y, z]
        self.nx, self.ny, self.nz = grid_size

    def forward(self, pillar_features, coords, batch_size):
        # pillar_features: (P, C)
        # coords: (P, 4) -> (b, z, y, x)

        # Create empty canvas (B, C, H, W)
        canvas = torch.zeros(
            (batch_size, self.num_features, self.ny, self.nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        if pillar_features.shape[0] == 0:
            return canvas

        b_idx = coords[:, 0].long()
        y_idx = coords[:, 2].long()
        x_idx = coords[:, 3].long()

        # Scatter features to canvas
        # Note: In case of collisions (shouldn't happen with unique pillars), last write wins
        canvas[b_idx, :, y_idx, x_idx] = pillar_features

        return canvas


class ConvBlock(nn.Module):
    """Standard Conv-BN-ReLU block with optional Downsampling for ResNet-like structure"""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class UNetBackbone(nn.Module):
    def __init__(self, in_channels, layer_strides, layer_channels):
        super().__init__()

        # Encoder
        # C1: Stride 1
        self.c1 = ConvBlock(in_channels, layer_channels[0], stride=layer_strides[0])
        # C2: Stride 2
        self.c2 = ConvBlock(
            layer_channels[0], layer_channels[1], stride=layer_strides[1]
        )
        # C3: Stride 2
        self.c3 = ConvBlock(
            layer_channels[1], layer_channels[2], stride=layer_strides[2]
        )

        # Decoder
        # Up3: Upsample C3 to C2 size
        self.up3 = nn.ConvTranspose2d(
            layer_channels[2], layer_channels[1], kernel_size=2, stride=2
        )
        self.conv3 = ConvBlock(layer_channels[1] * 2, layer_channels[1])

        # Up2: Upsample to C1 size
        self.up2 = nn.ConvTranspose2d(
            layer_channels[1], layer_channels[0], kernel_size=2, stride=2
        )
        self.conv2 = ConvBlock(layer_channels[0] * 2, layer_channels[0])

    def forward(self, x):
        # Encoder
        x1 = self.c1(x)  # (B, C1, H, W)
        x2 = self.c2(x1)  # (B, C2, H/2, W/2)
        x3 = self.c3(x2)  # (B, C3, H/4, W/4)

        # Decoder
        u3 = self.up3(x3)  # (B, C2, H/2, W/2)
        c3 = torch.cat([u3, x2], dim=1)
        o3 = self.conv3(c3)

        u2 = self.up2(o3)  # (B, C1, H, W)
        c2 = torch.cat([u2, x1], dim=1)
        o2 = self.conv2(c2)

        return o2


class CenterHead(nn.Module):
    def __init__(self, in_channels, num_classes, head_conv=64):
        super().__init__()

        # Heatmap Head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, num_classes, 1, bias=True),
        )

        # Regression Head
        # Targets: [off_x, off_y, z, log_w, log_l, log_h, sin, cos] -> 8 channels
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(),
            nn.Conv2d(head_conv, 8, 1, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize heatmap bias to -2.19 (prob ~ 0.1) to prevent instability at start
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        hm = self.heatmap_head(x)
        reg = self.reg_head(x)
        return hm, reg


class PillarUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config

        # 1. Pillar Feature Net
        self.pfn = PillarFeatureNet(
            num_input_features=self.config.NUM_POINT_FEATURES,
            num_filters=self.config.PFN_OUT_CHANNELS,
            voxel_size=self.config.VOXEL_SIZE,
            pc_range=self.config.POINT_CLOUD_RANGE,
        )

        # 2. Scatter to BEV
        self.scatter = PointPillarsScatter(
            num_features=self.config.PFN_OUT_CHANNELS, grid_size=self.config.GRID_SIZE
        )

        # 3. U-Net Backbone
        self.backbone = UNetBackbone(
            in_channels=self.config.PFN_OUT_CHANNELS,
            layer_strides=self.config.BACKBONE_LAYER_STRIDES,
            layer_channels=self.config.BACKBONE_LAYER_CHANNELS,
        )

        # 4. Center Head
        self.head = CenterHead(
            in_channels=self.config.BACKBONE_LAYER_CHANNELS[0],
            num_classes=self.config.NUM_CLASSES,
            head_conv=self.config.HEAD_HIDDEN_CHANNELS,
        )

    def forward(self, batch_dict):
        voxels = batch_dict["voxels"]
        num_points = batch_dict["num_points"]
        coords = batch_dict["coordinates"]
        batch_size = batch_dict["batch_size"]

        # 1. Extract Pillar Features
        pillar_feats = self.pfn(voxels, num_points, coords)

        # 2. Scatter to Grid
        bev_map = self.scatter(pillar_feats, coords, batch_size)

        # 3. Backbone
        feature_map = self.backbone(bev_map)

        # 4. Head
        hm, reg = self.head(feature_map)

        preds = {"hm": hm, "reg": reg}

        # 5. Loss Calculation (if training/validation with targets)
        if "hm" in batch_dict:
            loss, loss_stats = self.compute_loss(preds, batch_dict)
            return preds, loss, loss_stats

        return preds

    def compute_loss(self, preds, targets):
        pred_hm = preds["hm"]
        pred_reg = preds["reg"]

        gt_hm = targets["hm"].to(pred_hm.device)
        gt_reg = targets["target_reg"].to(pred_reg.device)
        gt_ind = targets["ind"].to(pred_reg.device)
        gt_mask = targets["mask"].to(pred_reg.device)

        # 1. Heatmap Loss (Penalty Reduced Focal Loss)
        pred_hm = torch.sigmoid(pred_hm)
        hm_loss = self.focal_loss(pred_hm, gt_hm)

        # 2. Regression Loss (L1)
        reg_loss = self.reg_l1_loss(pred_reg, gt_reg, gt_ind, gt_mask)

        # Total Loss
        loss = hm_loss + reg_loss

        stats = {
            "hm_loss": hm_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, stats

    def focal_loss(self, pred, gt):
        """
        Modified focal loss.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def reg_l1_loss(self, pred, target, ind, mask):
        """
        L1 loss gathered at ground truth indices.
        """
        # Transpose and gather features at specific indices
        pred = self._transpose_and_gather_feat(pred, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()

        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
