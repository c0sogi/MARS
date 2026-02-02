import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    save_submission,
    calculate_roc_auc,
    AverageMeter,
)
from library.dataset import get_dataloaders

# --- Helper Functions for Re-parameterization ---


def fuse_conv_bn(conv, bn):
    """
    Fuses a Conv2d and BatchNorm2d into a single Conv2d.
    """
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    # Reshape gamma/var for broadcasting
    # w shape: (out_channels, in_channels/groups, k, k)
    view_shape = (w.shape[0], 1, 1, 1)

    w_fused = w * (gamma / var_sqrt).view(view_shape)
    b_fused = (b - mean) * (gamma / var_sqrt) + beta

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    )

    fused_conv.weight.data = w_fused
    fused_conv.bias.data = b_fused

    return fused_conv


def fuse_bn_only(bn, channels, groups, kernel_size=3):
    """
    Fuses a BN layer (representing an Identity branch) into a Conv2d.
    The Conv2d represents an identity mapping.
    """
    input_dim = channels // groups
    w_id = torch.zeros(channels, input_dim, kernel_size, kernel_size)

    # Set 1s at the center for the corresponding input channel within the group
    for c in range(channels):
        idx = c % input_dim
        center = kernel_size // 2
        w_id[c, idx, center, center] = 1.0

    # Create dummy conv to hold weights for fusion
    dummy_conv = nn.Conv2d(
        channels,
        channels,
        kernel_size,
        padding=kernel_size // 2,
        groups=groups,
        bias=False,
    )
    dummy_conv.weight.data = w_id
    dummy_conv = dummy_conv.to(bn.weight.device)

    return fuse_conv_bn(dummy_conv, bn)


def pad_kernel_center(kernel, target_k=3):
    """
    Pads a smaller kernel (e.g. 1x1) to a larger size (e.g. 3x3) with zeros, keeping it centered.
    """
    current_k = kernel.shape[2]
    pad = (target_k - current_k) // 2
    return F.pad(kernel, (pad, pad, pad, pad))


# --- Model Components ---


class ECALayer(nn.Module):
    """
    Efficient Channel Attention Layer.
    Uses 1D convolution to model channel interactions without dimensionality reduction.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class RepNeXtUnit(nn.Module):
    """
    RepNeXt Block with optional ECA and Re-parameterization.
    Training: Multi-branch (3x3, 1x1, Identity).
    Inference: Single fused 3x3 convolution.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=32, use_eca=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.use_eca = use_eca
        self.deploy = False

        # Fallback for stem or mismatched dimensions where groups don't divide channels
        if in_channels % groups != 0 or out_channels % groups != 0:
            self.groups = 1

        padding = 1

        if not self.deploy:
            # Branch 3x3
            self.branch_3x3_conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            )
            self.branch_3x3_bn = nn.BatchNorm2d(out_channels)

            # Branch 1x1
            self.branch_1x1_conv = nn.Conv2d(
                in_channels,
                out_channels,
                1,
                stride=stride,
                padding=0,
                groups=self.groups,
                bias=False,
            )
            self.branch_1x1_bn = nn.BatchNorm2d(out_channels)

            # Branch Identity (only if dims match and stride is 1)
            if in_channels == out_channels and stride == 1:
                self.branch_identity_bn = nn.BatchNorm2d(out_channels)
            else:
                self.branch_identity_bn = None
        else:
            self.fused_conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=padding,
                groups=self.groups,
                bias=True,
            )

        if self.use_eca:
            self.eca = ECALayer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            out = self.fused_conv(x)
        else:
            out = self.branch_3x3_bn(self.branch_3x3_conv(x))
            out += self.branch_1x1_bn(self.branch_1x1_conv(x))
            if self.branch_identity_bn is not None:
                out += self.branch_identity_bn(x)

        if self.use_eca:
            out = self.eca(out)

        return self.relu(out)

    def switch_to_deploy(self):
        if self.deploy:
            return

        # Fuse 3x3
        fused_3x3 = fuse_conv_bn(self.branch_3x3_conv, self.branch_3x3_bn)
        w_3x3, b_3x3 = fused_3x3.weight, fused_3x3.bias

        # Fuse 1x1
        fused_1x1 = fuse_conv_bn(self.branch_1x1_conv, self.branch_1x1_bn)
        w_1x1, b_1x1 = fused_1x1.weight, fused_1x1.bias
        w_1x1 = pad_kernel_center(w_1x1, 3)  # Pad 1x1 to 3x3

        # Fuse Identity
        if self.branch_identity_bn is not None:
            fused_id = fuse_bn_only(
                self.branch_identity_bn, self.out_channels, self.groups, 3
            )
            w_id, b_id = fused_id.weight, fused_id.bias
        else:
            w_id, b_id = 0, 0

        # Combine
        final_w = w_3x3 + w_1x1 + w_id
        final_b = b_3x3 + b_1x1 + b_id

        # Create deployed conv
        self.fused_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.fused_conv.weight.data = final_w
        self.fused_conv.bias.data = final_b

        # Cleanup
        del self.branch_3x3_conv, self.branch_3x3_bn
        del self.branch_1x1_conv, self.branch_1x1_bn
        if hasattr(self, "branch_identity_bn"):
            del self.branch_identity_bn

        self.deploy = True


class RepDownsampleUnit(RepNeXtUnit):
    """
    Specialized RepNeXt Unit for downsampling (Stride > 1).
    """

    def __init__(self, in_channels, out_channels, groups=32, use_eca=True):
        super().__init__(
            in_channels, out_channels, stride=2, groups=groups, use_eca=use_eca
        )
        self.branch_identity_bn = None  # No identity path for downsampling


class UltraWideECARepNeXt(nn.Module):
    """
    Custom Ultra-Wide ECA-RepNeXt with Multi-Scale Aggregation.
    """

    def __init__(self):
        super().__init__()

        channels = Config.STAGES_CHANNELS  # [96, 192, 384]
        groups = Config.CARDINALITY
        use_eca = Config.USE_ECA

        # Stem: 3 -> 96 (32x32)
        # Using groups=1 for dense connection on input
        self.stem = RepNeXtUnit(3, channels[0], stride=1, groups=1, use_eca=use_eca)

        # Stage 1: 32x32, 96 channels
        self.stage1 = nn.Sequential(
            RepNeXtUnit(channels[0], channels[0], groups=groups, use_eca=use_eca),
            RepNeXtUnit(channels[0], channels[0], groups=groups, use_eca=use_eca),
        )

        # Stage 2: 16x16, 192 channels
        self.stage2_down = RepDownsampleUnit(
            channels[0], channels[1], groups=groups, use_eca=use_eca
        )
        self.stage2_blocks = nn.Sequential(
            RepNeXtUnit(channels[1], channels[1], groups=groups, use_eca=use_eca),
            RepNeXtUnit(channels[1], channels[1], groups=groups, use_eca=use_eca),
        )

        # Stage 3: 8x8, 384 channels
        self.stage3_down = RepDownsampleUnit(
            channels[1], channels[2], groups=groups, use_eca=use_eca
        )
        self.stage3_blocks = nn.Sequential(
            RepNeXtUnit(channels[2], channels[2], groups=groups, use_eca=use_eca),
            RepNeXtUnit(channels[2], channels[2], groups=groups, use_eca=use_eca),
        )

        # Multi-Scale Head: Concat GAP(Stage2) + GAP(Stage3)
        head_dim = channels[1] + channels[2]
        self.classifier = nn.Linear(head_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        # Stage 2
        x = self.stage2_down(x)
        x = self.stage2_blocks(x)
        feat_s2 = x  # 16x16

        # Stage 3
        x = self.stage3_down(x)
        x = self.stage3_blocks(x)
        feat_s3 = x  # 8x8

        # Aggregation
        pool_s2 = F.adaptive_avg_pool2d(feat_s2, 1).flatten(1)
        pool_s3 = F.adaptive_avg_pool2d(feat_s3, 1).flatten(1)

        concat = torch.cat([pool_s2, pool_s3], dim=1)

        return self.classifier(concat)

    def switch_to_deploy(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()


# --- Training and Inference Logic ---


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()
    scores = AverageMeter()

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()

        losses.update(loss.item(), images.size(0))
        try:
            auc = calculate_roc_auc(lbls, probs)
            scores.update(auc, images.size(0))
        except:
            pass

    return losses.avg, scores.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            lbls = labels.cpu().numpy()

            losses.update(loss.item(), images.size(0))
            all_probs.extend(probs)
            all_labels.extend(lbls)

    auc = calculate_roc_auc(all_labels, all_probs)
    return losses.avg, auc


def predict_tta(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # TTA: Original, HFlip, VFlip
            # 1. Original
            probs = torch.sigmoid(model(images))
            # 2. HFlip
            probs += torch.sigmoid(model(torch.flip(images, [3])))
            # 3. VFlip
            probs += torch.sigmoid(model(torch.flip(images, [2])))

            avg_probs = probs / 3.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    return all_ids, all_preds


def run_experiment():
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()

    test_preds_accumulator = None
    test_ids = None

    for seed in Config.SEEDS:
        print(f"\n--- Starting Seed {seed} ---")
        set_seed(seed)

        model = UltraWideECARepNeXt().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS
        )
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        patience = 0
        model_save_path = Config.get_model_save_path(seed)

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model.state_dict(), model_save_path)
                patience = 0
            else:
                patience += 1

            if patience >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Loading best model for seed {seed} (AUC: {best_auc:.6f})...")
        load_checkpoint(model, model_save_path, device)

        print("Switching to deploy mode (fusing weights)...")
        model.switch_to_deploy()

        ids, preds = predict_tta(model, test_loader, device)

        if test_preds_accumulator is None:
            test_preds_accumulator = np.array(preds)
            test_ids = ids
        else:
            test_preds_accumulator += np.array(preds)

    final_preds = test_preds_accumulator / len(Config.SEEDS)
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(test_ids, final_preds)
