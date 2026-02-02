import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.data import get_loaders

# ==========================================
# Architectures
# ==========================================


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class Res2NeXtBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, scales=4, groups=32):
        super(Res2NeXtBlock, self).__init__()

        self.scales = scales
        self.stride = stride

        # Determine width for splits
        # We use expansion=1 for the block to maintain "Wide" characteristics without explosion
        # planes is the output channels
        width = planes
        self.width = width

        # Calculate split width
        self.split_width = width // scales

        # Adjust groups if split_width is too small or incompatible
        # Ensure groups divides split_width
        # We want groups to be as close to 'groups' arg as possible (Cardinality=32)
        real_groups = groups
        if self.split_width % real_groups != 0:
            # Find largest divisor of split_width <= groups using GCD
            # This ensures valid grouped convolution
            real_groups = math.gcd(self.split_width, groups)

        self.groups = real_groups

        # 1x1 Projection
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 Cascaded Convs
        # We need (scales - 1) convs
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                nn.Conv2d(
                    self.split_width,
                    self.split_width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.groups,
                    bias=False,
                )
            )
            self.bns.append(nn.BatchNorm2d(self.split_width))

        # Pooling for the first split if stride > 1
        self.pool = (
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            if stride > 1
            else nn.Identity()
        )

        # 1x1 Expansion (Output)
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # Coordinate Attention
        self.ca = CoordinateAttention(planes)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut
        self.downsample = None
        if stride != 1 or in_planes != planes:
            # Strictly use 1x1 convs for projection
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Split
        spx = torch.split(out, self.split_width, 1)

        # Cascade
        sp = spx[0]
        sp = self.pool(sp)

        out_splits = [sp]

        for i in range(1, self.scales):
            # If stride > 1, we disable the hierarchical addition to avoid shape mismatch
            # or complex pooling logic, relying on the parallel 3x3 convs.
            # Standard Res2Net behavior for stride=1 is addition.
            if self.stride == 1:
                sp = sp + spx[i]
            else:
                sp = spx[i]

            sp = self.convs[i - 1](sp)
            sp = self.bns[i - 1](sp)
            sp = self.relu(sp)
            out_splits.append(sp)

        out = torch.cat(out_splits, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class CactusNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CactusNet, self).__init__()

        # Initial Conv
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stages: [64, 128, 256]
        # Stage 1: 32x32 -> 32x32 (Stride 1)
        self.layer1 = self._make_layer(64, stride=1)
        # Stage 2: 32x32 -> 16x16 (Stride 2)
        self.layer2 = self._make_layer(128, stride=2)
        # Stage 3: 16x16 -> 8x8 (Stride 2)
        self.layer3 = self._make_layer(256, stride=2)

        # Head: Multi-Scale Aggregation (Stage 2 + Stage 3)
        # Stage 2 out: 128 channels
        # Stage 3 out: 256 channels
        self.fc = nn.Linear(128 + 256, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, stride):
        layers = []
        # Block 1: Handles stride and channel change
        layers.append(
            Res2NeXtBlock(self.in_planes, planes, stride=stride, scales=4, groups=32)
        )
        self.in_planes = planes
        # Block 2: Refinement
        layers.append(
            Res2NeXtBlock(self.in_planes, planes, stride=1, scales=4, groups=32)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # Stage 1
        x2 = self.layer2(x1)  # Stage 2
        x3 = self.layer3(x2)  # Stage 3

        # Multi-Scale Aggregation
        # GAP on x2 (16x16)
        f2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        # GAP on x3 (8x8)
        f3 = F.adaptive_avg_pool2d(x3, (1, 1)).view(x3.size(0), -1)

        # Concatenate
        feat = torch.cat([f2, f3], dim=1)
        out = self.fc(feat)
        return out


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_targets.append(labels.detach().cpu().numpy())
            all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def train_model():
    """
    Main training routine.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Loop over seeds (Homogeneous Seed Averaging)
    for seed in Config.SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        model = CactusNet(num_classes=Config.NUM_CLASSES).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
        )

        best_auc = 0.0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "auc": best_auc,
                    },
                    f"model_seed_{seed}.pth",
                )

        print(f"Best AUC for Seed {seed}: {best_auc:.6f}")

    # Generate Submission
    generate_submission(test_loader, device)


def generate_submission(test_loader, device):
    print("\n--- Generating Submission with TTA ---")

    # Prepare to store predictions from all seeds
    all_seed_preds = []
    ids = test_loader.dataset.ids

    for seed in Config.SEEDS:
        model = CactusNet(num_classes=Config.NUM_CLASSES).to(device)
        try:
            load_checkpoint(f"model_seed_{seed}.pth", model, device=device)
        except FileNotFoundError:
            print(f"Warning: Checkpoint for seed {seed} not found. Skipping.")
            continue

        model.eval()

        seed_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # TTA: Original, H-Flip, V-Flip

                # 1. Original
                out1 = torch.sigmoid(model(images))

                # 2. H-Flip
                out2 = torch.sigmoid(model(torch.flip(images, [3])))

                # 3. V-Flip
                out3 = torch.sigmoid(model(torch.flip(images, [2])))

                # Average TTA
                avg_out = (out1 + out2 + out3) / 3.0
                seed_preds.append(avg_out.cpu().numpy())

        seed_preds = np.concatenate(seed_preds).flatten()
        all_seed_preds.append(seed_preds)

    if not all_seed_preds:
        print("Error: No models available for submission.")
        return

    # Average across seeds
    final_preds = np.mean(all_seed_preds, axis=0)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "has_cactus": final_preds})

    # Save
    df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
