import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.dbb_layers import DBBGroupedConv, transI_fusebn
from library.utils import set_seed, get_device, calculate_roc_auc, AverageMeter
from library.dataset import get_dataloaders

# --- Model Components ---


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DualPathDownsample(nn.Module):
    """
    Downsampling block that sums a Stride-2 3x3 Conv and a Stride-2 1x1 Conv.
    Fused into a single 3x3 Conv during inference.
    """

    def __init__(self, in_channels, out_channels):
        super(DualPathDownsample, self).__init__()
        self.deploy = False
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Path 1: 3x3 Conv, stride 2
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        # Path 2: 1x1 Conv, stride 2
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def switch_to_deploy(self):
        if self.deploy:
            return

        # Fuse 3x3 branch
        k3, b3 = transI_fusebn(self.conv3x3.weight, self.conv3x3.bias, self.bn3x3)

        # Fuse 1x1 branch
        k1, b1 = transI_fusebn(self.conv1x1.weight, self.conv1x1.bias, self.bn1x1)

        # Pad 1x1 kernel to 3x3 (center aligned) to match 3x3 kernel shape
        # k1 is (Out, In, 1, 1). Pad to (Out, In, 3, 3) -> (1, 1, 1, 1) padding
        k1_pad = F.pad(k1, (1, 1, 1, 1))

        # Sum kernels and biases
        k_eq = k3 + k1_pad
        b_eq = b3 + b1

        # Create new single layer
        self.fused_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.fused_conv.weight.data = k_eq
        self.fused_conv.bias.data = b_eq

        # Delete training branches
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        return self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))


class DBBResBlock(nn.Module):
    """
    Residual Block using DBBGroupedConv and SEBlock.
    Structure: Input -> DBB(3x3) -> ReLU -> SE -> Add(Input) -> ReLU
    """

    def __init__(self, channels, groups=32):
        super(DBBResBlock, self).__init__()
        # DBB Grouped Conv (3x3)
        self.dbb = DBBGroupedConv(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(channels)

    def switch_to_deploy(self):
        self.dbb.switch_to_deploy()

    def forward(self, x):
        identity = x
        out = self.dbb(x)
        out = self.relu(out)
        out = self.se(out)
        out += identity
        return F.relu(out)


class UltraWideDBBResNeXt(nn.Module):
    """
    Custom Ultra-Wide DBB-SE-ResNeXt Architecture.
    Features:
    - Ultra-Wide channels [96, 192, 384]
    - DBB Grouped Convolutions
    - Dual-Path Downsampling
    - Multi-Scale Aggregation Head
    """

    def __init__(self, num_classes=1, groups=32):
        super(UltraWideDBBResNeXt, self).__init__()

        # Stem: 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 96 channels, 32x32
        self.stage1 = nn.Sequential(
            DBBResBlock(96, groups=groups), DBBResBlock(96, groups=groups)
        )

        # Downsample 1: 96 -> 192 (16x16)
        self.down1 = DualPathDownsample(96, 192)

        # Stage 2: 192 channels, 16x16
        self.stage2 = nn.Sequential(
            DBBResBlock(192, groups=groups), DBBResBlock(192, groups=groups)
        )

        # Downsample 2: 192 -> 384 (8x8)
        self.down2 = DualPathDownsample(192, 384)

        # Stage 3: 384 channels, 8x8
        self.stage3 = nn.Sequential(
            DBBResBlock(384, groups=groups), DBBResBlock(384, groups=groups)
        )

        # Head: Multi-Scale Aggregation
        # Concat GAP(Stage2) + GAP(Stage3) -> 192 + 384 = 576
        self.classifier = nn.Linear(576, num_classes)

    def switch_to_deploy(self):
        """
        Recursively switches all re-parameterizable blocks to inference mode.
        """
        for m in self.modules():
            if isinstance(m, (DBBGroupedConv, DualPathDownsample, DBBResBlock)):
                m.switch_to_deploy()

    def forward(self, x):
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)

        # Down 1
        x = self.down1(x)

        # Stage 2
        x = self.stage2(x)
        feat2 = x

        # Down 2
        x = self.down2(x)

        # Stage 3
        x = self.stage3(x)
        feat3 = x

        # Multi-Scale Aggregation
        gap2 = F.adaptive_avg_pool2d(feat2, (1, 1)).flatten(1)
        gap3 = F.adaptive_avg_pool2d(feat3, (1, 1)).flatten(1)

        combined = torch.cat([gap2, gap3], dim=1)
        out = self.classifier(combined)

        return out


# --- Training & Inference Utilities ---


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).float()

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

            loss_meter.update(loss.item(), images.size(0))

    auc = calculate_roc_auc(np.array(targets_list), np.array(preds))
    return loss_meter.avg, auc


def predict_with_tta(model, loader, device):
    """
    Predicts using Test Time Augmentation (Original, H-Flip, V-Flip).
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for images, _, batch_ids in loader:
            images = images.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(images).squeeze(1))

            # 2. Horizontal Flip
            out2 = torch.sigmoid(model(torch.flip(images, [3])).squeeze(1))

            # 3. Vertical Flip
            out3 = torch.sigmoid(model(torch.flip(images, [2])).squeeze(1))

            # Average
            avg_pred = (out1 + out2 + out3) / 3.0

            preds.extend(avg_pred.cpu().numpy())
            ids.extend(batch_ids)

    return ids, preds


def run_training_pipeline(epochs=20, batch_size=64, seeds=[0, 1, 2, 3, 4]):
    """
    Main pipeline: Trains 5 models (Homogeneous Seed Averaging), performs TTA,
    aggregates predictions, and saves submission.
    """
    device = get_device()

    # Setup directories
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working/idea_39", exist_ok=True)

    all_preds = []
    test_ids = None

    for seed in seeds:
        print(f"Training Seed {seed}...")
        set_seed(seed)

        # Get dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size, seed=seed
        )

        # Initialize Model
        model = UltraWideDBBResNeXt(groups=32).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        best_model_state = None

        # Training Loop
        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()

        print(f"Seed {seed} Best Val AUC: {best_auc:.10f}")

        # Load best model for inference
        model.load_state_dict(best_model_state)

        # Switch to efficient inference mode
        model.switch_to_deploy()

        # Predict on Test Set with TTA
        ids, preds = predict_with_tta(model, test_loader, device)

        if test_ids is None:
            test_ids = ids

        all_preds.append(preds)

    # Aggregate predictions (Arithmetic Mean)
    final_preds = np.mean(all_preds, axis=0)

    # Save Submission
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
