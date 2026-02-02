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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CactusNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CactusNet, self).__init__()
        self.in_planes = 32

        # Initial Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Stages: [32, 64, 128] - Right-sized for 32x32 input (Cite Lesson 19)
        # Stage 1: 32x32 -> 32x32 (Stride 1)
        self.layer1 = self._make_layer(32, 2, stride=1)
        # Stage 2: 32x32 -> 16x16 (Stride 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        # Stage 3: 16x16 -> 8x8 (Stride 2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        # Head: Multi-Scale Aggregation (Stage 2 + Stage 3) (Cite Lesson 16)
        # Stage 2 out: 64 channels
        # Stage 3 out: 128 channels
        self.fc = nn.Linear(64 + 128, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        # Stage 1
        out = self.layer1(out)

        # Stage 2
        out2 = self.layer2(out)

        # Stage 3
        out3 = self.layer3(out2)

        # Multi-Scale Aggregation
        # GAP on out2 (16x16)
        f2 = F.adaptive_avg_pool2d(out2, (1, 1)).view(out2.size(0), -1)
        # GAP on out3 (8x8)
        f3 = F.adaptive_avg_pool2d(out3, (1, 1)).view(out3.size(0), -1)

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
