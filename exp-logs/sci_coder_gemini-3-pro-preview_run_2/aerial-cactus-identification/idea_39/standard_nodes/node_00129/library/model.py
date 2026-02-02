import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.utils import set_seed, get_device, calculate_roc_auc, AverageMeter
from library.dataset import get_dataloaders

# --- Model Components ---


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    Cite solution_lesson_node_00063: Prefer 1x1 convolutions for projection shortcuts.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MultiScaleResNet(nn.Module):
    """
    Multi-Scale ResNet with channels [32, 64, 128].
    Cite solution_lesson_node_00016: Efficiency via Multi-Scale Feature Aggregation.
    Cite solution_lesson_node_00049: Prioritize Model Width (32 start) over complex heads.
    """

    def __init__(self, num_classes=1):
        super(MultiScaleResNet, self).__init__()

        # Stem: 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 32 channels, 32x32
        self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))

        # Stage 2: 64 channels, 16x16
        self.layer2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))

        # Stage 3: 128 channels, 8x8
        self.layer3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))

        # Head: Multi-Scale Aggregation (Stage 2 + Stage 3)
        # GAP(Stage 2) = 64, GAP(Stage 3) = 128 -> Total 192
        self.classifier = nn.Linear(64 + 128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)

        x = self.layer2(x)
        feat2 = x

        x = self.layer3(x)
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

    for images, targets, _ in loader:
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
        for images, targets, _ in loader:
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
        model = MultiScaleResNet().to(device)
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
