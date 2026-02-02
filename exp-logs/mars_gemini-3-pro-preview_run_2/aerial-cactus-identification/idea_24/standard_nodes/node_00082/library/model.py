import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from library import config, dataset, utils

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """

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


class SEBasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock with Squeeze-and-Excitation.
    """

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideSEResNet(nn.Module):
    """
    Wide SE-ResNet Architecture with Multi-Scale Aggregation Head.
    """

    def __init__(
        self,
        num_classes=1,
        stages=[64, 128, 256],
        se_reduction=16,
        use_gap=True,
        dropout_rate=0.0,
    ):
        super(WideSEResNet, self).__init__()

        self.in_planes = 64
        self.use_gap = use_gap

        # Initial Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stages (3 blocks per stage)
        # Stage 1: 32x32
        self.layer1 = self._make_layer(stages[0], 3, stride=1, reduction=se_reduction)
        # Stage 2: 16x16
        self.layer2 = self._make_layer(stages[1], 3, stride=2, reduction=se_reduction)
        # Stage 3: 8x8
        self.layer3 = self._make_layer(stages[2], 3, stride=2, reduction=se_reduction)

        # Head: Multi-Scale Aggregation
        # Aggregates features from Stage 2 (16x16) and Stage 3 (8x8)
        self.final_dim = stages[1] + stages[2]

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.final_dim, num_classes)

        # Weights Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, reduction):
        layers = []
        layers.append(SEBasicBlock(self.in_planes, planes, stride, reduction))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.in_planes, planes, 1, reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)  # 32x32

        feat2 = self.layer2(x)  # 16x16
        feat3 = self.layer3(feat2)  # 8x8

        # Multi-Scale Aggregation
        # Global Average Pooling on both stages
        gap2 = F.adaptive_avg_pool2d(feat2, 1).view(feat2.size(0), -1)
        gap3 = F.adaptive_avg_pool2d(feat3, 1).view(feat3.size(0), -1)

        combined = torch.cat([gap2, gap3], dim=1)
        combined = self.dropout(combined)
        out = self.fc(combined)
        return out


# =============================================================================
# TRAINING & EXECUTION
# =============================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    try:
        auc = utils.calculate_roc_auc(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    try:
        auc = utils.calculate_roc_auc(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def predict_tta(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # TTA: Original, HFlip, VFlip
            # Original
            out1 = model(images)
            prob1 = torch.sigmoid(out1)

            # HFlip
            img_h = torch.flip(images, [3])
            out2 = model(img_h)
            prob2 = torch.sigmoid(out2)

            # VFlip
            img_v = torch.flip(images, [2])
            out3 = model(img_v)
            prob3 = torch.sigmoid(out3)

            # Average
            avg_prob = (prob1 + prob2 + prob3) / 3.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds), all_ids


def run_training():
    # Setup
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data Loading (using library.dataset with caching)
    train_dataset = dataset.CactusDataset(
        config.TRAIN_METADATA_PATH,
        phase="train",
        transform=dataset.get_transforms("train"),
        load_cached_data=True,
    )
    val_dataset = dataset.CactusDataset(
        config.VAL_METADATA_PATH,
        phase="val",
        transform=dataset.get_transforms("val"),
        load_cached_data=True,
    )
    test_dataset = dataset.CactusDataset(
        config.TEST_METADATA_PATH,
        phase="test",
        transform=dataset.get_transforms("test"),
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Accumulator for test predictions
    final_test_preds = np.zeros((len(test_dataset), 1))
    test_ids = None

    # Loop over seeds (Homogeneous Seed Averaging)
    for seed in config.SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        utils.set_seed(seed)

        # Init Model
        model = WideSEResNet(
            num_classes=config.NUM_CLASSES,
            stages=config.MODEL_PARAMS["stages"],
            se_reduction=config.MODEL_PARAMS["se_reduction"],
            use_gap=config.MODEL_PARAMS["use_gap"],
            dropout_rate=config.MODEL_PARAMS["dropout_rate"],
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc}"
            )

            # Checkpoint Logic
            is_best = val_auc > best_auc + config.MIN_DELTA
            if val_auc > best_auc:
                best_auc = val_auc

            if is_best:
                patience_counter = 0
                utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_auc": best_auc,
                    },
                    is_best=True,
                    checkpoint_dir=config.CHECKPOINT_DIR,
                    filename=f"model_seed_{seed}.pth",
                )
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model for this seed
        best_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
        utils.load_checkpoint(best_path, model, device=device)

        # Predict on Test with TTA
        preds, ids = predict_tta(model, test_loader, device)
        final_test_preds += preds
        test_ids = ids

    # Average predictions across all seeds
    final_test_preds /= len(config.SEEDS)

    # Save Submission
    submission_df = pd.DataFrame(
        {"id": test_ids, "has_cactus": final_test_preds.flatten()}
    )
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
