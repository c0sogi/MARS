import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.dataset import get_dataloaders
from library.utils import seed_everything, get_device, calculate_roc_auc

# --- Configuration ---
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working/idea_16"

# Ensure directories exist
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)


# --- Model Architecture ---


class WideResNetBlock(nn.Module):
    """
    Basic Residual Block with 3x3 convolutions.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(WideResNetBlock, self).__init__()
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


class WideResNetMultiScale(nn.Module):
    """
    Custom Wide ResNet with Multi-Scale Feature Aggregation.
    Backbone: [32, 64, 128] channels.
    Head: Concatenates GAP features from Stage 2 (16x16) and Stage 3 (8x8).
    """

    def __init__(self):
        super(WideResNetMultiScale, self).__init__()

        # Initial Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        # Stage 1: 32 -> 32, 32x32
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        # Stage 2: 32 -> 64, 16x16
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        # Stage 3: 64 -> 128, 8x8
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)

        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: 64 (from Stage 2) + 128 (from Stage 3) = 192
        self.fc = nn.Linear(64 + 128, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(WideResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(WideResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)  # 32x32, 32ch
        x2 = self.layer2(x1)  # 16x16, 64ch
        x3 = self.layer3(x2)  # 8x8, 128ch

        # Multi-Scale Aggregation
        # Global Average Pooling on Stage 2 and Stage 3
        f2 = self.avgpool(x2).flatten(1)  # Shape: (B, 64)
        f3 = self.avgpool(x3).flatten(1)  # Shape: (B, 128)

        # Concatenate
        f_cat = torch.cat([f2, f3], dim=1)  # Shape: (B, 192)

        # Classification
        out = self.fc(f_cat)
        return out


# --- Training & Inference Functions ---


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
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
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
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
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def predict_with_tta(model, loader, device):
    """
    Predicts using Test Time Augmentation: Original, H-Flip, V-Flip.
    Returns averaged probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # TTA 1: Original
            out1 = torch.sigmoid(model(images))

            # TTA 2: Horizontal Flip
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h))

            # TTA 3: Vertical Flip
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v))

            # Average
            avg_preds = (out1 + out2 + out3) / 3.0
            all_preds.append(avg_preds.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training_and_submission():
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    dataloaders = get_dataloaders(batch_size=BATCH_SIZE)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    test_ids = dataloaders["test_ids"]

    # Store predictions from all seeds
    ensemble_preds = np.zeros((len(test_ids), N_FOLDS))

    for seed in range(N_FOLDS):
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        model = WideResNetMultiScale().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        patience = 5
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.6f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model for this seed
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Inference with TTA
        print(f"Generating predictions for Seed {seed}...")
        seed_preds = predict_with_tta(model, test_loader, device)
        ensemble_preds[:, seed] = seed_preds

    # Average predictions across seeds
    final_preds = np.mean(ensemble_preds, axis=1)

    # Create submission
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")


# Execute the pipeline
run_training_and_submission()
