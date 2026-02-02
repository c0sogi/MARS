import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import (
    MODEL_CHANNELS,
    SEEDS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
    NUM_CLASSES,
    TTA_ENABLED,
)
from library.utils import set_seed, calculate_roc_auc, AverageMeter, EarlyStopping
from library.dataset import get_dataloaders


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block (Cite solution_lesson_node_00011).
    Removed SE blocks to reduce overhead (Cite solution_lesson_node_00013).
    """

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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class CactusNet(nn.Module):
    """
    Standard ResNet with Multi-Scale Aggregation Head (Cite solution_lesson_node_00016).
    """

    def __init__(self, num_classes=1):
        super(CactusNet, self).__init__()
        self.in_planes = MODEL_CHANNELS[0]

        # Stem: 32x32 input
        self.conv1 = nn.Conv2d(
            3, MODEL_CHANNELS[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(MODEL_CHANNELS[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(MODEL_CHANNELS[0], stride=1)

        # Stage 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(MODEL_CHANNELS[1], stride=2)

        # Stage 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(MODEL_CHANNELS[2], stride=2)

        # Multi-Scale Aggregation Head
        # Concatenating GAP of Stage 2 (64 channels) and Stage 3 (128 channels)
        self.fc = nn.Linear(MODEL_CHANNELS[1] + MODEL_CHANNELS[2], num_classes)

    def _make_layer(self, planes, stride):
        layers = []
        # First block handles stride/channel change
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        # Additional blocks (using 2 blocks per stage for depth)
        layers.append(BasicBlock(self.in_planes, planes, stride=1))
        layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8

        # Global Average Pooling on Stage 2 and Stage 3
        out2 = torch.mean(x2, dim=[2, 3])
        out3 = torch.mean(x3, dim=[2, 3])

        # Multi-Scale Concatenation
        out = torch.cat([out2, out3], dim=1)

        return self.fc(out)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = calculate_roc_auc(all_labels, all_preds)

    return losses.avg, auc


def run_training_pipeline():
    """
    Executes the full training pipeline:
    1. Loops through defined SEEDS.
    2. Trains a model for each seed with Early Stopping.
    3. Saves the best model for each seed.
    4. Generates predictions using TTA and Ensembling.
    5. Saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Get DataLoaders (cached)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    model_paths = []

    # --- Training Phase ---
    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        model = CactusNet(num_classes=NUM_CLASSES).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model_save_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        early_stopping = EarlyStopping(
            patience=PATIENCE, verbose=True, path=model_save_path
        )

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
            )

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        model_paths.append(model_save_path)

    # --- Inference Phase ---
    print("\n--- Generating Submission with TTA and Ensembling ---")

    # Load all models
    models = []
    for path in model_paths:
        m = CactusNet(num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    submission_data = []

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)
            batch_preds = []

            # Test Time Augmentation (Original, H-Flip, V-Flip)
            inputs = [images]
            if TTA_ENABLED:
                inputs.append(torch.flip(images, [3]))  # Horizontal
                inputs.append(torch.flip(images, [2]))  # Vertical

            # Aggregate predictions across views and models
            for x in inputs:
                for model in models:
                    logits = model(x)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

            # Average all predictions
            # Shape: (num_views * num_models, batch_size, 1)
            batch_preds = np.array(batch_preds)
            avg_preds = np.mean(batch_preds, axis=0).flatten()

            for img_id, pred in zip(ids, avg_preds):
                submission_data.append({"id": img_id, "has_cactus": pred})

    # Save Submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
