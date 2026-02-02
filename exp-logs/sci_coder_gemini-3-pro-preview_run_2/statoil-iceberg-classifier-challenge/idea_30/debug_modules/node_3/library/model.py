import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

# Import from library
from library.config import Config
from library.utils import seed_everything, EarlyStopping
from library.data_loader import load_data, IcebergDataset


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Mixed Pooling: Max and Avg (No Min)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DualStreamPooling(nn.Module):
    """
    Performs Max Pooling (Peaks) and Min Pooling (Shadows) and concatenates them.
    Expands channels by 2x.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualStreamPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Max Pooling
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride)
        # Min Pooling: -Max(-x)
        min_pool = -F.max_pool2d(-x, self.kernel_size, self.stride)
        # Concatenate along channel dimension
        return torch.cat([max_pool, min_pool], dim=1)


class SC_WBN(nn.Module):
    """
    Spatially-Contextualized Wide-Body Network.
    Features:
    - Wide-Body Delayed-Integration Backbone (128 filters)
    - CBAM (Mixed Pooling)
    - DualStreamPooling (Max+Min)
    - Spatial-Context Bottleneck Readout
    - Dedicated Metadata MLP
    """

    def __init__(self):
        super(SC_WBN, self).__init__()

        # --- Visual Branch ---
        # Stage 1
        self.conv1 = nn.Conv2d(
            Config.IMG_CHANNELS, Config.BACKBONE_FILTERS, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(Config.BACKBONE_FILTERS)
        self.cbam1 = CBAM(Config.BACKBONE_FILTERS)
        self.pool1 = DualStreamPooling()  # Out: 128*2 = 256

        # Stage 2
        self.conv2 = nn.Conv2d(256, Config.BACKBONE_FILTERS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(Config.BACKBONE_FILTERS)
        self.cbam2 = CBAM(Config.BACKBONE_FILTERS)
        self.pool2 = DualStreamPooling()  # Out: 256

        # Stage 3
        self.conv3 = nn.Conv2d(256, Config.BACKBONE_FILTERS, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(Config.BACKBONE_FILTERS)
        self.cbam3 = CBAM(Config.BACKBONE_FILTERS)
        self.pool3 = DualStreamPooling()  # Out: 256

        # Stage 4
        self.conv4 = nn.Conv2d(256, Config.BACKBONE_FILTERS, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(Config.BACKBONE_FILTERS)
        self.cbam4 = CBAM(Config.BACKBONE_FILTERS)
        self.pool4 = DualStreamPooling()  # Out: 256

        # Spatial-Context Bottleneck Readout
        # Input: 256 channels, 4x4 spatial
        # Compress to 64 channels, maintain 4x4 spatial
        self.sc_bottleneck = nn.Conv2d(
            256, Config.READOUT_CHANNELS, kernel_size=3, padding=1, stride=1
        )
        self.sc_bn = nn.BatchNorm2d(Config.READOUT_CHANNELS)

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # --- Fusion Head ---
        # Visual: 4x4x64 = 1024
        # Meta: 32
        # Total: 1056
        fusion_dim = Config.DENSE_INPUT_DIM + 32

        self.fc1 = nn.Linear(fusion_dim, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATE)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(Config.DROPOUT_RATE)

        self.output = nn.Linear(256, 1)

    def forward(self, x_img, x_inc):
        # --- Visual Branch ---
        # Block 1
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.cbam1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        # Spatial-Context Bottleneck
        x = F.relu(self.sc_bn(self.sc_bottleneck(x)))

        # Flatten
        x_vis = x.view(x.size(0), -1)  # (B, 1024)

        # --- Metadata Branch ---
        x_inc = x_inc.view(-1, 1)  # Ensure (B, 1)
        x_meta = self.meta_mlp(x_inc)

        # --- Fusion ---
        x_fused = torch.cat([x_vis, x_meta], dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x_fused)))
        x = self.dropout1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        out = self.output(x)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, inc_angles, labels in loader:
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(np.array(all_targets) > 0.5, np.array(all_preds) > 0.5)

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, inc_angles, labels in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_log_loss = log_loss(all_targets, all_preds, labels=[0, 1])
    epoch_acc = accuracy_score(np.array(all_targets) > 0.5, np.array(all_preds) > 0.5)

    return epoch_loss, epoch_log_loss, epoch_acc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, inc_angles in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)

            outputs = model(images, inc_angles)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)

    return np.array(all_preds).flatten()


def run_training():
    """
    Main function to execute the 5-fold cross-validation training and submission generation.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data = load_data(load_cached_data=True)
    (
        X_train_full,
        y_train_full,
        inc_train_full,
        X_val_holdout,
        y_val_holdout,
        inc_val_holdout,
        X_test,
        inc_test,
        ids_test,
    ) = data

    # Combine Train and Val for Cross-Validation (since we are doing 5-fold CV)
    # The provided validation set in metadata is a holdout, but for 5-fold CV we usually
    # want to use all available training data. However, to respect the provided metadata split strictly:
    # We will use X_train_full (which corresponds to train.csv) for CV.
    # X_val_holdout (val.csv) can be used as an external check or included if we merge.
    # Given the small dataset size, merging is often better, but let's stick to the training set
    # defined in metadata/train.csv for the CV splits to ensure we don't leak the fixed validation set
    # if it's intended for final scoring.
    # Actually, standard practice with provided train/val split is to use train for CV.

    X = X_train_full
    y = y_train_full
    inc = inc_train_full

    # Test Dataset
    test_dataset = IcebergDataset(X_test, inc_test, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Store predictions
    fold_test_preds = []

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Split Data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold, inc_train_fold, y_train_fold, transform=True
        )
        val_dataset = IcebergDataset(
            X_val_fold, inc_val_fold, y_val_fold, transform=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = SC_WBN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="min")

        # Training Loop
        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_log_loss, val_acc = validate(
                model, val_loader, criterion, device
            )

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val LogLoss: {val_log_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load Best Model
        model.load_state_dict(early_stopping.best_model_state)

        # Save Model Checkpoint
        fold_model_path = os.path.join(Config.WORKING_DIR, f"sc_wbn_fold_{fold}.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved best model for fold {fold+1} to {fold_model_path}")

        # Predict on Test
        print("Generating predictions for test set...")
        preds = predict(model, test_loader, device)
        fold_test_preds.append(preds)

    # Ensemble Predictions (Mean)
    avg_preds = np.mean(fold_test_preds, axis=0)

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is included for local testing if needed, but the function run_training
    # is the primary entry point expected by the system.
    run_training()
