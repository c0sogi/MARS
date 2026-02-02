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

from library.config import DEVICE, CACHE_DIR, SUBMISSION_DIR, RANDOM_SEED
from library.utils import get_logger, set_seed
from library.data_loader import process_and_cache_data, IcebergDataset

# Initialize logger
logger = get_logger("model")

# ==========================================
# 1. Architectural Components
# ==========================================


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Applies Max Pooling (Peaks) and Min Pooling (Shadows)
    and concatenates the outputs along the channel dimension.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Max Pooling (Peaks)
        max_p = self.pool(x)

        # Min Pooling (Shadows)
        # Implemented as negative of max pool of negative input
        min_p = -self.pool(-x)

        # Concatenate: (N, C, H, W) -> (N, 2C, H/2, W/2)
        return torch.cat([max_p, min_p], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module using Mixed Pooling (Max + Avg).
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Use Conv2d with kernel_size=1 for channel-wise fully connected layers
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Summation fusion
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module using Mixed Pooling (Max + Avg) across channels.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # Input channels = 2 (1 for Avg, 1 for Max)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Avg Pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max Pool across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Convolution
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Refine features sequentially
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class MSSCWBN(nn.Module):
    """
    Multi-Scale Spatial-Context Wide-Body Network.
    """

    def __init__(self):
        super(MSSCWBN, self).__init__()

        # --- Visual Backbone (Wide-Body Delayed-Integration) ---

        # Stage 1
        # Input: 3 channels (HH, HV, Avg)
        # Map to 128 filters immediately
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.cbam1 = CBAM(128)
        self.pool1 = DualPooling()  # Output: 128*2 = 256 channels

        # Stage 2
        # Input: 256 channels
        # Delayed Integration: 256 -> 128
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.pool2 = DualPooling()  # Output: 256 channels

        # Stage 3
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.pool3 = DualPooling()  # Output: 256 channels

        # Stage 4
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)
        self.pool4 = DualPooling()  # Output: 256 channels

        # Note: Input 75x75
        # Pool1 -> 37x37
        # Pool2 -> 18x18
        # Pool3 -> 9x9
        # Pool4 -> 4x4

        # --- Multi-Scale Readout Interface ---
        # Input Feature Map: 256 x 4 x 4

        # Path A: Context Stream (Spatial Adjacency)
        # 3x3 Conv, Pad 1 -> 32 channels
        self.path_a_conv = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        # Flatten: 32 * 4 * 4 = 512 dim

        # Path B: Detail Stream (Pixel Fidelity)
        # 1x1 Conv -> 16 channels
        self.path_b_conv = nn.Conv2d(256, 16, kernel_size=1)
        # Flatten: 16 * 4 * 4 = 256 dim

        # Path C: Global Stream (Intensity Statistics)
        # Global Avg Pooling -> 256 channels
        self.path_c_pool = nn.AdaptiveAvgPool2d(1)
        # Flatten: 256 * 1 * 1 = 256 dim

        # Total Visual Dim: 512 + 256 + 256 = 1024

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_fc2 = nn.Linear(16, 32)

        # --- Fusion Head ---
        # Input: 1024 (Visual) + 32 (Meta) = 1056
        self.head_fc = nn.Linear(1056, 512)
        self.head_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.out_fc = nn.Linear(512, 1)

    def forward(self, x, angle):
        # --- Visual Forward ---
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool1(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        # Readout Paths
        # Path A
        xa = self.path_a_conv(x)
        xa = xa.view(xa.size(0), -1)  # Flatten

        # Path B
        xb = self.path_b_conv(x)
        xb = xb.view(xb.size(0), -1)  # Flatten

        # Path C
        xc = self.path_c_pool(x)
        xc = xc.view(xc.size(0), -1)  # Flatten

        # Fuse Visual
        x_visual = torch.cat([xa, xb, xc], dim=1)

        # --- Metadata Forward ---
        # Angle is (N, 1)
        m = F.relu(self.meta_fc1(angle))
        m = F.relu(self.meta_fc2(m))

        # --- Global Fusion ---
        x_final = torch.cat([x_visual, m], dim=1)

        x_final = F.relu(self.head_bn(self.head_fc(x_final)))
        x_final = self.dropout(x_final)

        out = self.out_fc(x_final)
        return out


# ==========================================
# 2. Training Logic
# ==========================================


def train_model(
    train_loader, val_loader, epochs=50, patience=10, device=DEVICE, fold_idx=0
):
    """
    Trains a single instance of MSSC-WBN.
    """
    model = MSSCWBN().to(device)

    # Optimizer: Adam (reverted from AdamW)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    logger.info(f"Starting training for Fold {fold_idx}...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, angles, labels in train_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, angles, labels in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                labels = labels.to(device)

                outputs = model(inputs, angles)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_loss:.6f}, Train Acc: {epoch_acc:.6f}, "
            f"Val Loss: {val_epoch_loss:.6f}, Val Acc: {val_epoch_acc:.6f}"
        )

        # Step Scheduler
        scheduler.step(val_epoch_loss)

        # Early Stopping
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save model artifact
    save_path = os.path.join(CACHE_DIR, f"model_fold_{fold_idx}.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved best model for Fold {fold_idx} to {save_path}")

    return model


def predict(model, test_loader, device=DEVICE):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, angles in test_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            outputs = model(inputs, angles)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy().flatten())

    return np.array(predictions)


def run_kfold_pipeline(epochs=50, patience=10):
    """
    Executes Stratified 5-Fold Cross-Validation training and generates submission.
    """
    set_seed(RANDOM_SEED)

    # 1. Load Processed Data
    data = process_and_cache_data(load_cached_data=True)

    X = data["train_images"]
    angles = data["train_angles"]
    y = data["train_labels"]

    X_test = data["test_images"]
    angles_test = data["test_angles"]
    test_ids = data["test_ids"]

    # 2. Prepare Test Loader
    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 3. Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    fold_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Create Datasets
        train_ds = IcebergDataset(
            X[train_idx], angles[train_idx], y[train_idx], transform=True
        )
        val_ds = IcebergDataset(
            X[val_idx], angles[val_idx], y[val_idx], transform=False
        )

        # Create Loaders
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

        # Train
        model = train_model(
            train_loader, val_loader, epochs=epochs, patience=patience, fold_idx=fold
        )

        # Predict on Test Set
        preds = predict(model, test_loader)
        fold_preds.append(preds)

    # 4. Aggregate Predictions (Mean)
    avg_preds = np.mean(fold_preds, axis=0)

    # 5. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
    df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
