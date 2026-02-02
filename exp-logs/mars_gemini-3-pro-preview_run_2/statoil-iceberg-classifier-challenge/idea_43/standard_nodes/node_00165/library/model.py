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

# Import configuration and utilities from the provided library
from library.config import Config, process_data, IcebergDataset, seed_everything

# ==========================================
# MODEL ARCHITECTURE
# ==========================================


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Use a reduction ratio to reduce parameter count
        hidden_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

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
        # Mixed Pooling for Attention Logic (Max + Avg)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class WideBlock(nn.Module):
    """
    Wide-Body Delayed-Integration Block.
    Features:
    - Wide Convolution (Input -> 128)
    - Pre-Pooling CBAM
    - Dual-Stream Pooling (Max + Min)
    """

    def __init__(self, in_channels, out_channels):
        super(WideBlock, self).__init__()
        # Delayed Integration: Wide Conv -> BN -> ReLU
        # Maps input (e.g. 256) to internal width (e.g. 128)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-Pooling Attention
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.cbam(x)

        # Dual-Stream Pooling (Max + Min)
        # Min Pooling is implemented as -MaxPool(-x)
        max_p = F.max_pool2d(x, 2)
        min_p = -F.max_pool2d(-x, 2)

        # Concatenate -> Output channels doubles for next stage input
        out = torch.cat([max_p, min_p], dim=1)
        return out


class DMWBNet(nn.Module):
    def __init__(self):
        super(DMWBNet, self).__init__()

        # --- Visual Branch ---
        # Stage 1: Input 3 -> 128 (Wide)
        self.stage1_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.stage1_bn = nn.BatchNorm2d(128)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_cbam = CBAM(128)
        # Pooling 1 happens in forward: 128 -> 256 (due to dual pool)

        # Stage 2: Input 256 -> 128 internal -> 256 out
        self.stage2 = WideBlock(256, 128)

        # Stage 3: Input 256 -> 128 internal -> 256 out
        self.stage3 = WideBlock(256, 128)

        # Stage 4: Input 256 -> 128 internal -> 256 out
        self.stage4 = WideBlock(256, 128)

        # Output of Stage 4 is 256 channels
        # Spatial Dimensions: 75 -> 37 -> 18 -> 9 -> 4

        # --- Readout ---
        # Path A: Spatial Context
        # Compresses 256 -> 64, keeps spatial 4x4
        self.path_a_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        # Flattened size: 64 * 4 * 4 = 1024

        # Path B: Robust Intensity (GAP)
        self.path_b_pool = nn.AdaptiveAvgPool2d(1)
        # Size: 256

        # --- Metadata Branch ---
        # Deep MLP with Batch Normalization
        self.meta_fc1 = nn.Linear(1, 32)
        self.meta_fc2 = nn.Linear(32, 32)
        self.meta_bn = nn.BatchNorm1d(32)

        # --- Fusion Head ---
        # Input: 1024 (Path A) + 256 (Path B) + 32 (Meta) = 1312
        self.fusion_fc1 = nn.Linear(1312, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_drop = nn.Dropout(0.5)
        self.fusion_out = nn.Linear(512, 1)

    def forward(self, x_img, x_meta):
        # Visual Branch
        # Stage 1
        x = self.stage1_relu(self.stage1_bn(self.stage1_conv(x_img)))
        x = self.stage1_cbam(x)
        # Dual Pool Stage 1
        x = torch.cat([F.max_pool2d(x, 2), -F.max_pool2d(-x, 2)], dim=1)

        # Stages 2-4
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # Output (B, 256, 4, 4)

        # Readout Path A
        xa = self.path_a_conv(x)  # (B, 64, 4, 4)
        xa = xa.view(xa.size(0), -1)  # (B, 1024)

        # Readout Path B
        xb = self.path_b_pool(x).view(x.size(0), -1)  # (B, 256)

        # Metadata Branch
        xm = F.relu(self.meta_fc1(x_meta))
        xm = self.meta_fc2(xm)
        xm = F.relu(self.meta_bn(xm))  # (B, 32)

        # Fusion
        fused = torch.cat([xa, xb, xm], dim=1)
        x = F.relu(self.fusion_bn(self.fusion_fc1(fused)))
        x = self.fusion_drop(x)
        x = self.fusion_out(x)

        return torch.sigmoid(x)


# ==========================================
# TRAINING PIPELINE
# ==========================================


def train_and_evaluate(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Executes the training pipeline including:
    1. Data Loading & Caching
    2. Stratified 5-Fold Cross Validation
    3. Model Training with Early Stopping
    4. Inference and Submission Generation
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (Cached)
    # process_data handles loading or computing and caching
    X, y, inc, X_test, inc_test, test_ids = process_data(load_cached_data=True)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Placeholders for predictions
    test_preds_accumulator = np.zeros(len(X_test))

    # Iterate Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Split Data
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        inc_tr, inc_val = inc[train_idx], inc[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

        # DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = DMWBNet().to(device)

        # Optimizer & Loss
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.BCELoss()

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Training Loop Variables
        best_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

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

            epoch_loss = running_loss / len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, angles, labels in val_loader:
                    inputs = inputs.to(device)
                    angles = angles.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs, angles)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_ds)

            # Update Scheduler
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.10f} | Val Loss: {val_loss:.10f}"
            )

            # Early Stopping & Checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Save Best Model for this Fold
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        torch.save(best_model_wts, model_path)
        print(f"Saved best model for fold {fold} to {model_path}")

        # Inference on Test Set with Best Model
        model.load_state_dict(best_model_wts)
        model.eval()

        test_ds = IcebergDataset(X_test, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        fold_preds = []
        with torch.no_grad():
            for inputs, angles in test_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                outputs = model(inputs, angles)
                fold_preds.extend(outputs.cpu().numpy().flatten())

        # Add to accumulator (average later)
        test_preds_accumulator += np.array(fold_preds)

    # Average Predictions
    final_preds = test_preds_accumulator / Config.N_FOLDS

    # Save Submission
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
