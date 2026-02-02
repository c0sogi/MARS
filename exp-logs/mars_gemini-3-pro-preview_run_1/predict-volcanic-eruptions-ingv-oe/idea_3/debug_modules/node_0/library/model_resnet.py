import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import copy
import warnings

from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.dataset import get_dataset, SeismicDataset

# Suppress warnings
warnings.filterwarnings("ignore")


class BasicBlock1D(nn.Module):
    """
    Standard ResNet Basic Block adapted for 1D data.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=3):
        super(BasicBlock1D, self).__init__()
        # Padding to maintain dimension: (k-1)//2 for odd k
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet-18 architecture adapted for 1D Seismic Data.
    Input: (Batch, Sensors, Time) -> (B, 10, 60001)
    Output: (B, 1) -> Time to eruption
    """

    def __init__(self):
        super(ResNet1D, self).__init__()

        # Hyperparameters from Config
        self.in_planes = Config.RESNET_BASE_FILTERS
        base_kernel = Config.RESNET_KERNEL_SIZE

        # Initial Convolution
        # Input channels = 10 (Sensors)
        self.conv1 = nn.Conv1d(
            Config.NUM_SENSORS,
            self.in_planes,
            kernel_size=base_kernel,
            stride=2,
            padding=(base_kernel - 1) // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        # We use a fixed kernel size of 3 for the internal blocks, standard for ResNet
        block_kernel = 3

        self.layer1 = self._make_layer(
            BasicBlock1D, 64, 2, stride=1, kernel_size=block_kernel
        )
        self.layer2 = self._make_layer(
            BasicBlock1D, 128, 2, stride=2, kernel_size=block_kernel
        )
        self.layer3 = self._make_layer(
            BasicBlock1D, 256, 2, stride=2, kernel_size=block_kernel
        )
        self.layer4 = self._make_layer(
            BasicBlock1D, 512, 2, stride=2, kernel_size=block_kernel
        )

        # Final Regression Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, 1)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        layers = []
        layers.append(block(self.in_planes, planes, stride, kernel_size))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, 10, 60001)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # Squeeze outputs to match target shape (B,)
        loss = criterion(outputs.squeeze(), targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            preds.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Calculate MAE using the full arrays to be safe
    epoch_mae = calculate_mae(np.array(actuals), np.array(preds))

    return epoch_loss, epoch_mae


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs in loader:
            # Check if loader returns tuple (x, y) or just x
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0]

            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.squeeze().cpu().numpy())

    return np.array(preds)


def run_resnet_cv(load_cached_data=True):
    """
    Executes the 1D-ResNet training pipeline with 5-Fold Cross-Validation.

    Args:
        load_cached_data (bool): Whether to load raw data from cache if available.

    Returns:
        pd.DataFrame: DataFrame containing 'segment_id' and 'time_to_eruption' predictions for the test set.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing Deep Learning Branch (1D-ResNet)...")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    # We load train and val separately, then merge them for CV.
    print("Loading datasets...")
    train_ds_raw = get_dataset(Config.TRAIN_METADATA, "train", load_cached_data)
    val_ds_raw = get_dataset(Config.VAL_METADATA, "val", load_cached_data)
    test_ds_raw = get_dataset(Config.TEST_METADATA, "test", load_cached_data)

    # Merge Train and Val for full 5-Fold CV
    X_full = np.concatenate([train_ds_raw.data, val_ds_raw.data], axis=0)
    y_full = np.concatenate([train_ds_raw.targets, val_ds_raw.targets], axis=0)

    print(f"Total Training Samples: {len(X_full)}")
    print(f"Total Test Samples: {len(test_ds_raw.data)}")

    # Get Test Segment IDs for submission
    df_test_meta = pd.read_csv(Config.TEST_METADATA)
    test_ids = df_test_meta["segment_id"].values

    # ---------------------------------------------------------
    # 2. Cross-Validation Loop
    # ---------------------------------------------------------
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Store test predictions from each fold
    test_preds_accum = np.zeros(len(test_ds_raw.data))
    oof_preds = np.zeros(len(X_full))

    # Test Loader (Fixed)
    test_dataset = SeismicDataset(test_ds_raw.data, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split Data
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]

        # Create Datasets
        train_dataset = SeismicDataset(X_train, y_train)
        val_dataset = SeismicDataset(X_val, y_val)

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = ResNet1D().to(device)

        # Loss & Optimizer
        criterion = nn.L1Loss()  # MAE
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Training Loop
        best_mae = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience = 7
        counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_mae = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Print metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae}"
            )

            # Early Stopping Check
            if val_mae < best_mae:
                best_mae = val_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Fold {fold + 1} Best Val MAE: {best_mae}")

        # Load best model for inference
        model.load_state_dict(best_model_wts)

        # Generate OOF predictions
        oof_preds[val_idx] = predict(model, val_loader, device)

        # Generate Test predictions
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

    # ---------------------------------------------------------
    # 3. Aggregation & Results
    # ---------------------------------------------------------
    avg_test_preds = test_preds_accum / Config.N_FOLDS
    total_mae = calculate_mae(y_full, oof_preds)

    print(f"\nOverall CV MAE (ResNet): {total_mae}")

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": avg_test_preds}
    )

    # Ensure segment_id is integer
    submission_df["segment_id"] = submission_df["segment_id"].astype(int)

    return submission_df
