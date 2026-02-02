import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import process_data, IcebergDataset

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class MaxSELayer(nn.Module):
    """
    Max-Squeeze-and-Excitation Module.
    Uses Global Max Pooling to capture high-intensity signal peaks (icebergs)
    rather than average background noise.
    """

    def __init__(self, channels, reduction=16):
        super(MaxSELayer, self).__init__()
        # Ensure at least 1 channel in the bottleneck
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Max Pooling
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = F.adaptive_max_pool2d(x, 1).view(b, c)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale
        y = y.view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Custom Residual Block with:
    - Biased Convolutions (Bias Retention)
    - LeakyReLU Activation
    - Max-SE Attention
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main Path
        # Note: bias=True is explicitly used as per "Bias Retention" strategy
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention
        self.se = MaxSELayer(out_channels)

        # Shortcut Path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=True
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply Attention
        out = self.se(out)

        # Residual Connection
        out += self.shortcut(x)
        out = self.act(out)
        return out


class MADResNet(nn.Module):
    """
    Max-Attentive Deep-Downsampled ResNet.
    4-Stage Residual Network with aggressive downsampling and raw angle fusion.
    """

    def __init__(self):
        super(MADResNet, self).__init__()

        widths = Config.CHANNEL_WIDTHS  # [64, 128, 128, 128]

        # Stage 1: Input (3) -> 64. Stride 2 (75 -> 38)
        self.stage1 = ResidualBlock(Config.IN_CHANNELS, widths[0], stride=2)

        # Stage 2: 64 -> 128. Stride 2 (38 -> 19)
        self.stage2 = ResidualBlock(widths[0], widths[1], stride=2)

        # Stage 3: 128 -> 128. Stride 2 (19 -> 10)
        self.stage3 = ResidualBlock(widths[1], widths[2], stride=2)

        # Stage 4: 128 -> 128. Stride 2 (10 -> 5)
        self.stage4 = ResidualBlock(widths[2], widths[3], stride=2)

        # Classification Head
        # Global Max Pool reduces (B, 128, 5, 5) -> (B, 128)
        # Concatenate 1 angle feature -> 129 input features
        head_in = widths[3] + 1
        hidden_dim = 256

        self.head_fc1 = nn.Linear(head_in, hidden_dim)
        self.head_act = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)
        self.head_drop = nn.Dropout(Config.DROPOUT_RATE)
        self.head_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, angle):
        # Backbone
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global Max Pooling
        x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        # Feature Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Head
        x = self.head_fc1(x)
        x = self.head_act(x)
        x = self.head_drop(x)
        x = self.head_fc2(x)

        return x


# -------------------------------------------------------------------------
# Training and Execution Logic
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        # Squeeze logits to match label shape (B,)
        loss = criterion(logits.squeeze(1), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            logits = model(images, angles)
            loss = criterion(logits.squeeze(1), labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).squeeze(1)
            preds.extend(probs.cpu().numpy())

    return np.array(preds)


def run_training_and_submission():
    """
    Main execution pipeline:
    1. Load data.
    2. Perform 5-Fold Cross-Validation Training.
    3. Generate Ensemble Predictions.
    4. Save Submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 1. Load Data
    # process_data handles caching automatically
    data = process_data(load_cached_data=True)

    # Merge Train and Val for Stratified K-Fold
    # The provided dataset split in metadata is 80/20, but for 5-fold CV
    # we want to use the entire labeled dataset.
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # Debugging Subset
    if Config.DEBUG:
        print("DEBUG MODE: Using subset of data.")
        subset_size = Config.DEBUG_SUBSET_SIZE
        X_full = X_full[:subset_size]
        angle_full = angle_full[:subset_size]
        y_full = y_full[:subset_size]
        X_test = X_test[:subset_size]
        angle_test = angle_test[:subset_size]
        ids_test = ids_test[:subset_size]
        epochs_to_run = 2
    else:
        epochs_to_run = Config.NUM_EPOCHS

    # 2. 5-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Array to store test predictions from each fold
    test_preds_accum = np.zeros(len(X_test))

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nStarting Fold {fold}...")

        # Split Data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        ang_tr, ang_va = angle_full[train_idx], angle_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = MADResNet().to(device)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")

        for epoch in range(epochs_to_run):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": best_val_loss,
                    },
                    is_best=True,
                    fold=fold,
                )

            # Optional: Print progress occasionally
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs_to_run} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
                )

        print(f"Fold {fold} Best Val Loss: {best_val_loss:.5f}")

        # 3. Inference on Test Set for this Fold
        # Load best model
        best_model_path = os.path.join(Config.IDEA_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = load_checkpoint(best_model_path, model, device=device)

        # Create Test Loader
        test_ds = IcebergDataset(X_test, angle_test, ids=ids_test, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict
        fold_preds = predict(model, test_loader, device)
        test_preds_accum += fold_preds

        # Clean up to save memory
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Average Predictions and Submit
    avg_preds = test_preds_accum / Config.NUM_FOLDS

    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    print(f"\nSaving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


# Run the pipeline
run_training_and_submission()
