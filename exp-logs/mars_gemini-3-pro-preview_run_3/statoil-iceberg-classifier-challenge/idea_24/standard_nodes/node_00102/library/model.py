import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library files
from library.utils import load_data, set_seed
from library.dataset import IcebergDataset


class MaxSEModule(nn.Module):
    """
    Max-Squeeze-and-Excitation Module.
    Uses Global Max Pooling to capture peak signal intensity (icebergs)
    instead of average background (sea).
    """

    def __init__(self, channels, reduction=16):
        super(MaxSEModule, self).__init__()
        # Ensure reduction doesn't make hidden dim 0
        hidden_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Max Pooling
        y = F.adaptive_max_pool2d(x, 1).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class CAMA_CNN(nn.Module):
    """
    Contrast-Aware Max-Attentive CNN.
    Features:
    - 4-Stage Plain CNN Backbone (LeakyReLU, Bias=True)
    - Max-SE Attention Blocks
    - Dual Global Pooling (Max + Avg) to model Signal-to-Clutter Ratio
    - Fusion with unnormalized incidence angle
    """

    def __init__(self):
        super(CAMA_CNN, self).__init__()

        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = MaxSEModule(64)

        # Block 2: 64 -> 128 (Early Expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = MaxSEModule(128)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = MaxSEModule(128)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.se4 = MaxSEModule(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Classifier Head
        # Input: 128 (Max) + 128 (Avg) + 1 (Angle) = 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x, angle):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.se1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.se2(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.se3(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        x = self.se4(x)
        x = self.pool(x)

        # Dual Global Pooling
        # x is (N, 128, H', W')
        x_max = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)  # (N, 128)
        x_avg = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # (N, 128)

        # Concatenate pooling features
        feat = torch.cat([x_max, x_avg], dim=1)  # (N, 256)

        # Fuse Angle
        angle = angle.view(-1, 1)  # (N, 1)
        feat = torch.cat([feat, angle], dim=1)  # (N, 257)

        # Classification
        out = self.classifier(feat)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (N, 1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(targets)


def train_cama_cnn(
    batch_size=32,
    epochs=75,
    patience=12,
    lr=1e-3,
    weight_decay=1e-4,
    n_folds=5,
    seed=42,
    cache_dir="./working/idea_24",
    submission_path="./submission/submission.csv",
):
    """
    Executes the 5-Fold Cross-Validation training pipeline for CAMA-CNN.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # We use load_data to get the raw arrays, then merge train/val for CV
    data = load_data(cache_dir=cache_dir, load_cached_data=True)
    (
        X_train_raw,
        y_train_raw,
        angles_train_raw,
        X_val_raw,
        y_val_raw,
        angles_val_raw,
        X_test,
        ids_test,
        angles_test,
    ) = data

    # Merge Train and Val for Stratified K-Fold
    X_all = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)
    angles_all = np.concatenate([angles_train_raw, angles_val_raw], axis=0)

    print(f"Total training samples for CV: {len(y_all)}")

    # Prepare Test Loader (used for inference after each fold)
    test_dataset = IcebergDataset(X_test, angles_test, transform=None)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Store test predictions from each fold
    fold_test_preds = np.zeros((len(X_test), n_folds))

    # 2. K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # Split Data
        X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
        y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]
        angles_train_fold, angles_val_fold = angles_all[train_idx], angles_all[val_idx]

        # Create Datasets
        # Apply augmentation only to training set
        train_transform = torch.nn.Sequential(
            torch.nn.Identity()  # Placeholder, augmentation handled in Dataset if needed
            # Note: torchvision transforms are usually passed to Dataset.
            # We'll use the same logic as dataset.py but re-instantiate here.
        )
        # Using torchvision transforms for augmentation
        from torchvision import transforms

        train_aug = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

        train_ds = IcebergDataset(
            X_train_fold, angles_train_fold, y_train_fold, transform=train_aug
        )
        val_ds = IcebergDataset(X_val_fold, angles_val_fold, y_val_fold, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Initialize Model
        model = CAMA_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, _, _ = validate(model, val_loader, criterion, device)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}"
                )
                break

        # Load best model for this fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Predict on Test Set
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy().flatten())

        fold_test_preds[:, fold] = fold_preds
        print(f"Fold {fold+1} complete. Best Val Loss: {best_val_loss:.6f}")

    # 3. Ensemble Predictions
    print("\nGenerating final submission...")
    avg_preds = np.mean(fold_test_preds, axis=1)

    # Save Submission
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
