import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import set_seed
from library.data_loader import load_data, IcebergDataset, get_transforms


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block with Bias, BatchNorm, ReLU, SE, and MaxPool.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Bias is explicitly retained as per stability protocols
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class StabilizedSECNN(nn.Module):
    """
    Stabilized Selective Hierarchical SE-CNN.
    Features:
    - 4-Stage Plain CNN Backbone (64 -> 128 -> 128 -> 128)
    - Early Expansion at Block 2
    - Selective Hierarchical Pooling (Block 3 & 4)
    - Raw Incidence Angle Fusion
    - Single Hidden Layer Classifier
    """

    def __init__(self):
        super(StabilizedSECNN, self).__init__()

        # Backbone
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 128)
        self.block4 = ConvBlock(128, 128)

        # Classifier
        # Input Dimension Calculation:
        # Block 3 Global Max Pool: 128 channels
        # Block 4 Global Max Pool: 128 channels
        # Incidence Angle: 1 scalar
        # Total: 128 + 128 + 1 = 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # Weights are initialized using PyTorch defaults (Kaiming Uniform approx)

    def forward(self, x, angle):
        # x: (Batch, 3, 75, 75)
        # angle: (Batch, )

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        # Selective Hierarchical Pooling
        # Global Max Pooling on Block 3 (Medium scale features)
        p3 = F.adaptive_max_pool2d(x3, (1, 1)).flatten(1)

        # Global Max Pooling on Block 4 (Abstract features)
        p4 = F.adaptive_max_pool2d(x4, (1, 1)).flatten(1)

        # Reshape angle for concatenation
        ang = angle.view(-1, 1)

        # Feature Fusion
        features = torch.cat([p3, p4, ang], dim=1)

        # Classification
        out = self.classifier(features)
        return out


def train_and_predict(epochs=50, batch_size=32, patience=10, seed=42):
    """
    Executes the 5-Fold Cross-Validation training pipeline and generates the submission.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # load_data handles caching and imputation
    data = load_data(load_cached_data=True)

    # 2. Prepare for K-Fold
    # Merge train and validation sets provided by loader to perform stratified K-Fold
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Accumulator for test predictions (probabilities)
    test_preds_accum = np.zeros(len(X_test))

    # Prepare Test Loader
    test_dataset = IcebergDataset(X_test, angle_test, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Starting 5-Fold Cross-Validation on {len(X_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nFold {fold + 1}/5")

        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        angle_train_fold, angle_val_fold = angle_full[train_idx], angle_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold,
            angle_train_fold,
            y_train_fold,
            transform=get_transforms("train"),
        )
        val_dataset = IcebergDataset(
            X_val_fold, angle_val_fold, y_val_fold, transform=get_transforms("test")
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = StabilizedSECNN().to(device)

        # Optimizer: Adam with constant LR
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Loss: BCEWithLogitsLoss (combines Sigmoid + BCELoss)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss_sum = 0.0
            for imgs, angles, labels in train_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                labels = labels.to(device).unsqueeze(1)  # Match output shape (B, 1)

                optimizer.zero_grad()
                outputs = model(imgs, angles)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * imgs.size(0)

            train_loss = train_loss_sum / len(train_dataset)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for imgs, angles, labels in val_loader:
                    imgs = imgs.to(device)
                    angles = angles.to(device)
                    labels = labels.to(device).unsqueeze(1)

                    outputs = model(imgs, angles)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * imgs.size(0)

            val_loss = val_loss_sum / len(val_dataset)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Check Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Inference on Test Set for this Fold
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for imgs, angles in test_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                outputs = model(imgs, angles)
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        test_preds_accum += np.array(fold_preds)

    # 4. Ensemble and Submission
    # Average probabilities across 5 folds
    avg_preds = test_preds_accum / 5.0

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    submission.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
