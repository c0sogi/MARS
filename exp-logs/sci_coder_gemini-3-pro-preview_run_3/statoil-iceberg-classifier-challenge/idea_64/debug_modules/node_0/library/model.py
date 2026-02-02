import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import os

# Import from provided libraries
from library.utils import (
    load_and_process_data,
    IcebergDataset,
    train_one_epoch,
    validate,
    set_seed,
    logger,
)
from library.data import get_transforms


class HybridSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = HybridSE(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        feat = self.se(x)
        out = self.pool(feat)
        return out, feat


class MSICNN(nn.Module):
    def __init__(self):
        super(MSICNN, self).__init__()

        # Backbone: Plain CNN with 4 stages
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 128)
        self.block4 = ConvBlock(128, 128)

        # Readout Projections (Decoupled)
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # Head: Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        # Features: Max3(64) + Min3(64) + Max4(64) + Min4(64) + Angle(1) = 257
        self.fc = nn.Linear(257, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Stage 1
        x, _ = self.block1(x)
        # Stage 2
        x, _ = self.block2(x)
        # Stage 3
        x, feat3 = self.block3(x)
        # Stage 4
        x, feat4 = self.block4(x)

        # Isomorphic Readout Stage 3
        f3 = self.proj3(feat3)
        max3 = F.adaptive_max_pool2d(f3, 1).view(f3.size(0), -1)
        min3 = -F.adaptive_max_pool2d(-f3, 1).view(f3.size(0), -1)

        # Isomorphic Readout Stage 4
        f4 = self.proj4(feat4)
        max4 = F.adaptive_max_pool2d(f4, 1).view(f4.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-f4, 1).view(f4.size(0), -1)

        # Feature Fusion
        features = torch.cat([max3, min3, max4, min4, angle.view(-1, 1)], dim=1)

        # Multi-Sample Dropout Head
        if self.training:
            out = []
            for drop in self.dropouts:
                out.append(self.fc(drop(features)))
            return torch.stack(out, dim=1)  # (B, 5, 1)
        else:
            # Inference: In eval mode, dropout is identity, so all branches are identical.
            # We return a single pass which is equivalent to the mean of identical branches.
            return self.fc(features)  # (B, 1)


def train_and_submit(epochs=75, batch_size=32, patience=12):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data using library utils (handles caching)
    data = load_and_process_data(load_cached_data=True)
    X, y, angles = data["X_train"], data["y_train"], data["angle_train"]

    # Prepare test data container
    test_preds_accum = np.zeros(len(data["X_test"]))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Starting Fold {fold+1}/5")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        ang_tr, ang_val = angles[train_idx], angles[val_idx]

        # Leak-free angle imputation
        valid_ang_tr = ang_tr[~np.isnan(ang_tr)]
        fold_median = np.median(valid_ang_tr) if len(valid_ang_tr) > 0 else 0.0

        train_ds = IcebergDataset(
            X_tr,
            y_tr,
            ang_tr,
            transform=get_transforms("train"),
            angle_impute_val=fold_median,
        )
        val_ds = IcebergDataset(
            X_val,
            y_val,
            ang_val,
            transform=get_transforms("val"),
            angle_impute_val=fold_median,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        model = MSICNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            print(
                f"Fold {fold+1} Epoch {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Predict on Test with Best Model for this fold
        model.load_state_dict(best_state)
        model.eval()

        test_ds = IcebergDataset(
            data["X_test"],
            None,
            data["angle_test"],
            transform=get_transforms("test"),
            angle_impute_val=fold_median,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        fold_preds = []
        with torch.no_grad():
            for images, angs in test_loader:
                images = images.to(device)
                angs = angs.to(device)
                out = model(images, angs)  # (B, 1)
                fold_preds.append(torch.sigmoid(out).cpu().numpy())

        test_preds_accum += np.concatenate(fold_preds).flatten() / 5.0

    # Submission
    submission = pd.DataFrame({"id": data["ids_test"], "is_iceberg": test_preds_accum})
    os.makedirs("./submission", exist_ok=True)
    submission.to_csv("./submission/submission.csv", index=False)
    logger.info("Submission saved successfully.")
