import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_FOLDS,
    SEED,
    DEVICE,
    DROPOUT_RATE,
    LEAKY_RELU_SLOPE,
    PATIENCE,
    WEIGHT_DECAY,
)
from library.utils import set_seed, get_logger
from library.data_loader import get_dataloaders, get_test_loader

# ==========================================
# Model Architecture (IDPH-CNN)
# ==========================================


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
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
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True)
        self.se = SEModule(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class IDPH_CNN(nn.Module):
    def __init__(self):
        super(IDPH_CNN, self).__init__()

        # Backbone: Plain CNN 4 blocks
        # Input: 75x75
        self.block1 = ConvBlock(3, 64)  # -> 37x37
        self.block2 = ConvBlock(64, 128)  # -> 18x18
        self.block3 = ConvBlock(128, 128)  # -> 9x9 (Stage 3 Output)
        self.block4 = ConvBlock(128, 128)  # -> 4x4 (Stage 4 Output)

        # Isomorphic Projections (1x1 Conv)
        # Compresses 128 channels to 64 to maintain capacity after dual-pooling
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # Classifier Head
        # Input: 256 (Image features) + 1 (Angle) = 257
        # Image features = 64 (S3 Max) + 64 (S3 Min) + 64 (S4 Max) + 64 (S4 Min)
        self.head = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Backbone
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)  # Stage 3 Output
        x4 = self.block4(x3)  # Stage 4 Output

        # Isomorphic Readout Stage 3
        p3 = self.proj3(x3)  # (B, 64, H, W)
        p3_max = torch.amax(p3, dim=(2, 3))  # Global Max Pool
        p3_min = -torch.amax(-p3, dim=(2, 3))  # Global Min Pool (via Max(-x))

        # Isomorphic Readout Stage 4
        p4 = self.proj4(x4)  # (B, 64, H, W)
        p4_max = torch.amax(p4, dim=(2, 3))
        p4_min = -torch.amax(-p4, dim=(2, 3))

        # Aggregate Image Features
        img_feats = torch.cat([p3_max, p3_min, p4_max, p4_min], dim=1)  # 64*4 = 256

        # Feature Fusion
        angle = angle.view(-1, 1)
        combined = torch.cat([img_feats, angle], dim=1)  # 257

        # Classification
        out = self.head(combined)
        return out


# ==========================================
# Training Logic
# ==========================================


def train_model(load_cached_data=True):
    logger = get_logger("IDPH_CNN_Trainer")
    set_seed(SEED)

    logger.info("Initializing Training with IDPH-CNN Architecture")
    logger.info(f"Device: {DEVICE}, Folds: {NUM_FOLDS}, Epochs: {NUM_EPOCHS}")

    # Prepare for submission accumulation
    # We need test_ids to structure the submission file.
    # We can get them from the test loader.
    test_loader, test_ids = get_test_loader(
        batch_size=BATCH_SIZE, load_cached=load_cached_data
    )
    test_preds_accum = np.zeros(len(test_ids))

    os.makedirs("./checkpoints", exist_ok=True)

    for fold in range(NUM_FOLDS):
        logger.info(f"\n--- Starting Fold {fold + 1}/{NUM_FOLDS} ---")

        # Get Fold Dataloaders
        train_loader, val_loader = get_dataloaders(
            fold, batch_size=BATCH_SIZE, load_cached=load_cached_data
        )

        # Initialize Model, Optimizer, Criterion
        model = IDPH_CNN().to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = f"./checkpoints/model_fold_{fold}.pth"

        # Training Loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_sum = 0.0
            total_train_samples = 0

            for images, angs, labels in train_loader:
                images = images.to(DEVICE)
                angs = angs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images, angs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * images.size(0)
                total_train_samples += images.size(0)

            avg_train_loss = train_loss_sum / total_train_samples

            # Validation
            model.eval()
            val_loss_sum = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for images, angs, labels in val_loader:
                    images = images.to(DEVICE)
                    angs = angs.to(DEVICE)
                    labels = labels.to(DEVICE).unsqueeze(1)

                    outputs = model(images, angs)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item() * images.size(0)
                    total_val_samples += images.size(0)

            avg_val_loss = val_loss_sum / total_val_samples

            logger.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
            )

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        # Load Best Model for Inference
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Test Inference (No TTA)
        fold_test_preds = []
        with torch.no_grad():
            for images, angs in test_loader:
                images = images.to(DEVICE)
                angs = angs.to(DEVICE)
                out = torch.sigmoid(model(images, angs))
                fold_test_preds.extend(out.cpu().numpy().flatten())

        test_preds_accum += np.array(fold_test_preds)

    # Average Predictions
    avg_test_preds = test_preds_accum / NUM_FOLDS

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
