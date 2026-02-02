import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import (
    DEVICE,
    CHECKPOINT_DIR,
    SUBMISSION_PATH,
    NUM_FOLDS,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    SEED,
)
from library.utils import set_seed, get_device
from library.data import get_loaders, get_test_loader


# -----------------------------------------------------------------------------
# 1. MAD-SE Module
# -----------------------------------------------------------------------------
class MADSELayer(nn.Module):
    """
    Mean Absolute Deviation Squeeze-and-Excitation Module.
    Uses Mean and MAD (Mean Absolute Deviation) as statistics for channel attention.
    Robust to speckle noise compared to Variance/StdDev.
    """

    def __init__(self, channels, reduction=16):
        super(MADSELayer, self).__init__()
        self.channels = channels
        # Input to MLP is 2 * channels (Mean + MAD)
        # Reduction ratio applies to the intermediate dimension
        reduced_dim = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. Compute Global Mean per channel
        # Shape: (B, C, 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)

        # 2. Compute Global Mean Absolute Deviation (MAD) per channel
        # |x - mu| -> mean spatial
        # Shape: (B, C, 1, 1)
        mad = (x - mu).abs().mean(dim=[2, 3], keepdim=True)

        # 3. Flatten statistics for MLP
        mu_flat = mu.view(b, c)
        mad_flat = mad.view(b, c)

        # 4. Concatenate statistics
        # Shape: (B, 2C)
        stats = torch.cat([mu_flat, mad_flat], dim=1)

        # 5. Excitation
        # Shape: (B, C, 1, 1)
        scale = self.fc(stats).view(b, c, 1, 1)

        # 6. Scale input
        return x * scale


# -----------------------------------------------------------------------------
# 2. RTI-CNN Architecture
# -----------------------------------------------------------------------------
class RTICNN(nn.Module):
    """
    Robust-Texture Isomorphic CNN.
    4-Stage Plain CNN with MAD-SE blocks and Corrected Decoupled Isomorphic Readout.
    """

    def __init__(self):
        super(RTICNN, self).__init__()

        # --- Stage 1 ---
        # 3 channels (HH, HV, Avg) -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(64, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 75 -> 37
        )

        # --- Stage 2 ---
        # 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 37 -> 18
        )

        # --- Stage 3 ---
        # 128 -> 128
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 18 -> 9
        )

        # --- Stage 4 ---
        # 128 -> 128
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MADSELayer(128, reduction=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 9 -> 4
        )

        # --- Corrected Decoupled Isomorphic Readout ---
        # Projections for Stage 3 and Stage 4
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1, bias=True)

        # --- Classifier Head ---
        # Input Feature Size:
        # Stage 3: 64 (Max) + 64 (Min) = 128
        # Stage 4: 64 (Max) + 64 (Min) = 128
        # Angle: 1
        # Total: 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming Uniform Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Backbone
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # --- Readout Stage 3 ---
        p3 = self.proj3(x3)  # (B, 64, H, W)
        p3_flat = p3.view(p3.size(0), p3.size(1), -1)
        max3 = p3_flat.max(dim=2)[0]
        min3 = p3_flat.min(dim=2)[0]

        # --- Readout Stage 4 ---
        p4 = self.proj4(x4)  # (B, 64, H, W)
        p4_flat = p4.view(p4.size(0), p4.size(1), -1)
        max4 = p4_flat.max(dim=2)[0]
        min4 = p4_flat.min(dim=2)[0]

        # --- Fusion ---
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate: (B, 64+64+64+64+1) = (B, 257)
        features = torch.cat([max3, min3, max4, min4, angle], dim=1)

        # Classification
        logits = self.classifier(features)

        return logits.squeeze(1)


# -----------------------------------------------------------------------------
# 3. Training Logic
# -----------------------------------------------------------------------------
def train_fold(fold_idx):
    """
    Trains the RTI-CNN for a single fold.
    """
    print(f"\n=== Training Fold {fold_idx} ===")

    # Reproducibility
    set_seed(SEED)
    device = get_device()

    # Data Loaders
    train_loader, val_loader = get_loaders(fold_idx, batch_size=32)

    # Model Setup
    model = RTICNN().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for images, angles, labels in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, angles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = train_loss_sum / train_samples

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                labels = labels.to(device)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_samples += images.size(0)

        avg_val_loss = val_loss_sum / val_samples

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}"
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss


def train_all_folds():
    """
    Sequentially trains all 5 folds.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    scores = []
    for fold in range(NUM_FOLDS):
        score = train_fold(fold)
        scores.append(score)

    print("\n=== Cross-Validation Complete ===")
    print(f"Fold Scores: {scores}")
    print(f"Average Val Loss: {np.mean(scores)}")


# -----------------------------------------------------------------------------
# 4. Inference & Submission
# -----------------------------------------------------------------------------
def generate_submission():
    """
    Generates submission file by averaging predictions from all 5 fold models.
    """
    print("\n=== Generating Submission ===")
    device = get_device()

    # Load Test Data
    test_loader, test_ids = get_test_loader(batch_size=32)

    # Initialize array for accumulated probabilities
    # Shape: (N_test, )
    avg_preds = np.zeros(len(test_ids), dtype=np.float32)

    # Iterate over each fold model
    for fold in range(NUM_FOLDS):
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint {model_path} not found. Skipping.")
            continue

        print(f"Loading model from {model_path}...")
        model = RTICNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        avg_preds += fold_preds

    # Average over folds
    avg_preds /= NUM_FOLDS

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def train_and_submit():
    """
    Helper function to run the full pipeline.
    """
    train_all_folds()
    generate_submission()
