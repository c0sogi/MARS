import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import SETIDataset
from library.utils import (
    seed_everything,
    mixup_data,
    mixup_criterion,
    get_score,
    apply_tta,
)


class SiameseModel(nn.Module):
    """
    Siamese Network with Hybrid Pooling and Explicit Difference.
    Uses a BN-based backbone (EfficientNet) for better stability on spectrograms.

    Architecture:
    1. Backbone: EfficientNet-B0 (Pretrained ImageNet-1k), shared weights.
    2. Input: 6-channel spectrogram.
       - Stream A (Signal): Channels 0, 2, 4 (On-Target)
       - Stream B (Reference): Channels 1, 3, 5 (Off-Target)
    3. Feature Extraction: Last stage feature maps.
    4. Interaction: Explicit Difference (Signal - Reference).
    5. Pooling: GAP + GMP on Signal, Reference, and Difference maps.
    6. Head: Linear layer on concatenated features.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(SiameseModel, self).__init__()

        # Initialize backbone
        # in_chans=3 because we process each stream (3 channels) independently
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,  # Remove default head
            global_pool="",  # Return spatial feature maps
        )

        # Determine feature dimension
        # EfficientNet-B0 usually has 1280 channels at the final stage
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features shape: (B, C, H, W)
            self.num_features = features.shape[1]

        # Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # Inputs:
        #   - F_on (GAP, GMP) -> 2 * C
        #   - F_off (GAP, GMP) -> 2 * C
        #   - F_diff (GAP, GMP) -> 2 * C
        # Total: 6 * C
        self.fc = nn.Linear(self.num_features * 6, 1)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        # x shape: (B, 6, H, W)

        # Split into On-Target (A) and Off-Target (B) streams
        # A: Indices 0, 2, 4
        # B: Indices 1, 3, 5
        x_on = x[:, [0, 2, 4], :, :]
        x_off = x[:, [1, 3, 5], :, :]

        # Shared Backbone Extraction
        f_on = self.forward_features(x_on)  # (B, C, H', W')
        f_off = self.forward_features(x_off)  # (B, C, H', W')

        # Explicit Difference
        f_diff = f_on - f_off

        # Hybrid Pooling (Flattening spatial dims)
        # On-Target
        ap_on = self.avg_pool(f_on).flatten(1)
        mp_on = self.max_pool(f_on).flatten(1)

        # Off-Target
        ap_off = self.avg_pool(f_off).flatten(1)
        mp_off = self.max_pool(f_off).flatten(1)

        # Difference
        ap_diff = self.avg_pool(f_diff).flatten(1)
        mp_diff = self.max_pool(f_diff).flatten(1)

        # Concatenate all features
        features = torch.cat([ap_on, mp_on, ap_off, mp_off, ap_diff, mp_diff], dim=1)

        # Classification
        out = self.fc(features)

        return out


def train_model():
    """
    Executes the training pipeline with Early Stopping and Mixup.
    """
    seed_everything(Config.SEED)

    # --- Data Loading ---
    train_dataset = SETIDataset(mode="train")
    val_dataset = SETIDataset(mode="val")

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

    # --- Model Setup ---
    device = Config.DEVICE
    model = SiameseModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=1e-6
    )

    # --- Training Loop ---
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss_sum = 0.0

        # Training Step
        for images, targets in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]",
            leave=False,
        ):
            images = images.to(device)
            targets = targets.to(device)

            # Mixup
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.MIXUP_ALPHA, device
            )

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        avg_train_loss = train_loss_sum / len(train_dataset)

        # Validation Step
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]",
                leave=False,
            ):
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * images.size(0)

                # Sigmoid for AUC calculation
                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_dataset)
        val_auc = get_score(np.array(val_targets), np.array(val_preds))

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> Model Saved! New Best AUC: {best_val_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training completed. Best Validation AUC: {best_val_auc}")


def predict_and_submit():
    """
    Loads the best model, performs TTA inference on the test set,
    and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Data
    test_dataset = SETIDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = SiameseModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model weights from {Config.MODEL_PATH}")
    else:
        print(
            "Warning: No trained model found. Using random weights (expect poor performance)."
        )

    model.eval()

    # Inference
    all_preds = []
    ids = test_dataset.df["id"].tolist()

    print("Starting inference with TTA...")
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            # apply_tta handles moving to device and sigmoid
            avg_probs = apply_tta(model, images, device)
            all_preds.extend(avg_probs.cpu().numpy())

    # Create Submission
    submission = pd.DataFrame({"id": ids, "target": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
