import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
import os
import sys

from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders


class SymmetryDifferenceSiameseNet(nn.Module):
    """
    Spatial Symmetry-Difference Siamese Network with EfficientNet-B2 backbone.

    Architecture:
    1. Shared Backbone: EfficientNet-B2 (pretrained) extracting spatial features.
    2. Fusion: Spatial Difference (Target - Contralateral) + Concatenation.
    3. Head: Depthwise Separable Conv -> GAP -> Dense -> Logit.
    """

    def __init__(self):
        super(SymmetryDifferenceSiameseNet, self).__init__()

        # Load pretrained backbone
        # num_classes=0 and global_pool='' ensures we get feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine backbone output channels dynamically
        # EfficientNet-B2 usually has 1408 channels at the final conv layer
        with torch.no_grad():
            dummy = torch.zeros(1, Config.IN_CHANNELS, 256, 256)
            features = self.backbone(dummy)
            in_channels = features.shape[1]

        # Fusion results in 2x channels (Target + Difference)
        fusion_channels = in_channels * 2
        hidden_dim = 512

        # Depthwise Separable Convolution Block
        self.head_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(fusion_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_pair):
        """
        Args:
            x_pair: Tuple of (target_image, contralateral_image)
        """
        x_target, x_contra = x_pair

        # Shared Backbone Feature Extraction
        # Shape: (B, C, H', W')
        f_target = self.backbone(x_target)
        f_contra = self.backbone(x_contra)

        # Spatial Difference
        f_diff = f_target - f_contra

        # Concatenation (Early Fusion strategy)
        # Shape: (B, 2*C, H', W')
        f_fused = torch.cat([f_target, f_diff], dim=1)

        # Head Processing
        x = self.head_conv(f_fused)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        # Unpack inputs (tuple of tensors) and move to device
        target_img = inputs[0].to(device, non_blocking=True)
        contra_img = inputs[1].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model((target_img, contra_img))

        # Loss calculation
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # No Gradient Clipping as per instructions

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            target_img = inputs[0].to(device, non_blocking=True)
            contra_img = inputs[1].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model((target_img, contra_img))
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * labels.size(0)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / len(loader.dataset)
    pf1 = probabilistic_f1(all_labels, all_preds)

    return epoch_loss, pf1


def train_model():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 2. Model
    print("Initializing Model...")
    model = SymmetryDifferenceSiameseNet().to(device)

    # 3. Optimization
    # Pos weight for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 4. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_pf1:.10f}"
        )

        # Save best model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best pF1! Model saved to {best_model_path}")

    print(f"Training complete. Best Validation pF1: {best_pf1:.10f}")
    return best_model_path


def predict_and_submit(model_path):
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    print("Initializing Test DataLoader...")
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model
    print(f"Loading model from {model_path}...")
    model = SymmetryDifferenceSiameseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    print("Running Inference...")
    results = []

    with torch.no_grad():
        for inputs, prediction_ids in test_loader:
            target_img = inputs[0].to(device, non_blocking=True)
            contra_img = inputs[1].to(device, non_blocking=True)

            logits = model((target_img, contra_img))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # prediction_ids is a tuple/list from the loader
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    df_results = pd.DataFrame(results)

    # 4. Aggregation
    # Group by prediction_id and take MAX probability (as per idea description)
    # This handles cases where one prediction_id maps to multiple images (e.g. different views)
    print("Aggregating predictions...")
    submission = df_results.groupby("prediction_id")["cancer"].max().reset_index()

    # 5. Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


# Entry point functions for external calls
def run_training_pipeline():
    best_path = train_model()
    predict_and_submit(best_path)
