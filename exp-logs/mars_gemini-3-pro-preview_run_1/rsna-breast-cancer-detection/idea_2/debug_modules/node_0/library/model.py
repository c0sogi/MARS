import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders

# =========================================================================
# Model Architecture
# =========================================================================


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(MultiTaskEfficientNet, self).__init__()

        # Initialize backbone as feature extractor
        # in_chans=3 matches our input (Image, Age, Implant)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Get the number of features output by the backbone
        n_features = self.backbone.num_features

        # Primary Head: Cancer Detection (Binary)
        self.cancer_head = nn.Linear(n_features, 1)

        # Auxiliary Head: Density Classification (4 classes)
        self.density_head = nn.Linear(n_features, Config.NUM_AUX_CLASSES)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Multi-task outputs
        cancer_logits = self.cancer_head(features)
        density_logits = self.density_head(features)

        return cancer_logits, density_logits


# =========================================================================
# Metrics
# =========================================================================


def probabilistic_f1(probabilities, targets, beta=1):
    """
    Calculates the Probabilistic F1 score.

    Args:
        probabilities (np.ndarray): Predicted probabilities (0-1).
        targets (np.ndarray): Binary targets (0 or 1).
        beta (float): Weight of recall in F-score.

    Returns:
        float: pF1 score.
    """
    # Avoid division by zero
    epsilon = 1e-7

    # pTP = Sum(p * y)
    p_tp = np.sum(probabilities * targets)

    # pPrecision = pTP / Sum(p)
    p_precision = p_tp / (np.sum(probabilities) + epsilon)

    # pRecall = pTP / Sum(y)
    p_recall = p_tp / (np.sum(targets) + epsilon)

    # F1 formula
    f1 = (
        (1 + beta**2)
        * (p_precision * p_recall)
        / ((beta**2 * p_precision) + p_recall + epsilon)
    )

    return f1


# =========================================================================
# Training & Validation
# =========================================================================


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_loss_meter = 0.0
    cancer_loss_meter = 0.0
    density_loss_meter = 0.0

    # Loss functions
    # Weighted BCE for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    cancer_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # CrossEntropy for density (ignore -1 for missing labels)
    density_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for batch_idx, (inputs, cancer_targets, density_targets) in enumerate(loader):
        inputs = inputs.to(device)
        cancer_targets = cancer_targets.to(device).unsqueeze(1)  # [B, 1]
        density_targets = density_targets.to(device)  # [B]

        optimizer.zero_grad()

        # Forward
        cancer_logits, density_logits = model(inputs)

        # Compute Losses
        loss_c = cancer_criterion(cancer_logits, cancer_targets)
        loss_d = density_criterion(density_logits, density_targets)

        # Weighted Sum
        loss = loss_c + Config.AUX_WEIGHT * loss_d

        # Backward
        loss.backward()

        # No gradient clipping as per lesson 00009

        optimizer.step()

        # Update meters
        total_loss_meter += loss.item()
        cancer_loss_meter += loss_c.item()
        density_loss_meter += loss_d.item()

    avg_loss = total_loss_meter / len(loader)
    avg_cancer_loss = cancer_loss_meter / len(loader)
    avg_density_loss = density_loss_meter / len(loader)

    print(
        f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} (Cancer: {avg_cancer_loss:.4f}, Density: {avg_density_loss:.4f})"
    )
    return avg_loss


def validate(model, loader, device):
    model.eval()

    all_cancer_probs = []
    all_cancer_targets = []

    all_density_preds = []
    all_density_targets = []

    with torch.no_grad():
        for inputs, cancer_targets, density_targets in loader:
            inputs = inputs.to(device)

            cancer_logits, density_logits = model(inputs)

            # Cancer Probs
            probs = torch.sigmoid(cancer_logits).cpu().numpy().flatten()
            all_cancer_probs.extend(probs)
            all_cancer_targets.extend(cancer_targets.numpy())

            # Density Preds
            preds_d = torch.argmax(density_logits, dim=1).cpu().numpy()
            all_density_preds.extend(preds_d)
            all_density_targets.extend(density_targets.numpy())

    # Metrics
    all_cancer_probs = np.array(all_cancer_probs)
    all_cancer_targets = np.array(all_cancer_targets)

    pf1 = probabilistic_f1(all_cancer_probs, all_cancer_targets)

    # Filter density targets for accuracy (remove -1)
    all_density_preds = np.array(all_density_preds)
    all_density_targets = np.array(all_density_targets)
    valid_mask = all_density_targets != -1
    if valid_mask.sum() > 0:
        density_acc = accuracy_score(
            all_density_targets[valid_mask], all_density_preds[valid_mask]
        )
    else:
        density_acc = 0.0

    print(f"Validation | pF1: {pf1:.8f} | Density Acc: {density_acc:.4f}")

    return pf1


# =========================================================================
# Inference
# =========================================================================


def predict_and_submit(model, loader, device, output_path):
    print("Starting Inference...")
    model.eval()

    predictions = []
    prediction_ids = []

    with torch.no_grad():
        for inputs, batch_ids in loader:
            inputs = inputs.to(device)

            cancer_logits, _ = model(inputs)
            probs = torch.sigmoid(cancer_logits).cpu().numpy().flatten()

            predictions.extend(probs)
            prediction_ids.extend(batch_ids)

    # Create DataFrame
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": predictions})

    # Aggregate by prediction_id (Max Pooling across views)
    # The submission format requires one row per prediction_id
    df_sub = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(df_sub.head())


# =========================================================================
# Main Runner
# =========================================================================


def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 1. Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 2. Model
    print(f"Initializing Model: {Config.BACKBONE}")
    model = MultiTaskEfficientNet(Config.BACKBONE, pretrained=True)
    model.to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 4. Training Loop
    best_pf1 = -1.0
    best_model_path = Config.MODEL_SAVE_PATH

    # Early Stopping Variables
    patience = 4
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_pf1 = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Save Best
        if val_pf1 > best_pf1:
            print(f"New Best pF1: {val_pf1:.8f} (was {best_pf1:.8f}). Saving model.")
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
