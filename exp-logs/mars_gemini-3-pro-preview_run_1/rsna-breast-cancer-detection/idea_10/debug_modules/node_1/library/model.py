import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
import os

from library.config import Config
from library.utils import probabilistic_f1, seed_everything
from library.data import get_dataloaders


class PyramidSymmetryDifferenceModel(nn.Module):
    """
    Pyramid Symmetry-Difference Siamese Network.

    Architecture:
    1. Siamese EfficientNet-B2 Backbone (Shared Weights).
    2. Multi-Scale Feature Extraction at P3 (Stride 8), P4 (Stride 16), P5 (Stride 32).
    3. Difference Module: Computes (Target - Contralateral) at each scale.
    4. Fusion: Concatenates Global Average Pooled features from both Target and Difference maps.
    5. Classification Head.
    """

    def __init__(self):
        super().__init__()

        # We target P3, P4, P5 features for texture, pattern, and global structure.
        # In timm's EfficientNet implementation, these correspond to indices 2, 3, and 4.
        # Index 0: Stride 2, Index 1: Stride 4, Index 2: Stride 8 (P3),
        # Index 3: Stride 16 (P4), Index 4: Stride 32 (P5).
        target_indices = (2, 3, 4)

        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=target_indices,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts for the selected levels
        feature_channels = self.backbone.feature_info.channels()

        # Calculate input dimension for the classifier head
        # For each level i, we extract GAP(Target_i) and GAP(Difference_i).
        # Total dimension = Sum(2 * Channels_i) for all i.
        total_features = sum([c * 2 for c in feature_channels])

        self.classifier = nn.Linear(total_features, 1)

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: Tensor (B, 3, H, W) - Target Image + Age + Implant
            x_contra: Tensor (B, 3, H, W) - Contralateral Image + Age + Implant
        """
        # Extract features for target and contralateral images
        # Returns list of tensors [P3, P4, P5]
        feats_t = self.backbone(x_target)
        feats_c = self.backbone(x_contra)

        pooled_vectors = []

        for ft, fc in zip(feats_t, feats_c):
            # ft, fc: (B, C, H, W)

            # 1. Compute Signed Feature Difference
            # (Age/Implant channels in input are identical, so they cancel out here,
            # removing demographic bias from the difference signal)
            diff = ft - fc

            # 2. Global Average Pooling (Spatial mean)
            gap_t = ft.mean(dim=(2, 3))
            gap_d = diff.mean(dim=(2, 3))

            pooled_vectors.append(gap_t)
            pooled_vectors.append(gap_d)

        # 3. Concatenate all vectors
        # Shape: (B, Sum(2*C_i))
        concat = torch.cat(pooled_vectors, dim=1)

        # 4. Classification
        logits = self.classifier(concat)

        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs training for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move inputs to device
        img_target = batch["image"].to(device)
        img_contra = batch["contra_image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(img_target, img_contra)

        # Loss calculation
        loss = criterion(logits, labels)

        # Backward pass (No gradient clipping as per strategy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img_target.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Runs validation and calculates Loss and pF1.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            img_target = batch["image"].to(device)
            img_contra = batch["contra_image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(img_target, img_contra)
            loss = criterion(logits, labels)

            total_loss += loss.item() * img_target.size(0)

            # Store probabilities and targets for metric calculation
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(loader.dataset)
    pf1 = probabilistic_f1(all_targets, all_preds)

    return avg_loss, pf1


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_target = batch["image"].to(device)
            img_contra = batch["contra_image"].to(device)
            prediction_ids = batch[Config.ID_COL]

            logits = model(img_target, img_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({Config.ID_COL: pid, Config.TARGET_COL: prob})

    return pd.DataFrame(results)


def run_workflow(epochs=Config.NUM_EPOCHS, debug=Config.DEBUG):
    """
    Orchestrates the entire training and inference pipeline.
    """
    print("Initializing Workflow...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Loading
    print("Loading Data...")
    # Uses caching mechanism implemented in library.data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 2. Model Setup
    print("Building Model...")
    model = PyramidSymmetryDifferenceModel().to(device)

    # 3. Optimization Setup
    # Aggressive positive weighting for class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val pF1: {val_pf1:.6f}"
        )

        # Save Best Model (Early Stopping Check)
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Model Saved! (pF1: {best_pf1:.6f})")

    # 5. Inference
    print("Starting Inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found, using last epoch weights.")

    df_preds = predict_test(model, test_loader, device)

    # 6. Aggregation and Submission
    # Group by prediction_id (breast level) and take the Max probability across views (CC/MLO)
    submission = df_preds.groupby(Config.ID_COL)[Config.TARGET_COL].max().reset_index()

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
