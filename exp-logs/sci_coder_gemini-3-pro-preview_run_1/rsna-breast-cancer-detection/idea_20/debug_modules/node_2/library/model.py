import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, probabilistic_f1, save_checkpoint
from library.data import get_dataloaders

# Initialize Logger
logger = get_logger("model")


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Used to re-weight channels in the difference map to suppress noise from misalignment.
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


class SiameseEfficientNet(nn.Module):
    """
    Channel-Attentive Symmetry-Difference Siamese Network.
    Backbone: EfficientNet-B2
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Load backbone with features_only=True to access intermediate layers
        # out_indices=(2, 3, 4) corresponds to P3, P4, P5 (strides 8, 16, 32)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts for the selected feature maps
        # For EfficientNet-B2, typically [48, 120, 352]
        feature_channels = self.backbone.feature_info.channels()

        # Create SE Blocks for each feature level
        self.se_blocks = nn.ModuleList([SEBlock(c) for c in feature_channels])

        # Calculate total input dimension for the classifier
        # For each level: Concat(Target, Attended_Diff) -> 2 * C
        # Total = Sum(2 * C for all levels)
        total_channels = sum([c * 2 for c in feature_channels])

        self.classifier = nn.Linear(total_channels, 1)

    def forward_features(self, x):
        """Passes one image through the backbone."""
        return self.backbone(x)

    def forward(self, target, contra):
        """
        Args:
            target (torch.Tensor): Target breast images (B, C, H, W)
            contra (torch.Tensor): Contralateral breast images (B, C, H, W)
        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # Extract multi-scale features
        # feats_t and feats_c are lists of tensors [P3, P4, P5]
        feats_t = self.forward_features(target)
        feats_c = self.forward_features(contra)

        pooled_features = []

        # Process each scale
        for i, (ft, fc) in enumerate(zip(feats_t, feats_c)):
            # 1. Compute Signed Difference
            diff = ft - fc

            # 2. Apply Channel Attention (Noise Suppression)
            diff_att = self.se_blocks[i](diff)

            # 3. Concatenate Target Features and Attended Difference
            combined = torch.cat([ft, diff_att], dim=1)

            # 4. Global Average Pooling
            # (B, 2C, H, W) -> (B, 2C)
            pooled = combined.mean(dim=(2, 3))
            pooled_features.append(pooled)

        # 5. Concatenate pooled features from all scales
        global_feature = torch.cat(pooled_features, dim=1)

        # 6. Classification
        logits = self.classifier(global_feature)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Unpack inputs
        img_target = inputs["target"].to(device, non_blocking=True)
        img_contra = inputs["contra"].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        # Forward
        optimizer.zero_grad()
        logits = model(img_target, img_contra)

        # Loss
        loss = criterion(logits, targets)

        # Backward
        loss.backward()

        # No gradient clipping as per instructions
        optimizer.step()

        # Metrics
        running_loss += loss.item() * targets.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    epoch_pf1 = probabilistic_f1(all_targets, all_probs)

    return epoch_loss, epoch_pf1


def validate(model, loader, criterion, device):
    """
    Validates the model.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for inputs, targets in pbar:
            img_target = inputs["target"].to(device, non_blocking=True)
            img_contra = inputs["contra"].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            logits = model(img_target, img_contra)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    epoch_pf1 = probabilistic_f1(all_targets, all_probs)

    return epoch_loss, epoch_pf1


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    Aggregates predictions by prediction_id (Max aggregation).
    """
    model.eval()
    results = []

    logger.info("Starting inference...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference", leave=False)
        for inputs, prediction_ids in pbar:
            img_target = inputs["target"].to(device, non_blocking=True)
            img_contra = inputs["contra"].to(device, non_blocking=True)

            logits = model(img_target, img_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Aggregate by prediction_id (Max probability across views)
    df_sub = df_results.groupby("prediction_id")["cancer"].max().reset_index()

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {df_sub.shape}")

    return df_sub


def run_training(debug=False):
    """
    Main execution pipeline: Training -> Validation -> Inference.
    """
    # 1. Setup
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 3. Model
    model = SiameseEfficientNet().to(device)

    # 4. Optimizer & Loss
    # Weighted BCE Loss for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Train
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"  Train Loss: {train_loss:.6f} | Train pF1: {train_pf1:.6f}")

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)
        logger.info(f"  Val Loss:   {val_loss:.6f} | Val pF1:   {val_pf1:.6f}")

        # Scheduler Step
        scheduler.step()

        # Checkpointing & Early Stopping
        is_best = val_pf1 > best_pf1
        if is_best:
            best_pf1 = val_pf1
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_pf1": best_pf1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            logger.info(f"  New Best Model! pF1: {best_pf1:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            logger.info(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    # 6. Inference
    logger.info("Loading best model for inference...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        logger.warning("Best model not found, using current model weights.")

    inference(model, test_loader, device)


def run_pipeline():
    """Wrapper to run the full pipeline."""
    run_training(debug=Config.DEBUG)
