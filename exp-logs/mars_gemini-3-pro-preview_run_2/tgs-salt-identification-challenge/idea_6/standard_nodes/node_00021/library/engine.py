import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from library.utils import calc_map
from library.models import DepthRegressor, DepthAwareLinkNet34

# Constants
CHECKPOINT_DIR = "./working/idea_6/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE Loss
        bce = self.bce_loss(pred, target)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))

        return self.bce_weight * bce + self.dice_weight * dice


# -----------------------------------------------------------------------------
# Regression Engine (Stage 1)
# -----------------------------------------------------------------------------


def train_regressor_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        images = batch[0]
        depths = batch[-2]

        images = images.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_regressor_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            depths = batch[-2]

            images = images.to(device)
            depths = depths.to(device)

            outputs = model(images)
            loss = criterion(outputs, depths)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def run_regression_training(train_loader, val_loader, epochs=20, lr=1e-4, patience=5):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(42)

    model = DepthRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_regressor.pth")

    print("Starting Regression Training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_regressor_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate_regressor_epoch(model, val_loader, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {train_loss} | "
            f"Val MSE: {val_loss} | "
            f"Time: {time.time() - start_time:.2f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> Saved best regressor model (Loss: {best_val_loss})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

    print("Regression Training Complete.")
    return best_model_path


def predict_depths(model_path, loader):
    """
    Generates depth predictions using the trained regressor.
    """
    model = DepthRegressor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    predictions = []

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())

    return np.concatenate(predictions, axis=0)


# -----------------------------------------------------------------------------
# Segmentation Engine (Stage 2)
# -----------------------------------------------------------------------------


def train_segmenter_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch (dataset returns img, mask, depth, id)
        images, masks, depths, _ = batch

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass with depth injection
        outputs = model(images, depths)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_segmenter_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images, masks, depths, _ = batch

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            outputs = model(images, depths)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Prepare for mAP calculation
            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Convert to numpy arrays
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            probs_np = probs.squeeze(1).cpu().numpy()
            masks_np = masks.squeeze(1).cpu().numpy()

            for p, t in zip(probs_np, masks_np):
                all_preds.append(p)
                all_targets.append(t)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate mAP
    epoch_map = calc_map(all_preds, all_targets)

    return epoch_loss, epoch_map


def run_segmentation_training(
    train_loader, val_loader, epochs=50, lr=1e-3, patience=10
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(42)

    model = DepthAwareLinkNet34(num_classes=1).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_map = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_segmenter.pth")

    print("Starting Segmentation Training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_segmenter_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_map = validate_segmenter_epoch(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val mAP: {val_map} | "
            f"Time: {time.time() - start_time:.2f}s"
        )

        # Checkpointing based on mAP
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> Saved best segmenter model (mAP: {best_val_map})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

    print("Segmentation Training Complete.")
    return best_model_path


def predict_segmentation(model_path, loader, imputed_depths=None):
    """
    Generates segmentation predictions.
    Args:
        model_path: Path to trained segmenter.
        loader: DataLoader for test set.
        imputed_depths: Optional numpy array of shape (N, 1) with imputed depths.
                        If provided, overrides the depths from the loader.
    """
    model = DepthAwareLinkNet34(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []

    # If using imputed depths, we need to iterate them alongside the loader
    # Assuming loader is sequential and not shuffled

    depth_iter = None
    if imputed_depths is not None:
        depth_iter = iter(imputed_depths)

    with torch.no_grad():
        for batch in loader:
            # Test loader returns: image, depth, id
            images, depths, _ = batch
            images = images.to(DEVICE)

            if depth_iter is not None:
                # Replace batch depths with imputed depths
                batch_size = images.size(0)
                batch_imputed = []
                for _ in range(batch_size):
                    try:
                        batch_imputed.append(next(depth_iter))
                    except StopIteration:
                        break

                if len(batch_imputed) > 0:
                    depths = torch.tensor(np.array(batch_imputed), dtype=torch.float32)

            depths = depths.to(DEVICE)

            # TTA: Original + Horizontal Flip
            # 1. Forward Original
            out_orig = model(images, depths)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Forward Flip
            images_flipped = torch.flip(images, dims=[3])
            out_flipped = model(images_flipped, depths)
            prob_flipped = torch.sigmoid(out_flipped)
            prob_flipped = torch.flip(prob_flipped, dims=[3])

            # Average
            prob_avg = (prob_orig + prob_flipped) / 2.0

            # Store results
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            preds_np = prob_avg.squeeze(1).cpu().numpy()

            # Center crop from 128x128 back to 101x101
            # Padding was symmetric reflection.
            # 128 - 101 = 27. 13 on one side, 14 on other?
            # Albumentations PadIfNeeded centers the image.
            # Target 128, Source 101. (128-101)/2 = 13.5.
            # Usually Albumentations puts 13 top/left, 14 bottom/right or vice versa.
            # Let's assume center crop is safe.

            h_start = (128 - 101) // 2
            w_start = (128 - 101) // 2
            h_end = h_start + 101
            w_end = w_start + 101

            preds_cropped = preds_np[:, h_start:h_end, w_start:w_end]

            for p in preds_cropped:
                all_preds.append(p)

    return np.array(all_preds)
