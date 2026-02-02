import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coefficient
from library.architecture import GC_ConvNeXtUNet
from library.losses import HybridLoss
from library.data_loader import get_data_loaders


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    # Use tqdm for progress tracking but disable it for final submission if needed
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False)

    for images, metadata, masks in pbar:
        images = images.to(device, dtype=torch.float32)
        metadata = metadata.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass with metadata injection
        logits = model(images, metadata)

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()

    # Global Dice accumulators
    total_intersection = 0.0
    total_union = 0.0

    # For failure analysis
    sample_errors = []
    meta_features = []  # [lat, lon, time]

    with torch.no_grad():
        for images, metadata, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device, dtype=torch.float32)
            meta_gpu = metadata.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(images, meta_gpu)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Update Global Dice stats
            # Flatten batch
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

            # Per-sample analysis for failure analysis
            # We calculate dice per sample to correlate with metadata
            B = images.size(0)
            preds_B = preds.view(B, -1)
            masks_B = masks.view(B, -1)

            inter_B = (preds_B * masks_B).sum(dim=1)
            union_B = preds_B.sum(dim=1) + masks_B.sum(dim=1)

            # Dice per sample (smooth to avoid div by zero for analysis)
            dices = (2.0 * inter_B + 1e-6) / (union_B + 1e-6)
            errors = 1.0 - dices.cpu().numpy()

            sample_errors.extend(errors)
            meta_features.extend(metadata.numpy())

    # Compute Global Dice
    global_dice = (2.0 * total_intersection) / (total_union + 1e-6)

    return global_dice, np.array(sample_errors), np.array(meta_features)


def perform_failure_analysis(errors, metadata):
    """
    Correlates prediction error (1 - Dice) with metadata features.
    metadata columns: [norm_row (lat), norm_col (lon), time_of_day]
    """
    if len(errors) == 0:
        return

    # Metadata columns
    feature_names = ["Latitude (norm)", "Longitude (norm)", "Time of Day"]

    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)
    print(f"Analyzing {len(errors)} validation samples.")

    for i, name in enumerate(feature_names):
        feat_vals = metadata[:, i]
        # Compute Pearson correlation
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            print(f"Correlation Error vs {name}: {corr:.4f}")
        else:
            print(f"Correlation Error vs {name}: N/A (Constant values)")
    print("=" * 30 + "\n")


def predict_and_submit(model, loader, device, output_path):
    model.eval()
    submission_data = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for images, metadata, record_ids in tqdm(loader, desc="Inference"):
            images = images.to(device, dtype=torch.float32)
            metadata = metadata.to(device, dtype=torch.float32)

            # --- Test Time Augmentation (TTA) ---
            # 1. Original
            logits_1 = model(images, metadata)
            probs_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h, metadata)
            probs_2 = torch.flip(torch.sigmoid(logits_2), [3])

            # 3. Vertical Flip
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v, metadata)
            probs_3 = torch.flip(torch.sigmoid(logits_3), [2])

            # 4. Rotate 180
            images_r = torch.rot90(images, 2, [2, 3])
            logits_4 = model(images_r, metadata)
            probs_4 = torch.rot90(torch.sigmoid(logits_4), -2, [2, 3])

            # Average predictions
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Threshold
            preds = (avg_probs > 0.5).float().cpu().numpy()

            # Encode
            for i, record_id in enumerate(record_ids):
                # Shape (1, H, W) -> (H, W)
                mask = preds[i, 0, :, :]
                rle = rle_encode(mask)
                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Save submission
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline Execution
    # 6 epochs should be enough to get a decent score with pre-trained backbone
    Config.EPOCHS = 6
    Config.T_MAX = 6

    print(f"Running on {device}")
    print(f"Training for {Config.EPOCHS} epochs...")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = GC_ConvNeXtUNet(
        in_chans=Config.IN_CHANNELS,
        num_classes=1,
        metadata_dim=Config.METADATA_FEATURE_DIM,
    ).to(device)

    criterion = HybridLoss(bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # 4. Training Loop
    best_dice = 0.0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_dice, val_errors, val_meta = validate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.6f} | Time: {elapsed:.0f}s"
        )

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Validation & Failure Analysis
    print("\nLoading best model for final validation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    final_dice, final_errors, final_meta = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_dice}")

    # Failure Analysis
    perform_failure_analysis(final_errors, final_meta)

    # 6. Submission
    THRESHOLD = 0.5910660985501295

    if final_dice > THRESHOLD:
        print(f"Validation Metric {final_dice} > {THRESHOLD}. Generating submission...")
        predict_and_submit(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(f"Validation Metric {final_dice} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
