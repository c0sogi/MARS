import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr
import gc

# Import from provided library
from library.config import Config
from library.utils import set_seed, rle_encode, calculate_map, calculate_iou_batch
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.engine import train_one_epoch, evaluate


def get_inference_probs(model, dataloader, device):
    """
    Generates probability maps for a given model and dataloader using TTA.
    Un-pads the output to original 101x101 size.
    """
    model.eval()
    probs_list = []

    # Padding indices for un-padding (128x128 -> 101x101)
    # Pad H: (13, 14), Pad W: (13, 14)
    start_idx = 13
    end_idx = 128 - 14  # 114

    with torch.no_grad():
        for batch in dataloader:
            # Handle different dataloader returns (Test: img, id; Val: img, mask, id)
            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            images = images.to(device)

            # 1. Forward Pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. Forward Pass (Horizontal Flip TTA)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            # 3. Average Predictions
            avg_probs = (probs + probs_flipped_back) / 2.0

            # 4. Un-pad to original size (101x101)
            if avg_probs.dim() == 4:
                avg_probs = avg_probs[:, :, start_idx:end_idx, start_idx:end_idx]
                # Remove channel dim: (B, 1, H, W) -> (B, H, W)
                avg_probs = avg_probs.squeeze(1)
            elif avg_probs.dim() == 3:
                avg_probs = avg_probs[:, start_idx:end_idx, start_idx:end_idx]

            probs_list.append(avg_probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create directories if they don't exist (handled in Config.setup usually, but ensuring here)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached=True
    )

    # -------------------------------------------------------------------------
    # 3. Model & Optimization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = DeepResUNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing with Warm Restarts
    # T_0 = 50 epochs (Cycle length), T_mult = 1 (Constant cycle length)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.CYCLE_LENGTH, T_mult=1, eta_min=1e-6
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")

    best_map_cycle_2 = 0.0
    best_map_cycle_3 = 0.0

    path_cycle_2 = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth")
    path_cycle_3 = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Step scheduler
        scheduler.step()

        # Validate
        val_loss, val_map = evaluate(model, val_loader, device, epoch)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}"
        )

        # Checkpointing Logic
        # Cycle 2: Epochs 50 to 99 (Indices 50-99)
        if 50 <= epoch < 100:
            if val_map > best_map_cycle_2:
                best_map_cycle_2 = val_map
                torch.save(model.state_dict(), path_cycle_2)

        # Cycle 3: Epochs 100 to 149 (Indices 100-149)
        if epoch >= 100:
            if val_map > best_map_cycle_3:
                best_map_cycle_3 = val_map
                torch.save(model.state_dict(), path_cycle_3)

    # Free memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 5. Ensemble Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nLoading snapshot models for ensemble...")

    # Load Cycle 2 Model
    model_c2 = DeepResUNet().to(device)
    model_c2.load_state_dict(torch.load(path_cycle_2, map_location=device))

    # Load Cycle 3 Model
    model_c3 = DeepResUNet().to(device)
    model_c3.load_state_dict(torch.load(path_cycle_3, map_location=device))

    print("Generating validation predictions (Ensemble)...")
    # Get probabilities
    probs_c2 = get_inference_probs(model_c2, val_loader, device)
    probs_c3 = get_inference_probs(model_c3, val_loader, device)

    # Ensemble Average
    val_probs_ensemble = (probs_c2 + probs_c3) / 2.0

    # Get Ground Truth for Validation
    # We need to manually extract masks from loader to match order
    val_masks_list = []
    # Padding indices
    start_idx = 13
    end_idx = 114

    for _, masks, _ in val_loader:
        # Unpad masks
        masks_np = masks.numpy()
        if masks_np.ndim == 4:
            masks_cropped = masks_np[:, :, start_idx:end_idx, start_idx:end_idx]
            masks_cropped = masks_cropped.squeeze(1)
        else:
            masks_cropped = masks_np[:, start_idx:end_idx, start_idx:end_idx]
        val_masks_list.append(masks_cropped)

    val_targets = np.concatenate(val_masks_list, axis=0)

    # Calculate Final Validation Metric
    # calculate_map expects (N, H, W)
    final_val_metric = calculate_map(val_probs_ensemble, val_targets)
    print(f"Final Validation Metric: {final_val_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Calculate per-image mAP
    ious = calculate_iou_batch(val_probs_ensemble, val_targets)
    thresholds = np.arange(0.5, 1.0, 0.05)
    matches = ious[:, None] > thresholds[None, :]
    ap_per_image = matches.mean(axis=1)

    # Error metric: 1 - mAP
    errors = 1.0 - ap_per_image

    # Load Validation Metadata for correlation
    df_val = pd.read_csv(Config.VAL_CSV)
    if Config.DEBUG:
        df_val = df_val.iloc[: Config.DEBUG_SIZE]

    # Ensure alignment (DataLoader is shuffle=False)
    if len(df_val) != len(errors):
        print("Warning: Metadata length mismatch. Skipping correlation analysis.")
    else:
        # Extract features
        depths = df_val["z"].values
        coverages = df_val["coverage"].values

        # Calculate correlations
        corr_depth, _ = pearsonr(depths, errors)
        corr_cov, _ = pearsonr(coverages, errors)

        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.8156666666666668

    if final_val_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_val_metric}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Get Test Probabilities
        test_probs_c2 = get_inference_probs(model_c2, test_loader, device)
        test_probs_c3 = get_inference_probs(model_c3, test_loader, device)

        # Ensemble
        test_probs_ensemble = (test_probs_c2 + test_probs_c3) / 2.0

        # Threshold and Encode
        predictions = []
        ids_list = []

        # Get IDs from test loader
        # Re-iterate to get IDs (order is preserved)
        all_ids = []
        for _, ids in test_loader:
            all_ids.extend(ids)

        for i in range(len(test_probs_ensemble)):
            prob_map = test_probs_ensemble[i]
            img_id = all_ids[i]

            # Binary mask
            binary_mask = (prob_map > 0.5).astype(np.uint8)

            # Encode
            rle = rle_encode(binary_mask)

            ids_list.append(img_id)
            predictions.append(rle)

        # Save
        df_sub = pd.DataFrame({"id": ids_list, "rle_mask": predictions})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_val_metric}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
