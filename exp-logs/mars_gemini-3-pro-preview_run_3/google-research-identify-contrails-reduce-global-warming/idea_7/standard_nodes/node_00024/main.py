import os
import sys
import glob
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloader
from library.model import ContrailModel
from library.loss import ContrailLoss
from library.engine import train_one_epoch, validate, CheckpointManager
from library.utils import seed_everything, average_checkpoints, dice_coef, rle_encode


def main():
    # --- 1. Setup ---
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Override Config for fast baseline execution within 2 hours if necessary
    # The task asks for 40 epochs, but to ensure safety within 2h limit for a baseline,
    # we might need to be careful. However, we will attempt the configuration.
    # We will ensure directories exist (handled by Config, but good to double check)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PREDICTION_DIR, exist_ok=True)

    # Clear existing checkpoints to prevent loading incompatible weights from previous runs
    for f in glob.glob(os.path.join(Config.CHECKPOINT_DIR, "*.pth")):
        os.remove(f)

    # --- 2. Data Loading ---
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )
    val_loader = get_dataloader(
        split="validation", batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # --- 3. Model & Optimization ---
    print("Initializing Model...")
    model = ContrailModel()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    criterion = ContrailLoss(smooth=Config.SMOOTH)
    scaler = GradScaler()

    checkpoint_manager = CheckpointManager(
        Config.CHECKPOINT_DIR, top_k=Config.TOP_K_CHECKPOINTS
    )

    # --- 4. Training Loop ---
    print(f"Starting training for {Config.EPOCHS} epochs...")

    best_dice = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Save Checkpoint
        saved_path = checkpoint_manager.save(model, optimizer, epoch, val_dice)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if saved_path:
            print(f"  Saved checkpoint: {os.path.basename(saved_path)}")

    # --- 5. Model Averaging ---
    print("\n--- Model Averaging ---")
    # Get all checkpoints saved
    all_checkpoints = glob.glob(os.path.join(Config.CHECKPOINT_DIR, "*.pth"))

    # Filter checkpoints: Epoch > START_AVERAGING_EPOCH (20)
    # Filename format: checkpoint_epoch_{epoch}_dice_{score}.pth
    valid_checkpoints = []
    for ckpt in all_checkpoints:
        try:
            basename = os.path.basename(ckpt)
            parts = basename.split("_")
            # parts: ['checkpoint', 'epoch', '12', 'dice', '0.555.pth']
            epoch_num = int(parts[2])
            score_str = parts[4].replace(".pth", "")
            score = float(score_str)

            if epoch_num > Config.START_AVERAGING_EPOCH:
                valid_checkpoints.append(
                    {"path": ckpt, "score": score, "epoch": epoch_num}
                )
        except Exception as e:
            continue

    # Sort by score descending and take top K
    valid_checkpoints.sort(key=lambda x: x["score"], reverse=True)
    top_checkpoints = valid_checkpoints[: Config.TOP_K_CHECKPOINTS]
    checkpoint_paths = [x["path"] for x in top_checkpoints]

    if not checkpoint_paths:
        print(
            "No checkpoints found meeting averaging criteria. Using best available checkpoint."
        )
        if checkpoint_manager.checkpoints:
            # Sort manager checkpoints by score
            checkpoint_manager.checkpoints.sort(key=lambda x: x["score"], reverse=True)
            checkpoint_paths = [checkpoint_manager.checkpoints[0]["path"]]
        else:
            print("No checkpoints saved at all.")
            return

    print(
        f"Averaging {len(checkpoint_paths)} checkpoints: {[os.path.basename(p) for p in checkpoint_paths]}"
    )
    avg_state_dict = average_checkpoints(checkpoint_paths, device=device)
    model.load_state_dict(avg_state_dict)

    # --- 6. Final Validation & Failure Analysis ---
    print("\n--- Final Validation & Failure Analysis ---")
    model.eval()

    # We need to compute global dice and also per-image dice for failure analysis
    # To do this efficiently, we'll iterate and store stats

    val_meta_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    if Config.DEBUG:
        val_meta_df = val_meta_df.sample(
            n=min(len(val_meta_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Ensure the loader matches the dataframe order (dataset does not shuffle validation)
    # We will accumulate individual dice scores

    individual_dices = []
    intersection_sum = 0.0
    union_sum = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Global Dice Accumulation
            # Flatten batch
            p_flat = probs.view(-1)
            t_flat = masks.view(-1)
            intersection_sum += (p_flat * t_flat).sum().item()
            union_sum += (p_flat.sum() + t_flat.sum()).item()

            # Per-image Dice for Failure Analysis
            # Shape: (B, 1, H, W)
            B = images.size(0)
            for b in range(B):
                p_b = probs[b].view(-1)
                t_b = masks[b].view(-1)
                inter = (p_b * t_b).sum().item()
                union = p_b.sum().item() + t_b.sum().item()
                d = (2.0 * inter + Config.SMOOTH) / (union + Config.SMOOTH)
                individual_dices.append(d)

    final_metric = (2.0 * intersection_sum + Config.SMOOTH) / (
        union_sum + Config.SMOOTH
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Error Magnitude (1 - Dice)
    errors = [1.0 - d for d in individual_dices]

    # Add to dataframe
    # Note: The loader and dataframe must be aligned. The dataset class is deterministic.
    if len(errors) != len(val_meta_df):
        print(
            f"Warning: Mismatch in validation samples ({len(errors)}) and metadata ({len(val_meta_df)}). Skipping correlation analysis."
        )
    else:
        val_meta_df["error"] = errors

        # Calculate correlations
        # Features: timestamp, row_min, col_min
        features = ["timestamp", "row_min", "col_min"]
        print("\nCorrelation between Error (1-Dice) and Metadata:")
        for feat in features:
            if feat in val_meta_df.columns:
                corr = val_meta_df["error"].corr(val_meta_df[feat])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not found in metadata")

    # --- 7. Submission ---
    THRESHOLD = 0.6272749392944963

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = get_dataloader(
            split="test", batch_size=Config.BATCH_SIZE, debug=False
        )
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

        submission_records = []
        encoded_pixels_list = []

        # We need to match record_ids. The loader returns batches.
        # We will iterate and assume order is preserved (it is).

        batch_idx = 0
        current_record_idx = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # TTA: Original
                logits = model(images)
                probs = torch.sigmoid(logits)

                if Config.USE_TTA:
                    # Flip Horizontal (dim 3)
                    images_h = torch.flip(images, dims=[3])
                    logits_h = model(images_h)
                    probs_h = torch.sigmoid(logits_h)
                    probs += torch.flip(probs_h, dims=[3])

                    # Flip Vertical (dim 2)
                    images_v = torch.flip(images, dims=[2])
                    logits_v = model(images_v)
                    probs_v = torch.sigmoid(logits_v)
                    probs += torch.flip(probs_v, dims=[2])

                    probs /= 3.0

                # Threshold
                preds = (probs > 0.5).float().cpu().numpy()

                # Encode
                for b in range(preds.shape[0]):
                    # preds shape (B, 1, H, W) -> (H, W)
                    mask = preds[b, 0, :, :]

                    if mask.sum() == 0:
                        rle = "-"
                    else:
                        rle = rle_encode(mask)

                    encoded_pixels_list.append(rle)

                    # Get record_id
                    rec_id = test_meta_df.iloc[current_record_idx]["record_id"]
                    submission_records.append(rec_id)
                    current_record_idx += 1

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"record_id": submission_records, "encoded_pixels": encoded_pixels_list}
        )

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
