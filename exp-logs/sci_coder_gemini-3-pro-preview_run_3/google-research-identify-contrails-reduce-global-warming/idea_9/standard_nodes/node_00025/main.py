import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.dataset import ContrailDataset
from library.model import ContrailUNet
from library.engine import train_one_epoch, valid_one_epoch
from library.utils import seed_everything
from library.inference import make_submission


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(Config.SEED)

    # Configure for Fast Baseline to meet time constraints
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32

    # Define limits to ensure speed
    TRAIN_SAMPLES = 4000
    VAL_SAMPLES = 1000

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # --- 2. Data Loading ---
    # Manually create datasets to control sample size precisely for the baseline run
    train_dataset = ContrailDataset(
        split="train",
        max_samples=TRAIN_SAMPLES,
        debug=False,  # We control size via max_samples
    )

    val_dataset = ContrailDataset(
        split="validation", max_samples=VAL_SAMPLES, debug=False
    )

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
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- 3. Model Initialization ---
    model = ContrailUNet().to(device)

    # --- 4. Optimization ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # --- 5. Training Loop ---
    best_checkpoints = []  # List of (dice, path)

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, val_dice = valid_one_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        # Checkpointing
        ckpt_path = os.path.join(
            Config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}_dice_{val_dice:.6f}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)

        # Maintain Top-K
        best_checkpoints.append((val_dice, ckpt_path))
        best_checkpoints.sort(
            key=lambda x: x[0], reverse=True
        )  # Descending sort by dice

        if len(best_checkpoints) > Config.SAVE_TOP_K:
            removed = best_checkpoints.pop()  # Remove smallest dice (last in list)
            if os.path.exists(removed[1]):
                os.remove(removed[1])

    # --- 6. Weight Averaging ---
    print("Averaging best checkpoints...")
    avg_state_dict = None

    # Load and sum
    for _, path in best_checkpoints:
        state = torch.load(path, map_location=device)
        if avg_state_dict is None:
            avg_state_dict = state
        else:
            for k in state:
                avg_state_dict[k] += state[k]

    # Divide
    if avg_state_dict is not None:
        for k in avg_state_dict:
            if avg_state_dict[k].is_floating_point():
                avg_state_dict[k] /= len(best_checkpoints)
            else:
                # For non-floating point buffers, keep original
                pass

        model.load_state_dict(avg_state_dict)

    # Save Best Averaged Model
    final_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Best averaged model saved to {final_model_path}")

    # --- 7. Final Validation ---
    print("Running final validation on averaged model...")
    final_loss, final_dice = valid_one_epoch(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_dice}")

    # --- 8. Failure Analysis ---
    print("Performing Failure Analysis...")
    model.eval()

    errors = []
    timestamps = []
    row_mins = []
    col_mins = []

    # Get metadata dataframe from dataset for correlation
    val_df = val_dataset.df
    batch_start_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Simple inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch_size = images.size(0)

            for i in range(batch_size):
                # Calculate Dice for this single image
                p = preds[i].view(-1)
                t = masks[i].view(-1)

                intersection = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()
                d = (2.0 * intersection + Config.SMOOTH) / (union + Config.SMOOTH)

                error = 1.0 - d
                errors.append(error)

                # Get metadata (sequential access matches loader)
                global_idx = batch_start_idx + i
                if global_idx < len(val_df):
                    row = val_df.iloc[global_idx]
                    timestamps.append(row.get("timestamp", 0))
                    row_mins.append(row.get("row_min", 0))
                    col_mins.append(row.get("col_min", 0))

            batch_start_idx += batch_size

    # Compute Correlations
    if len(errors) > 0:
        analysis_df = pd.DataFrame(
            {
                "error": errors,
                "timestamp": timestamps,
                "row_min": row_mins,
                "col_min": col_mins,
            }
        )

        # Drop NaNs
        analysis_df = analysis_df.dropna()

        if not analysis_df.empty:
            corr = analysis_df.corr()["error"]
            print("Correlation between Error Magnitude (1-Dice) and Metadata:")
            print(f"Timestamp: {corr.get('timestamp', 0):.6f}")
            print(f"Row Min (Lat): {corr.get('row_min', 0):.6f}")
            print(f"Col Min (Lon): {corr.get('col_min', 0):.6f}")
        else:
            print("Not enough data for correlation analysis.")

    # --- 9. Submission ---
    SUBMISSION_THRESHOLD = 0.6272749392944963

    if final_dice > SUBMISSION_THRESHOLD:
        print(f"Metric {final_dice} > {SUBMISSION_THRESHOLD}. Generating submission...")
        make_submission(
            checkpoint_path=final_model_path,
            output_csv=Config.SUBMISSION_FILE,
            device=device,
            debug=False,  # Generate for full test set
        )
    else:
        print(f"Metric {final_dice} <= {SUBMISSION_THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
