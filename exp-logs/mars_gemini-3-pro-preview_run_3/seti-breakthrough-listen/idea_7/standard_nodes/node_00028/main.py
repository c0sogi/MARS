import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import SETIDataset
from library.model import SiameseMultiScaleDiffNet
from library.engine import fit
from library.inference import generate_submission


def main():
    # --- 1. Configuration & Setup ---
    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Full Training Configuration
    # Cite solution_lesson_node_00024: The Criticality of Data Volume
    EPOCHS = 15

    print(f"Initializing Full Training Run (Device: {device})")
    print(f"Settings: Epochs={EPOCHS}, Using Full Dataset")

    # --- 2. Data Preparation ---
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Define Training Augmentations (Synchronized Flips)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        additional_targets={"image_b": "image"},
    )

    # Create Datasets
    train_dataset = SETIDataset(train_df, transform=train_transform)
    val_dataset = SETIDataset(val_df, transform=None)

    # Create DataLoaders
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

    # --- 3. Model Initialization ---
    model = SiameseMultiScaleDiffNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 4. Training ---
    print("Starting Training Loop...")
    model = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=EPOCHS,
        patience=5,
    )

    # --- 5. Validation & Failure Analysis ---
    print("\nStarting Failure Analysis on Validation Set...")
    model.eval()

    val_preds = []
    val_targets = []
    meta_features = []

    with torch.no_grad():
        for data, target in val_loader:
            stream_a = data["stream_a"].to(device)
            stream_b = data["stream_b"].to(device)
            target_batch = target.to(device)

            # Inference
            logits = model(stream_a, stream_b)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(target_batch.cpu().numpy().flatten())

            # Extract features for failure analysis
            # Move to CPU for numpy operations
            sa_np = stream_a.cpu().numpy()  # (B, 3, H, W)
            sb_np = stream_b.cpu().numpy()  # (B, 3, H, W)

            # Compute stats per sample in the batch
            # We aggregate over channels (3), Height, and Width
            for i in range(len(sa_np)):
                # Flatten the 3 frames for stats
                flat_a = sa_np[i].flatten()
                flat_b = sb_np[i].flatten()

                mean_on = np.mean(flat_a)
                mean_off = np.mean(flat_b)

                stats = {
                    "mean_on_target": mean_on,
                    "std_on_target": np.std(flat_a),
                    "max_on_target": np.max(flat_a),
                    "mean_off_target": mean_off,
                    "std_off_target": np.std(flat_b),
                    "max_off_target": np.max(flat_b),
                    "mean_diff": mean_on - mean_off,
                }
                meta_features.append(stats)

    # Compute Final Metric
    final_auc = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation
    df_analysis = pd.DataFrame(meta_features)
    df_analysis["target"] = val_targets
    df_analysis["pred"] = val_preds
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["pred"])

    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = (
        df_analysis.corr()["error"]
        .drop(["error", "target", "pred"])
        .sort_values(ascending=False)
    )
    print(correlations)

    # --- 6. Submission ---
    # Threshold based on Idea 8 performance
    TARGET_THRESHOLD = 0.7871

    if final_auc > TARGET_THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device, output_path="./submission/submission.csv")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
