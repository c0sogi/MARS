import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import RNADataset
from library.model import RNA_Model
from library.loss import MCRMSELoss
from library.train_eval import train_epoch, validate


def run():
    # 1. Setup
    print("Initializing Regularized Non-Linear Dense-Context Network Pipeline...")
    Config.set_seed(Config.SEED)

    # Adjust Config for Fast Baseline execution
    # Restoring to 50 epochs to allow full convergence (Cite Lesson 00062 context)
    Config.EPOCHS = 50
    Config.PATIENCE = 5

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    # Force cache reload/creation if needed, but prefer loading existing cache
    train_dataset = RNADataset(mode="train", load_cached_data=True)
    val_dataset = RNADataset(mode="val", load_cached_data=True)
    test_dataset = RNADataset(mode="test", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 3. Model Initialization
    model = RNA_Model().to(device)
    criterion = MCRMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
    )

    # 4. Training Loop
    best_val_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("\nStarting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> Model saved (Improved from {best_val_score:.6f} to {val_score:.6f})")

        # Early stopping check handled by scheduler patience visually,
        # but we can break explicitly if learning rate drops too low
        if optimizer.param_groups[0]["lr"] < 1e-6:
            print("Learning rate too small. Stopping early.")
            break

    print(f"\nFinal Validation Metric: {best_val_score}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Calculate per-sample error on validation set
    val_errors = []

    # Indices for scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [Config.ALL_TARGETS.index(t) for t in Config.SCORED_TARGETS]

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, partner_indices)

            # Slice to scored length and columns
            preds_sliced = outputs[:, : Config.PRED_LEN, :]
            preds_scored = preds_sliced[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # MSE per sample (average over length and columns)
            # Shape: (Batch, Length, Cols) -> (Batch,)
            mse_per_sample = torch.mean(
                (preds_scored - targets_scored) ** 2, dim=(1, 2)
            )
            val_errors.extend(mse_per_sample.cpu().numpy())

    val_errors = np.array(val_errors)
    rmse_per_sample = np.sqrt(val_errors)

    # Load validation metadata to correlate
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Ensure alignment (dataset loader preserves order)
    if len(val_df) == len(rmse_per_sample):
        val_df["model_rmse"] = rmse_per_sample

        # Correlation with Signal to Noise
        corr_sn = val_df["model_rmse"].corr(val_df["signal_to_noise"])
        print(f"Correlation between Error (RMSE) and Signal-to-Noise: {corr_sn:.4f}")

        # Correlation with Sequence Length (Constant 107, so this will be NaN/0, skipping)

        # Correlation with SN_filter (Binary)
        corr_filter = val_df["model_rmse"].corr(val_df["SN_filter"])
        print(f"Correlation between Error (RMSE) and SN_filter: {corr_filter:.4f}")

        # Correlation with Mean Reactivity (using the metadata column if available)
        if "mean_reactivity" in val_df.columns:
            corr_react = val_df["model_rmse"].corr(val_df["mean_reactivity"])
            print(
                f"Correlation between Error (RMSE) and Mean Reactivity: {corr_react:.4f}"
            )
    else:
        print(
            "Warning: Validation metadata length mismatch. Skipping correlation analysis."
        )

    # 6. Submission
    THRESHOLD = 0.5417620723771521
    if best_val_score < THRESHOLD:
        print(
            f"\nValidation score ({best_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)

        preds_list = []
        ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                ids = batch["ids"]

                # Inference
                outputs = model(inputs, partner_indices)
                # Outputs shape: (Batch, 107, 5)

                preds_np = outputs.cpu().numpy()

                for i, sample_id in enumerate(ids):
                    # For each sample, we have 107 positions
                    sample_preds = preds_np[i]  # (107, 5)

                    for seqpos in range(Config.SEQ_LEN):
                        # Construct row ID
                        row_id = f"{sample_id}_{seqpos}"

                        # Get predictions for this position
                        # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                        # Config.ALL_TARGETS matches this order
                        vals = sample_preds[seqpos]

                        row_data = {
                            "id_seqpos": row_id,
                            "reactivity": vals[0],
                            "deg_Mg_pH10": vals[1],
                            "deg_pH10": vals[2],
                            "deg_Mg_50C": vals[3],
                            "deg_50C": vals[4],
                        }
                        preds_list.append(row_data)

        # Create DataFrame
        submission_df = pd.DataFrame(preds_list)

        # Ensure column order
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        submission_df = submission_df[cols]

        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation score ({best_val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
