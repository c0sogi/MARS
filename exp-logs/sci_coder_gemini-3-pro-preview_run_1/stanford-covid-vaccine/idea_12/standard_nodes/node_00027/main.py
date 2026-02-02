import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

# Import provided library components
from library.config import Config
from library.utils import set_seed, format_submission
from library.data import get_dataloaders
from library.model import InteractionAwareModel
from library.train import train_one_epoch, validate, MaskedMSELoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Modify Config for a fast baseline run
    Config.EPOCHS = 15
    Config.PATIENCE = 5

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing Model...")
    model = InteractionAwareModel().to(device)

    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Model Saving & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  >>> New Best Model Saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # =========================================================================
    # 5. Final Evaluation
    # =========================================================================
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to correlate errors with features
    if os.path.exists(Config.VAL_DATA_PATH):
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)

        # Generate predictions on validation set
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(inputs)

                # Slice to scored length (Batch, 68, 3)
                outputs_sliced = outputs[:, : Config.PRED_LEN, :]

                all_preds.append(outputs_sliced.cpu().numpy())
                all_targets.append(targets.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Calculate Mean Absolute Error (MAE) per sample
        # Shape: (N_samples, 68, 3) -> Average over positions and targets -> (N_samples,)
        abs_diff = np.abs(y_true - y_pred)
        sample_mae = np.mean(abs_diff, axis=(1, 2))

        # Add error to dataframe (assuming order is preserved, which is true for non-shuffled val loader)
        # Verify length
        if len(val_df) == len(sample_mae):
            val_df["model_error_mae"] = sample_mae

            # Calculate correlations
            features_to_analyze = ["signal_to_noise", "SN_filter", "seq_length"]

            print("Correlation between Model Error (MAE) and Input Features:")
            for feat in features_to_analyze:
                if feat in val_df.columns:
                    corr = val_df[feat].corr(val_df["model_error_mae"])
                    print(f"  {feat}: {corr:.4f}")
        else:
            print(
                "Warning: Validation dataframe length does not match prediction length. Skipping correlation analysis."
            )
    else:
        print("Validation metadata not found. Skipping failure analysis.")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    THRESHOLD = 0.6226052641868591

    if final_val_score < THRESHOLD:
        print(
            f"\nValidation score ({final_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_ids = []
        test_preds = []

        model.eval()
        with torch.no_grad():
            for inputs, ids in test_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(inputs)

                # Slice to scored length
                outputs_sliced = outputs[:, : Config.PRED_LEN, :]

                test_preds.append(outputs_sliced.cpu().numpy())
                test_ids.extend(ids)

        test_preds_arr = np.concatenate(test_preds, axis=0)

        # Define submission path
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        format_submission(test_ids, test_preds_arr, save_path=submission_path)
    else:
        print(
            f"\nValidation score ({final_val_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
