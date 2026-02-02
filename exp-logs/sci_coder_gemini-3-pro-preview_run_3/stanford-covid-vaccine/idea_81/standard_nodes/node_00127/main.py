import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse, get_scored_indices
from library.data import get_dataloaders
from library.model import DeepResBiGRU
from library.engine import Trainer


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # We use load_cached_data=True to utilize preprocessed .npz files if available
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Model...")
    model = DeepResBiGRU()
    model.to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # ==========================================
    # 5. Validation & Metric Calculation
    # ==========================================
    print("\n==== Validation Evaluation ====")
    # Load the best model saved during training
    best_model_path = Config.MODEL_PATH
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model state.")

    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            adjacency = batch["adjacency"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(inputs, adjacency, pair_mask)

            # Move to CPU to save GPU memory
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 68, 5)

    # Calculate MCRMSE
    # The utility function handles slicing predictions to match target length (68)
    # and filtering for the scored columns.
    val_metric = mcrmse(all_targets, all_preds, only_scored=True).item()

    # Print the required metric string
    print(f"Final Validation Metric: {val_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n==== Failure Analysis ====")

    # Calculate RMSE per sample for analysis
    # Slice predictions to match targets for error calculation
    seq_len_scored = all_targets.shape[1]
    preds_sliced = all_preds[:, :seq_len_scored, :]

    # Squared Error: (N, 68, 5)
    squared_error = (all_targets - preds_sliced) ** 2

    # Filter for scored columns for relevant analysis
    scored_indices = get_scored_indices()
    squared_error_scored = squared_error[:, :, scored_indices]

    # Mean over sequence and targets -> MSE per sample -> RMSE
    mse_per_sample = torch.mean(squared_error_scored, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Create Analysis DataFrame
    error_df = pd.DataFrame({"id": all_ids, "rmse": rmse_per_sample})

    # Load metadata to correlate errors with features
    if os.path.exists(Config.VAL_FILE):
        val_meta = pd.read_parquet(Config.VAL_FILE)

        # Merge error metrics with metadata
        analysis_df = pd.merge(error_df, val_meta, on="id", how="left")

        # Calculate GC Content as a feature
        if "sequence" in analysis_df.columns:
            analysis_df["gc_content"] = analysis_df["sequence"].apply(
                lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0
            )

        # Compute Correlations
        features_to_check = ["signal_to_noise", "SN_filter", "gc_content"]
        print("Correlation of RMSE with features:")
        for feat in features_to_check:
            if feat in analysis_df.columns:
                corr = analysis_df["rmse"].corr(analysis_df[feat])
                print(f"  {feat}: {corr:.4f}")
    else:
        print("Validation metadata not found. Skipping feature correlation analysis.")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.5884495377540588

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )

        test_preds_list = []
        test_ids_list = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                adjacency = batch["adjacency"].to(device)
                pair_mask = batch["pair_mask"].to(device)
                ids = batch["id"]

                outputs = model(inputs, adjacency, pair_mask)
                # Outputs: (B, 107, 5) - We need predictions for all 107 positions

                test_preds_list.append(outputs.cpu().numpy())
                test_ids_list.extend(ids)

        # Concatenate all test predictions
        test_preds = np.concatenate(test_preds_list, axis=0)  # (N_test, 107, 5)

        # Format for submission
        submission_rows = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids_list):
            sample_pred = test_preds[i]  # (107, 5)

            for seqpos in range(sample_pred.shape[0]):
                # Format: id_seqpos
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()
                submission_rows.append([row_id] + row_values)

        # Create DataFrame
        submission_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + target_cols
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(f"\nValidation metric {val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
