import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import config
from library.utils import seed_everything, calculate_mcrmse
from library import data
from library import model
from library import train


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(config.SEED)
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Modify configuration for a fast baseline execution
    # 15 epochs is sufficient for convergence on this dataset size while keeping runtime low.
    config.EPOCHS = 15

    # 2. Data Loading
    print("Loading data...")
    # Load cached data if available to save processing time
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    net = model.HC_BD_BiGRU()
    net.to(config.DEVICE)

    # 4. Training
    print("Starting training...")
    trainer = train.Trainer(net, train_loader, val_loader, test_loader)
    trainer.fit()

    # 5. Validation and Metric Calculation
    print("Performing final validation...")

    # Load the best model weights saved during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        net.load_state_dict(
            torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        )
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    net.eval()

    all_preds = []
    all_targets = []

    # Run inference on the validation set
    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(config.DEVICE)
            pair_indices = batch["pair_indices"].to(config.DEVICE)
            pair_mask = batch["pair_mask"].to(config.DEVICE)
            targets = batch["targets"].cpu().numpy()

            outputs = net(sequence, pair_indices, pair_mask)
            preds = outputs.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE using the provided utility
    # This function handles slicing to seq_scored and filtering for scored columns
    val_score = calculate_mcrmse(all_preds, all_targets)

    # Print the required metric string
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\nRunning failure analysis...")

    # Load validation metadata to get input features
    try:
        val_df = pd.read_parquet(config.VAL_PATH)
    except Exception as e:
        print(f"Could not load validation metadata for analysis: {e}")
        val_df = None

    if val_df is not None:
        # Align lengths if necessary (though they should match)
        min_len = min(len(val_df), len(all_preds))
        val_df = val_df.iloc[:min_len].reset_index(drop=True)
        preds_analysis = all_preds[:min_len]
        targets_analysis = all_targets[:min_len]

        # Identify indices of the scored columns
        scored_indices = [
            i for i, col in enumerate(config.TARGET_COLS) if col in config.SCORED_COLS
        ]

        # Slice predictions and targets to the scored region and columns
        # Predictions: (N, 107, 5) -> slice length -> slice columns
        p_sliced = preds_analysis[:, : config.SEQ_SCORED, :][:, :, scored_indices]

        # Targets: (N, 68, 5) -> slice columns
        # Note: Targets from loader are already length 68, but we double check dimensions
        t_sliced = targets_analysis
        if t_sliced.shape[1] > config.SEQ_SCORED:
            t_sliced = t_sliced[:, : config.SEQ_SCORED, :]
        t_sliced = t_sliced[:, :, scored_indices]

        # Calculate RMSE per sample (scalar value representing error magnitude)
        # Mean squared error over positions and columns, then sqrt
        mse_per_sample = np.mean((p_sliced - t_sliced) ** 2, axis=(1, 2))
        rmse_per_sample = np.sqrt(mse_per_sample)

        val_df["rmse"] = rmse_per_sample

        # Construct analysis dataframe
        analysis_data = pd.DataFrame()
        analysis_data["rmse"] = val_df["rmse"]

        # Add metadata features
        if "signal_to_noise" in val_df.columns:
            analysis_data["signal_to_noise"] = val_df["signal_to_noise"]
        if "SN_filter" in val_df.columns:
            analysis_data["SN_filter"] = val_df["SN_filter"]

        # Add sequence content features
        analysis_data["len"] = val_df["sequence"].apply(len)
        analysis_data["pct_A"] = val_df["sequence"].apply(
            lambda s: s.count("A") / len(s)
        )
        analysis_data["pct_G"] = val_df["sequence"].apply(
            lambda s: s.count("G") / len(s)
        )
        analysis_data["pct_C"] = val_df["sequence"].apply(
            lambda s: s.count("C") / len(s)
        )
        analysis_data["pct_U"] = val_df["sequence"].apply(
            lambda s: s.count("U") / len(s)
        )
        analysis_data["pct_paired"] = val_df["structure"].apply(
            lambda s: 1.0 - s.count(".") / len(s)
        )

        # Compute Correlation
        corr = analysis_data.corr()["rmse"].sort_values(ascending=False)
        print("Correlation of Error (RMSE) with features:")
        print(corr)

    # 7. Submission Generation
    THRESHOLD = 0.5884495377540588

    if val_score < THRESHOLD:
        print(
            f"\nValidation score {val_score} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Setup submission path
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Update config path so trainer saves to the correct location
        config.SUBMISSION_PATH = submission_path

        # Generate submission
        trainer.generate_submission()
    else:
        print(
            f"\nValidation score {val_score} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
