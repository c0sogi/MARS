import os
import shutil
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.engine import Engine
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders


def main():
    # ---------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # ---------------------------------------------------------
    # Limit epochs to ensure quick execution within the time limit
    Config.EPOCHS = 10

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # Initialize the engine (builds model, optimizer, etc.)
    engine = Engine()

    # Run the training loop
    engine.run_training()

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    engine.model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    engine.model.eval()

    # Get Validation Data
    # reload dataloaders to ensure access to the validation set
    _, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            seq_input = batch["seq_input"].to(Config.DEVICE)
            loop_input = batch["loop_input"].to(Config.DEVICE)
            dist_input = batch["dist_input"].to(Config.DEVICE)

            # Forward pass
            outputs = engine.model(seq_input, loop_input, dist_input)

            # Extract degradation predictions
            # Shape: (B, 107, 5)
            pred_deg = outputs["pred_degradation"]

            # Slice to scored length (68) and move to CPU for metric calc
            pred_deg_scored = pred_deg[:, : Config.SCORED_LEN, :].cpu().numpy()
            targets = batch["targets"].numpy()  # (B, 68, 5)
            ids = batch["id"]

            all_preds.append(pred_deg_scored)
            all_targets.append(targets)
            all_ids.extend(ids)

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute Global Metric
    final_metric = mcrmse_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # 1. Calculate error per sample (RMSE of the sample across all targets and positions)
    # (N, 68, 5) -> (N,)
    sample_mse = np.mean((all_targets - all_preds) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # 2. Load Metadata to get features
    df_val = pd.read_parquet(Config.VAL_FILE)

    # Ensure alignment: filter and sort df_val to match the order of all_ids
    df_val = df_val.set_index("id").loc[all_ids].reset_index()

    # 3. Feature Engineering for Analysis
    df_val["error_magnitude"] = sample_rmse

    # Sequence based features
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))
    df_val["gc_content"] = (df_val["len_G"] + df_val["len_C"]) / df_val[
        "sequence"
    ].str.len()

    # 4. Correlation Analysis
    analysis_features = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
        "gc_content",
    ]

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    for feat in analysis_features:
        if feat in df_val.columns:
            # Drop NaNs for correlation calculation
            subset = df_val[[feat, "error_magnitude"]].dropna()
            if len(subset) > 1:
                corr, _ = pearsonr(subset[feat], subset["error_magnitude"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found in metadata")

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.6226052641868591

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Generate submission using Engine's method
        engine.generate_submission()

        # Move/Copy to the required location: ./submission/submission.csv
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        if os.path.exists(Config.SUBMISSION_PATH):
            shutil.copy(Config.SUBMISSION_PATH, target_path)
            print(f"Final submission saved to: {target_path}")
        else:
            print("Error: Source submission file not found.")
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
