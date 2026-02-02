import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device, MCRMSE
from library.dataset import get_dataloader
from library.model import RNAModel
from library.engine import train_and_evaluate, get_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing configuration and environment...")
    seed_everything(Config.SEED)
    device = get_device()

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    # Train loader: Shuffle=True for training dynamics
    train_loader = get_dataloader("train", shuffle=True)
    # Val loader: Shuffle=False to align with metadata for failure analysis
    val_loader = get_dataloader("val", shuffle=False)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel(Config).to(device)

    # 4. Training
    print("Starting training...")
    # This function saves the best model to Config.MODEL_SAVE_PATH
    train_and_evaluate(model, train_loader, val_loader)

    # 5. Final Evaluation
    print("Reloading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Compute Final Validation Metric
    # We manually iterate to get predictions and targets aligned
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for seq, loop, dist, targets in val_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            outputs = model(seq, loop, dist)

            # Slice to scored positions
            outputs_scored = outputs[:, : Config.SEQ_SCORED, :]
            targets_scored = targets[:, : Config.SEQ_SCORED, :]

            all_val_preds.append(outputs_scored.cpu())
            all_val_targets.append(targets_scored.cpu())

    y_pred = torch.cat(all_val_preds, dim=0)
    y_true = torch.cat(all_val_targets, dim=0)

    # Compute global MCRMSE
    metric_fn = MCRMSE()
    final_score = metric_fn(y_true, y_pred).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Compute error per sample
    # y_true/pred shape: (N_samples, 68, 3)
    squared_diff = (y_true - y_pred) ** 2
    # Mean over sequence length (dim 1) -> (N_samples, 3)
    mse_per_sample_col = torch.mean(squared_diff, dim=1)
    # RMSE per column -> (N_samples, 3)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)
    # Mean over columns -> (N_samples,)
    sample_errors = torch.mean(rmse_per_sample_col, dim=1).numpy()

    # Load metadata to correlate
    df_val = pd.read_parquet(Config.VAL_PATH)

    # Ensure alignment (dataset loader with shuffle=False should match parquet order)
    if len(df_val) != len(sample_errors):
        print("Warning: Validation dataframe length mismatch with predictions.")
    else:
        df_val["error_mcrmse"] = sample_errors

        # Derived features for analysis
        df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
        df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
        df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
        df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

        analysis_cols = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
        ]

        print("Correlation between Error and Features:")
        correlations = {}
        for col in analysis_cols:
            if col in df_val.columns:
                corr = df_val[col].corr(df_val["error_mcrmse"])
                correlations[col] = corr
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Not found in metadata")

    # 7. Submission
    THRESHOLD = 0.6199890971183777
    if final_score < THRESHOLD:
        print(
            f"\nValidation score ({final_score:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        test_loader = get_dataloader("test", shuffle=False)
        test_preds, test_ids = get_predictions(model, test_loader)

        # test_preds shape: (N_test, 107, 3)
        # We need to map to 5 columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Model outputs: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)

        submission_rows = []

        # Iterate over samples
        for i, sample_id in enumerate(test_ids):
            pred_tensor = test_preds[i]  # (107, 3)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"

                # Extract predictions
                reactivity = float(pred_tensor[seqpos, 0])
                deg_Mg_pH10 = float(pred_tensor[seqpos, 1])
                deg_Mg_50C = float(pred_tensor[seqpos, 2])

                # Unscored columns set to 0
                deg_pH10 = 0.0
                deg_50C = 0.0

                submission_rows.append(
                    {
                        "id_seqpos": row_id,
                        "reactivity": reactivity,
                        "deg_Mg_pH10": deg_Mg_pH10,
                        "deg_pH10": deg_pH10,
                        "deg_Mg_50C": deg_Mg_50C,
                        "deg_50C": deg_50C,
                    }
                )

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({final_score:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
