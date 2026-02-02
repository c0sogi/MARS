import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import RNA_Model
from library.engine import fit, validate
from library.loss import MaskedMSELoss, mcrmse


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set by correlating
    sample-wise errors with metadata features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    model.eval()

    # 1. Collect Predictions and Targets
    all_preds = []
    all_targets = []
    all_ids = []

    # We need to load metadata to correlate with features
    # The dataset returns 'id', which we can use to join with the parquet file
    df_val = pd.read_parquet(Config.VAL_PATH)

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)

            # Move to CPU
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate
    preds_arr = np.concatenate(all_preds, axis=0)  # (N, 107, 3)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 68, 3)

    # Slice preds to scored region for error calculation
    preds_scored = preds_arr[:, : Config.SEQ_SCORED, :]

    # 2. Calculate Sample-wise Error (MCRMSE per sample)
    # Error = Mean of RMSEs across the 3 columns
    # squared diff: (N, 68, 3)
    sq_diff = (preds_scored - targets_arr) ** 2
    # MSE per sample per column: mean over seq_len (axis 1) -> (N, 3)
    mse_per_sample_col = np.mean(sq_diff, axis=1)
    # RMSE per sample per column -> (N, 3)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # MCRMSE per sample: mean over columns (axis 1) -> (N,)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Create a DataFrame for analysis
    df_errors = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # Merge with metadata
    # We need features like signal_to_noise, SN_filter, sequence content
    # Calculate sequence content from the dataframe
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # Merge
    df_analysis = pd.merge(df_errors, df_val, on="id", how="left")

    # 3. Compute Correlations
    features_to_check = ["signal_to_noise", "len_A", "len_G", "len_C", "len_U"]
    if "SN_filter" in df_analysis.columns:
        features_to_check.append("SN_filter")

    print(f"Correlation between Model Error (MCRMSE) and Features:")
    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Drop NaNs if any
            valid_data = df_analysis[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found")


def generate_submission(model, test_loader, device):
    """
    Generates the submission file for the test set.
    """
    print("\n" + "=" * 30)
    print("GENERATING SUBMISSION")
    print("=" * 30)

    model.eval()

    submission_data = []

    # Columns required in submission
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Model output mapping (indices in output tensor)
    # 0: reactivity
    # 1: deg_Mg_pH10
    # 2: deg_Mg_50C

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            # Forward pass: (B, 107, 3)
            preds = model(seq, loop, dist)
            preds = preds.cpu().numpy()

            batch_size, seq_len, _ = preds.shape

            for i in range(batch_size):
                sample_id = ids[i]
                sample_preds = preds[i]  # (107, 3)

                for seqpos in range(seq_len):
                    # Construct row id
                    row_id = f"{sample_id}_{seqpos}"

                    # Extract predictions
                    reactivity = float(sample_preds[seqpos, 0])
                    deg_Mg_pH10 = float(sample_preds[seqpos, 1])
                    deg_Mg_50C = float(sample_preds[seqpos, 2])

                    # Fill others with 0
                    deg_pH10 = 0.0
                    deg_50C = 0.0

                    submission_data.append(
                        {
                            "id_seqpos": row_id,
                            "reactivity": reactivity,
                            "deg_Mg_pH10": deg_Mg_pH10,
                            "deg_pH10": deg_pH10,
                            "deg_Mg_50C": deg_Mg_50C,
                            "deg_50C": deg_50C,
                        }
                    )

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data)

    # Ensure column order
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = df_sub[cols]

    # Save
    Config.create_dirs()
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Rows: {len(df_sub)}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNA_Model().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training
    print("Starting training...")
    fit(model, train_loader, val_loader, optimizer, scheduler, device, Config.EPOCHS)

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    criterion = MaskedMSELoss()
    _, final_metric = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.6176461577
    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
