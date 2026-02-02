import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.train import train_model
from library.model import DARDN
from library.data import get_loader


def run_failure_analysis(model, device):
    """
    Performs failure analysis on the validation set by correlating
    prediction errors with input features.
    """
    print("\n==== Running Failure Analysis ====")

    # Load validation metadata
    if not os.path.exists(Config.VAL_METADATA):
        print("Validation metadata not found. Skipping failure analysis.")
        return

    val_df = pd.read_csv(Config.VAL_METADATA)

    # Get loader
    val_loader = get_loader("val", batch_size=64, shuffle=False)

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            # Forward pass (use refined output y_2)
            _, y_2 = model(inputs, partner_indices)

            all_preds.append(y_2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 107, 5)

    # Calculate MCRMSE per sample
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    seq_scored = Config.SEQ_SCORED

    # Slice to scored region (0-67) and scored columns
    preds_scored = all_preds[:, :seq_scored, scored_indices]
    targets_scored = all_targets[:, :seq_scored, scored_indices]

    # MSE per sample: mean over (seq_len, channels)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    # Ensure IDs match
    merged_df = analysis_df.merge(val_df, on="id", how="left")

    # Features to correlate
    features = ["signal_to_noise", "SN_filter", "mean_reactivity"]

    # Add sequence composition features
    if "sequence" in merged_df.columns:
        merged_df["len_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
        merged_df["len_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
        merged_df["len_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
        merged_df["len_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))
        features.extend(["len_A", "len_G", "len_C", "len_U"])

    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in features:
        if feat in merged_df.columns:
            # Drop NaNs
            valid_data = merged_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat:<20} | {corr:<12.4f} | {p_val:<12.4e}")


def generate_submission(model, device):
    """
    Generates the submission file for the test set.
    """
    print("\n==== Generating Submission ====")

    # Load Test Loader
    test_loader = get_loader("test", batch_size=64, shuffle=False)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["ids"]

            # Forward pass
            _, y_2 = model(inputs, partner_indices)

            ids_list.extend(ids)
            preds_list.append(y_2.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)  # (N_test, 107, 5)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    target_cols = Config.TARGET_COLS

    seq_len = Config.SEQ_LENGTH  # 107

    print("Formatting submission rows...")

    N = len(ids_list)
    # Reshape predictions to (N*107, 5)
    flat_preds = all_preds.reshape(-1, 5)

    # Create IDs: Repeat each ID 107 times
    repeated_ids = np.repeat(ids_list, seq_len)
    # Create seqpos: 0..106 repeated N times
    tiled_seqpos = np.tile(np.arange(seq_len), N)

    # Combine to form id_seqpos
    id_seqpos = [f"{i}_{p}" for i, p in zip(repeated_ids, tiled_seqpos)]

    sub_df = pd.DataFrame(flat_preds, columns=target_cols)
    sub_df.insert(0, "id_seqpos", id_seqpos)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {sub_df.shape}")


def main():
    # 1. Train
    # Using 20 epochs with batch_size=16 to ensure convergence (Cite solution_lesson_node_00129)
    best_score = train_model(epochs=20, batch_size=16)

    # 2. Print Metric
    print(f"Final Validation Metric: {best_score}")

    # 3. Load Best Model
    device = torch.device(Config.DEVICE)
    model = DARDN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Best model not found. Using current model state.")

    model.eval()

    # 4. Failure Analysis
    run_failure_analysis(model, device)

    # 5. Submission
    THRESHOLD = 0.47142532743789534
    if best_score < THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Validation score {best_score} is not lower than threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
