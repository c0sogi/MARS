import os
import torch
import pandas as pd
import numpy as np
import shutil

from library.config import Config
from library.train import train_model
from library.dataset import get_dataloader
from library.model import SPMHABiGRU
from library.utils import set_seed, compute_global_mcrmse


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution
    # 15 epochs is sufficient for convergence on this dataset size (~1700 samples)
    Config.EPOCHS = 15

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("\n==== Starting Training ====")
    # train_model handles the loop, validation, and saving the best checkpoint
    history = train_model()

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n==== Final Validation ====")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = SPMHABiGRU().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Load validation data
    val_loader = get_dataloader("val", shuffle=False, load_cached_data=True)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(inputs, pair_indices)

            # Move to CPU
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate batches
    preds_concat = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    targets_concat = np.concatenate(all_targets, axis=0)  # (N, 107, 5)

    # Define Scored Columns and Length
    # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    pred_len = Config.PRED_LEN  # 68

    # Slice for metric calculation
    # We only score the first 68 positions and the 3 specific columns
    preds_scored = preds_concat[:, :pred_len, scored_indices]
    targets_scored = targets_concat[:, :pred_len, scored_indices]

    # Compute Final Metric
    final_metric = compute_global_mcrmse(preds_scored, targets_scored)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n==== Failure Analysis ====")

    # Calculate RMSE per sample (averaged over scored positions and scored columns)
    # Shape: (N, 68, 3) -> Mean over (68, 3) per sample
    # MSE per sample
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to get features
    val_df = pd.read_parquet(Config.VAL_METADATA)
    val_df.set_index("id", inplace=True)

    analysis_data = []

    for i, sample_id in enumerate(all_ids):
        if sample_id not in val_df.index:
            continue

        row = val_df.loc[sample_id]

        # Extract features
        sn = row.get("signal_to_noise", 0)
        seq_len = len(row["sequence"])

        # Nucleotide content
        seq = row["sequence"]
        pct_A = seq.count("A") / seq_len
        pct_G = seq.count("G") / seq_len
        pct_C = seq.count("C") / seq_len
        pct_U = seq.count("U") / seq_len

        # Structure content
        struc = row["structure"]
        pct_paired = struc.count("(") / seq_len  # '(' count is half of total paired

        analysis_data.append(
            {
                "error": rmse_per_sample[i],
                "signal_to_noise": sn,
                "pct_A": pct_A,
                "pct_G": pct_G,
                "pct_C": pct_C,
                "pct_U": pct_U,
                "pct_paired": pct_paired,
            }
        )

    analysis_df = pd.DataFrame(analysis_data)

    if not analysis_df.empty:
        # Compute correlation
        corrs = analysis_df.corr()["error"].sort_values(ascending=False)
        print("Correlation of Error with Features:")
        print(corrs)

        print("\nTop 3 Features associated with High Error:")
        print(corrs.abs().sort_values(ascending=False).head(4).drop("error"))
    else:
        print("Could not perform failure analysis due to missing metadata alignment.")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_loader = get_dataloader("test", shuffle=False, load_cached_data=True)

        test_preds_list = []
        test_ids_list = []

        # Inference on Test Set
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                ids = batch["id"]

                # Forward pass
                outputs = model(inputs, pair_indices)  # (B, 107, 5)

                test_preds_list.append(outputs.cpu().numpy())
                test_ids_list.extend(ids)

        test_preds_concat = np.concatenate(test_preds_list, axis=0)

        # Format Submission
        submission_rows = []
        target_cols = Config.TARGET_COLS

        print(f"Processing {len(test_ids_list)} test samples for submission...")

        for i, sample_id in enumerate(test_ids_list):
            sample_preds = test_preds_concat[i]  # (107, 5)

            # We must predict for ALL sequence positions (0 to 106)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_preds = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_preds[col_idx])

                submission_rows.append(row_dict)

        sub_df = pd.DataFrame(submission_rows)

        # Ensure output directory exists
        os.makedirs("./submission", exist_ok=True)
        save_path = "./submission/submission.csv"

        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Submission shape: {sub_df.shape}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
