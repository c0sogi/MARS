import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # Using 25 epochs for a fast baseline execution as requested.
    # The dataset is small enough that this will finish quickly.
    print("Starting training...")
    run_training(epochs=25, load_cached_data=True)

    # 3. Load Best Model for Evaluation
    print("Loading best model...")
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 4. Validation & Metric Calculation
    print("Running validation inference...")
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            pair_index = batch.get("pair_index")
            pair_mask = batch.get("pair_mask")
            if pair_index is not None:
                pair_index = pair_index.to(device)
            if pair_mask is not None:
                pair_mask = pair_mask.to(device)

            outputs = model(inputs, pair_index=pair_index, pair_mask=pair_mask)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ids.extend(ids)

    # Concatenate results
    val_preds = np.concatenate(val_preds, axis=0)  # (N, 107, 5)
    val_targets = np.concatenate(val_targets, axis=0)  # (N, 107, 5)

    # Calculate MCRMSE on Scored Columns and Positions
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # Scored length: 68
    scored_len = Config.PRED_LEN
    scored_cols = [0, 1, 3]

    preds_sliced = val_preds[:, :scored_len, scored_cols]
    targets_sliced = val_targets[:, :scored_len, scored_cols]

    # MCRMSE: Mean of RMSEs per column
    mse = np.mean((preds_sliced - targets_sliced) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    mcrmse = np.mean(rmse)

    print(f"Final Validation Metric: {mcrmse}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate per-sample error (mean of RMSEs across the 3 scored columns)
    sample_mse = np.mean((preds_sliced - targets_sliced) ** 2, axis=1)  # (N, 3)
    sample_rmse = np.sqrt(sample_mse)
    sample_error = np.mean(sample_rmse, axis=1)  # (N,)

    # Load metadata
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Align metadata with predictions using IDs
    val_df.set_index("id", inplace=True)
    val_df = val_df.reindex(val_ids)
    val_df["error"] = sample_error

    # Feature Engineering for Analysis
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    val_df["unpaired_ratio"] = val_df["structure"].apply(
        lambda s: s.count(".") / len(s)
    )

    # Calculate Correlations
    analysis_cols = [
        "error",
        "signal_to_noise",
        "SN_filter",
        "gc_content",
        "unpaired_ratio",
    ]
    # Filter only existing columns
    analysis_cols = [c for c in analysis_cols if c in val_df.columns]

    correlations = val_df[analysis_cols].corr()["error"].sort_values(ascending=False)
    print("Correlation with Error Magnitude:")
    print(correlations)

    # 6. Submission Generation
    threshold = 0.5884495377540588
    if mcrmse < threshold:
        print("\nMetric passed threshold. Generating submission...")

        test_preds = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                ids = batch["id"]

                pair_index = batch.get("pair_index")
                pair_mask = batch.get("pair_mask")
                if pair_index is not None:
                    pair_index = pair_index.to(device)
                if pair_mask is not None:
                    pair_mask = pair_mask.to(device)

                outputs = model(inputs, pair_index=pair_index, pair_mask=pair_mask)
                test_preds.append(outputs.cpu().numpy())
                test_ids_list.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)  # (N_test, 107, 5)

        # Format submission
        # Rows: id_seqpos
        # Cols: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_col_names = Config.TARGET_COLS

        for i, sample_id in enumerate(test_ids_list):
            pred_matrix = test_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = pred_matrix[seqpos].tolist()
                submission_rows.append([row_id] + row_values)

        sub_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + target_col_names)

        # Save to ./submission/submission.csv
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"\nMetric {mcrmse} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
