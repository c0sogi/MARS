import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import DeepInputAwareBiGRU
from library.engine import fit, evaluate


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify systematic errors.
    Calculates correlations between sample error and metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # Store sample-wise errors
    sample_errors = []
    sample_ids = []

    criterion = torch.nn.L1Loss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            # Forward pass
            preds = model(features, pair_indices, pair_masks)

            # Slice to scored length (68)
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            # Calculate MAE per sample (average over seq_len and targets)
            # shape: (Batch, 68, 5) -> (Batch,)
            errors = criterion(preds_sliced, targets_sliced).mean(dim=(1, 2))

            sample_errors.extend(errors.cpu().numpy())
            sample_ids.extend(ids)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": sample_ids, "mae": sample_errors})

    # Load Metadata to get features
    if os.path.exists(Config.VAL_METADATA):
        meta_df = pd.read_parquet(Config.VAL_METADATA)

        # Merge errors with metadata
        analysis_df = pd.merge(error_df, meta_df, on="id", how="left")

        # Feature Engineering for Analysis
        # 1. GC Content
        analysis_df["gc_content"] = analysis_df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s)
        )

        # 2. Paired Percentage
        analysis_df["paired_pct"] = analysis_df["structure"].apply(
            lambda s: (s.count("(") + s.count(")")) / len(s)
        )

        # Calculate Correlations
        correlations = {}
        features_to_check = ["signal_to_noise", "SN_filter", "gc_content", "paired_pct"]

        print(f"{'Feature':<20} | {'Correlation with Error':<20}")
        print("-" * 45)

        for feat in features_to_check:
            if feat in analysis_df.columns:
                # Drop NaNs just in case
                valid_data = analysis_df[[feat, "mae"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[feat], valid_data["mae"])
                    correlations[feat] = corr
                    print(f"{feat:<20} | {corr:.4f}")
                else:
                    print(f"{feat:<20} | N/A (Insufficient Data)")
            else:
                print(f"{feat:<20} | Not Found")

    else:
        print("Validation metadata not found. Skipping detailed correlation analysis.")


def generate_submission(model, test_loader, device):
    """
    Generates the submission file for the test set.
    """
    print("\n==== Generating Submission ====")
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]

            # Forward pass
            preds = model(features, pair_indices, pair_masks)

            # Slice to scored length (68)
            preds_sliced = preds[:, : Config.PRED_LEN, :]

            ids_list.extend(ids)
            preds_list.append(preds_sliced.cpu().numpy())

    # Concatenate all predictions: (Total_Test_Samples, 68, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (68, 5)

        for seq_pos in range(Config.PRED_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def main():
    # 1. Configuration Override for Fast Baseline
    # Limit epochs to ensure execution within time limits while maintaining performance
    Config.EPOCHS = 20

    # 2. Setup
    Config.setup()
    set_seed(Config.SEED)

    print(f"Running on device: {Config.DEVICE}")

    # 3. Data Loading
    # Using cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Model Initialization
    model = DeepInputAwareBiGRU()
    model.to(Config.DEVICE)

    # 5. Training
    model = fit(model, train_loader, val_loader, device=Config.DEVICE)

    # 6. Validation & Metrics
    val_score = evaluate(model, val_loader, device=Config.DEVICE)
    # REQUIRED: Print exact metric format
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device=Config.DEVICE)

    # 8. Submission
    THRESHOLD = 0.5978901386
    if val_score < THRESHOLD:
        print(
            f"Validation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device=Config.DEVICE)
    else:
        print(
            f"Validation score ({val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
