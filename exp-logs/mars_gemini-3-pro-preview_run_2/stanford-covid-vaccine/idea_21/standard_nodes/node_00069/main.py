import os
import pandas as pd
import numpy as np
import torch
from library.config import Config, DEVICE
from library.utils import set_seed
from library.train import run_training
from library.model import ScaleDecoupledDenseNet
from library.data import get_dataloaders


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Train
    # We limit epochs to 10 for a fast baseline execution as requested.
    # The dataset is small enough that we don't need to subsample the data itself.
    print("Starting training pipeline...")
    best_score = run_training(max_epochs=10)

    # Required output format
    print(f"Final Validation Metric: {best_score}")

    # 3. Failure Analysis
    print("\nStarting Failure Analysis...")

    # Load the best model state
    model = ScaleDecoupledDenseNet().to(DEVICE)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(
            "Warning: Best model not found. Using initialized model for analysis (results will be random)."
        )

    model.eval()

    # Load validation data
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, p_indices, targets in val_loader:
            inputs, p_indices = inputs.to(DEVICE), p_indices.to(DEVICE)
            preds = model(inputs, p_indices)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSE per sample
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # Scored length: 68
    scored_cols = [0, 1, 3]
    scored_len = Config.SCORED_LEN

    # Slice to valid region
    p_valid = all_preds[:, :scored_len, scored_cols]
    t_valid = all_targets[:, :scored_len, scored_cols]

    # MSE per sample (average over length and channels)
    mse_per_sample = np.mean((p_valid - t_valid) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata to correlate with error
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure IDs match (dataloader preserves order for val set)
    val_ids = val_loader.dataset.ids

    analysis_df = pd.DataFrame({"id": val_ids, "rmse": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id")

    # Feature Engineering for analysis
    merged_df["len_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
    merged_df["len_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
    merged_df["len_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
    merged_df["len_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))

    # Calculate correlations
    features_to_check = ["signal_to_noise", "len_A", "len_G", "len_C", "len_U"]
    correlations = merged_df[["rmse"] + features_to_check].corr()["rmse"]

    print("Correlation between Error (RMSE) and Input Features:")
    print(correlations.drop("rmse"))

    # 4. Submission
    THRESHOLD = 0.5417620723771521

    if best_score < THRESHOLD:
        print(
            f"\nValidation score {best_score} < {THRESHOLD}. Generating submission..."
        )

        # Load test data
        _, _, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=True
        )

        test_preds = []
        with torch.no_grad():
            for inputs, p_indices, _ in test_loader:
                inputs, p_indices = inputs.to(DEVICE), p_indices.to(DEVICE)
                preds = model(inputs, p_indices)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)  # (N_samples, 107, 5)
        test_ids = test_loader.dataset.ids

        # Format submission
        submission_rows = []
        cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            sample_p = test_preds[i]
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_dict = {"id_seqpos": row_id}

                for col_idx, col_name in enumerate(cols):
                    row_dict[col_name] = sample_p[seqpos, col_idx]

                submission_rows.append(row_dict)

        sub_df = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score {best_score} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
