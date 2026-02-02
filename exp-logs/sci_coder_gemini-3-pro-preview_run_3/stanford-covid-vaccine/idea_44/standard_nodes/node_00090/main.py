import pandas as pd
import numpy as np
import torch
import os
import sys
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.engine import run_training, eval_fn


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # 1. Collect predictions and targets per sample
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(features, pair_indices, pair_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # 2. Compute RMSE per sample
    # Slice to scored sequence length and scored columns
    preds_sliced = preds[:, : Config.SEQ_SCORED, Config.SCORING_INDICES]
    targets_sliced = targets[:, : Config.SEQ_SCORED, Config.SCORING_INDICES]

    # MSE per sample: Mean over sequence (dim 1) and channels (dim 2)
    sample_mse = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(1, 2))
    sample_rmse = torch.sqrt(sample_mse).numpy()

    # 3. Load Metadata to get features
    val_df = pd.read_parquet(Config.VAL_PATH)

    # Ensure alignment by ID
    analysis_df = pd.DataFrame({"id": all_ids, "rmse": sample_rmse})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="inner")

    # 4. Calculate Correlations
    features_to_analyze = ["signal_to_noise"]
    if "SN_filter" in merged_df.columns:
        features_to_analyze.append("SN_filter")

    # Add sequence composition features
    merged_df["pct_unpaired"] = merged_df["structure"].apply(
        lambda s: s.count(".") / len(s)
    )
    features_to_analyze.append("pct_unpaired")

    print("Correlation between Error (RMSE) and Input Features:")
    for feat in features_to_analyze:
        if feat in merged_df.columns:
            # Handle potential NaNs or infinite values
            valid_data = merged_df[[feat, "rmse"]].dropna()
            valid_data = valid_data[np.isfinite(valid_data[feat])]

            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")


def generate_submission(model, test_loader, device):
    """
    Generates the submission file for the test set.
    """
    print("\n==== Generating Submission ====")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]

            outputs = model(features, pair_indices, pair_masks)
            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Shape: (Num_Samples, 107, 5)
    preds_array = np.concatenate(all_preds, axis=0)

    # Prepare data for DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_data = []

    for i, sample_id in enumerate(all_ids):
        sample_preds = preds_array[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()
            submission_data.append([row_id] + row_values)

    submission_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + target_cols)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    print(f"Running fast baseline with {Config.EPOCHS} epochs on {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training
    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        Config.EPOCHS,
        Config.PATIENCE,
        Config.MODEL_SAVE_PATH,
    )

    # 6. Validation
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current model state.")

    val_score = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.5884495377540588
    if val_score < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {val_score} is not lower than {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
