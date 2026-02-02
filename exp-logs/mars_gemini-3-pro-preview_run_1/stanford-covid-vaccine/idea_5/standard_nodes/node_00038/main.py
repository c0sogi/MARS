import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.stats as stats

# Suppress tqdm progress bars as per requirements
# We must do this before importing modules that use tqdm
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = noop_tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    VAL_PATH,
    SEQ_SCORED,
    TARGET_COLS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    EPOCHS,
    SEED,
    PATIENCE,
)
from library.dataset import get_dataloaders
from library.model import HybridRNNTransformer
from library.loss import SignalWeightedMSELoss
from library.trainer import Trainer
from library.inference import generate_submission


def seed_everything(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    ids = []

    # Scored target indices: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["predicted_loop_type"].to(device)
            targets = batch["targets"].to(device)
            batch_ids = batch["id"]

            outputs = model(seq, struct, loop)

            # Extract scored positions and columns
            # Outputs: (B, 107, 5) -> (B, 68, 3)
            preds_scored = outputs[:, :SEQ_SCORED, scored_indices].cpu().numpy()
            targets_scored = targets[:, :SEQ_SCORED, scored_indices].cpu().numpy()

            all_preds.append(preds_scored)
            all_targets.append(targets_scored)
            ids.extend(batch_ids)

    # Concatenate
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE per sample
    # Error shape: (N, 68, 3)
    squared_diff = (preds_arr - targets_arr) ** 2
    # Mean over positions (axis 1) and targets (axis 2) -> (N,)
    # Note: MCRMSE is usually root-mean-squared.
    # Per sample error metric: mean of RMSEs per column for that sample
    # RMSE per column per sample: sqrt(mean(diff^2, axis=1)) -> (N, 3)
    rmse_per_col_per_sample = np.sqrt(np.mean(squared_diff, axis=1))
    # Mean across columns -> (N,)
    sample_errors = np.mean(rmse_per_col_per_sample, axis=1)

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": ids, "error": sample_errors})

    # Merge with metadata
    # Ensure val_df is indexed by id or we merge on id
    analysis_df = pd.merge(error_df, val_df, on="id", how="left")

    # Feature Engineering for Correlation
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))
    analysis_df["gc_content"] = (
        analysis_df["len_G"] + analysis_df["len_C"]
    ) / analysis_df["seq_length"]

    # Select numeric columns for correlation
    # We look for signal_to_noise, SN_filter, and sequence properties
    features = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
        "gc_content",
    ]

    print("Correlation between Model Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs if any
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=BATCH_SIZE
    )

    # 3. Model Initialization
    model = HybridRNNTransformer().to(device)

    # 4. Training Setup
    criterion = SignalWeightedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Adjust epochs for fast baseline if needed, but 15-25 is fast enough on this data size
    # We use a slightly reduced epoch count to ensure completion within 2 hours easily
    fast_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fast_epochs, eta_min=1e-6
    )

    trainer = Trainer(model, device, criterion, optimizer, scheduler)

    # 5. Train
    print("Starting Training...")
    trainer.fit(train_loader, val_loader, epochs=fast_epochs, patience=PATIENCE)

    # 6. Final Evaluation
    # Load best model
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Calculate Metric
    print("Calculating Final Validation Metric...")
    val_mcrmse = trainer.validate(val_loader)
    print(f"Final Validation Metric: {val_mcrmse}")

    # 7. Failure Analysis
    # Load validation dataframe for metadata
    val_df = pd.read_parquet(VAL_PATH)
    run_failure_analysis(model, val_loader, val_df, device)

    # 8. Submission
    # Threshold from instructions: 0.7462618350982666
    THRESHOLD = 0.7462618350982666

    if val_mcrmse < THRESHOLD:
        print(
            f"Validation metric {val_mcrmse} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model_path=best_model_path,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
        )
    else:
        print(
            f"Validation metric {val_mcrmse} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
