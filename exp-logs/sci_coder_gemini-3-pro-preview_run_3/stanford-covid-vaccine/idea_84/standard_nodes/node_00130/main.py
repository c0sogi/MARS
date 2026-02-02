import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import warnings

# Import from provided library
from library.config import Config
from library.data_utils import get_dataloaders, load_data
from library.model import MCSDBiGRU
from library.engine import train_fn, eval_fn, get_scored_indices
from library.loss import MCRMSELoss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_gc_content(sequence):
    """Calculates GC content of a sequence."""
    if not sequence:
        return 0.0
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Correlates model error with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Run inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(features, pair_indices)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate error per sample (MCRMSE on scored columns)
    scored_indices = get_scored_indices()

    # Slice to scored length
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, scored_indices]
    targets_sliced = all_targets[:, : Config.SEQ_SCORED, scored_indices]

    # MSE per sample: (N, L, C) -> Mean over (L, C) -> Sqrt -> (N,)
    diff_sq = (preds_sliced - targets_sliced) ** 2
    mse_per_sample = torch.mean(diff_sq, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load validation metadata to get features
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Map errors to dataframe
    # Ensure order matches
    id_to_error = dict(zip(all_ids, rmse_per_sample))
    val_df["error"] = val_df["id"].map(id_to_error)

    # Calculate GC content
    val_df["gc_content"] = val_df["sequence"].apply(calculate_gc_content)

    # Features to analyze
    features = ["signal_to_noise", "SN_filter", "gc_content", "seq_length"]

    print(f"{'Feature':<20} {'Correlation with Error':<25}")
    print("-" * 45)

    for feat in features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["error"])
            print(f"{feat:<20} {corr:<25.4f}")
        else:
            print(f"{feat:<20} {'Not Found':<25}")

    return val_df


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\nGenerating submission...")
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # (B, SeqLen, 5)
            outputs = model(features, pair_indices)

            ids_list.extend(ids)
            preds_list.append(outputs.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # Prepare data for CSV
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()
            submission_data.append([row_id] + row_values)

    columns = ["id_seqpos"] + target_cols
    submission_df = pd.DataFrame(submission_data, columns=columns)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    Config.EPOCHS = 10  # Limit epochs for speed

    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    # Using full dataset but limited epochs as per strategy analysis
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = MCSDBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS * len(train_loader))

    # 4. Training Loop
    best_metric = float("inf")

    print("\nStarting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)
        val_metric = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_metric:.4f}"
        )

        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"\nTraining complete. Best Validation Metric: {best_metric}")

    # 5. Final Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on full validation set
    final_val_metric = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    failure_analysis(model, val_loader, device)

    # 6. Submission
    # Threshold from requirements
    THRESHOLD = 0.5884495377540588

    if final_val_metric < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric ({final_val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
