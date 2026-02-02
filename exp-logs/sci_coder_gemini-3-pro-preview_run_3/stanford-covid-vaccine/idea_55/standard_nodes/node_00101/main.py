import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import warnings

# Ensure current directory is in path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SDBR_BiGRU
from library.engine import Engine

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def analyze_failures(engine, val_loader, val_parquet_path):
    """
    Performs failure analysis on the validation set by correlating model error
    with input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Get Predictions and Targets
    engine.model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(Config.DEVICE)
            pair_indices = batch["pair_indices"].to(Config.DEVICE)
            pair_masks = batch["pair_masks"].to(Config.DEVICE)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = engine.model(features, pair_indices, pair_masks)
            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    if not all_preds:
        print("No validation data found for analysis.")
        return

    preds = torch.cat(all_preds, dim=0).numpy()  # (N, 107, 5)
    targets = torch.cat(all_targets, dim=0).numpy()  # (N, 107, 5)

    # 2. Calculate Error per Sample (RMSE)
    # Focus on scored sequence length and scored targets
    preds_sliced = preds[:, : Config.SEQ_SCORED, Config.SCORED_INDICES]
    targets_sliced = targets[:, : Config.SEQ_SCORED, Config.SCORED_INDICES]

    # MSE per sample: Average over positions (68) and targets (3)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Load Metadata
    try:
        df_val = pd.read_parquet(val_parquet_path)
    except Exception as e:
        print(f"Could not load validation parquet for metadata: {e}")
        return

    # Map errors to IDs
    error_map = dict(zip(all_ids, rmse_per_sample))
    df_val["model_error"] = df_val["id"].map(error_map)

    # Drop any rows that didn't get a prediction (shouldn't happen)
    df_val = df_val.dropna(subset=["model_error"])

    # 4. Feature Extraction for Correlation
    # GC Content
    df_val["gc_content"] = df_val["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    # Paired Percentage
    df_val["paired_pct"] = df_val["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    # Analysis Columns
    # signal_to_noise and SN_filter are already in the parquet
    analysis_cols = ["signal_to_noise", "SN_filter", "gc_content", "paired_pct"]

    # Calculate Correlations
    print(f"Correlation with Model Error (RMSE):")
    correlations = df_val[analysis_cols].corrwith(df_val["model_error"]).sort_values()
    print(correlations)

    # Show worst samples
    print("\nTop 5 Worst Predictions (Highest RMSE):")
    worst_samples = df_val.nlargest(5, "model_error")[
        ["id", "model_error", "signal_to_noise"]
    ]
    print(worst_samples)


def generate_submission(engine, test_loader, submission_path):
    """
    Generates predictions for the test set and saves them in the submission format.
    """
    print("\n==== Generating Submission ====")

    # Get predictions
    preds, ids = engine.predict(test_loader)  # preds: (N, 107, 5)

    if len(preds) == 0:
        print("No predictions generated.")
        return

    # Prepare data for DataFrame
    data = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            # Format: id_{sample_id}_{seqpos}
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(cols):
                row_dict[col_name] = float(row_values[col_idx])

            data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(data)

    # Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # 15 epochs is sufficient for convergence on this dataset size (~2k samples)
    Config.NUM_EPOCHS = 15

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model Initialization
    print("Initializing Model...")
    model = SDBR_BiGRU()

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Engine Initialization
    engine = Engine(model, optimizer, scheduler, device=Config.DEVICE)

    # 5. Training
    print("Starting Training...")
    engine.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Final Evaluation
    print("Loading best model for evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        engine.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    val_score = engine.evaluate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(engine, val_loader, Config.VAL_PATH)

    # 8. Submission Generation
    THRESHOLD = 0.5884495377540588

    if val_score < THRESHOLD:
        generate_submission(engine, test_loader, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score {val_score} is not lower than threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
