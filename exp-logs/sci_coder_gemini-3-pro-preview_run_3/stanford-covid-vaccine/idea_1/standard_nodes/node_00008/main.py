import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_data, RNADataset
from library.engine import train_model, generate_submission, predict_fn


def main():
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Train
    X_train, y_train = get_data("train", load_cached_data=True)
    train_dataset = RNADataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Validation
    X_val, y_val = get_data("val", load_cached_data=True)
    val_dataset = RNADataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Test
    X_test, _ = get_data("test", load_cached_data=True)
    test_dataset = RNADataset(X_test, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Training
    print("\nStarting training...")
    model, history = train_model(train_loader, val_loader, device)

    # 4. Validation Assessment
    print("\nEvaluating on Validation set...")
    # Generate raw predictions (N, 107, 5)
    val_preds = predict_fn(model, val_loader, device)

    # Reshape to (N, 68, 5)
    val_preds_reshaped = val_preds.reshape(-1, Config.SEQ_SCORED, Config.NUM_TARGETS)
    y_val_reshaped = y_val.reshape(-1, Config.SEQ_SCORED, Config.NUM_TARGETS)

    # Filter for scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]
    val_preds_scored = val_preds_reshaped[:, :, scored_indices]
    y_val_scored = y_val_reshaped[:, :, scored_indices]

    # Compute MCRMSE on scored columns only
    # mcrmse(y_true, y_pred)
    val_metric_tensor = mcrmse(y_val_scored, val_preds_scored)
    val_metric = val_metric_tensor.item()

    print(f"Final Validation Metric (Scored Columns Only): {val_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load metadata
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Calculate error magnitude per sample (RMSE across scored targets for that sample)
    # diff: (N, 68, 3)
    diff = val_preds_scored - y_val_scored
    # Mean Squared Error per sample
    sample_mse = np.mean(diff**2, axis=(1, 2))
    # Root Mean Squared Error per sample
    sample_rmse = np.sqrt(sample_mse)

    df_val["error_magnitude"] = sample_rmse

    # Prepare features for correlation
    analysis_features = ["signal_to_noise", "SN_filter"]

    # Add nucleotide content features
    for char in ["A", "G", "C", "U"]:
        col_name = f"pct_{char}"
        # Ensure we don't duplicate if already in df (though parquet load shouldn't have them computed)
        df_val[col_name] = df_val["sequence"].apply(lambda s: s.count(char) / len(s))
        analysis_features.append(col_name)

    # Compute correlations
    correlations = (
        df_val[analysis_features].corrwith(df_val["error_magnitude"]).sort_values()
    )

    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    print("\nGenerating Submission...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)
    generate_submission(model, test_loader, df_test, device)

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
