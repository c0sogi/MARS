import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import the library modules correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.engine import run_training, generate_submission, predict_fn
from library.dataset import prepare_data, get_dataloader


def calculate_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculates the Log Loss (Cross Entropy) for soft targets.

    Args:
        y_true: (N, C) array of ground truth probabilities.
        y_pred: (N, C) array of predicted probabilities.
        eps: Epsilon to prevent log(0).

    Returns:
        float: The average log loss.
    """
    # Clip predictions to avoid undefined log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Normalize rows to sum to 1 to ensure valid probability distribution
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    # Compute Cross Entropy: - sum(target * log(pred))
    # Sum over classes, then average over samples
    loss_per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss_per_sample)


def main():
    # 1. Configuration & Setup
    config = Config()

    # Override for Fast Baseline:
    # Limit to 1 epoch to ensure execution completes quickly within the time limit.
    # We retain symmetric augmentation (default in Config) to maximize performance within that epoch.
    config.epochs = 1

    # Set random seeds for full reproducibility
    seed_everything(config.seed)

    print("Starting Fast Baseline Run...")
    print(
        f"Configuration: Epochs={config.epochs}, Batch Size={config.train_batch_size}, Device={config.device}"
    )

    # 2. Training Pipeline
    # run_training handles data loading, training loop, validation, and checkpointing.
    # It returns the model with the best validation score loaded.
    model = run_training(config)

    # 3. Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    device = get_device()

    # Load validation data loader (for predictions) and dataframe (for targets/features)
    # Using load_cached_data=True to speed up loading if cache exists
    val_loader = get_dataloader(config, partition="val", load_cached_data=True)
    val_df = prepare_data(config, partition="val", load_cached_data=True)

    # Generate predictions on the validation set
    # predict_fn handles mixed precision and device movement
    val_preds = predict_fn(val_loader, model, device, config)

    # Extract ground truth targets
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    y_true = val_df[target_cols].values

    # Compute the Final Metric (Log Loss)
    metric = calculate_log_loss(y_true, val_preds)

    # REQUIRED OUTPUT: Print the metric with full precision
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample error magnitude (Cross Entropy)
    # We re-calculate this vector to correlate with features
    y_pred_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    error_magnitude = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    # Extract input features (text lengths)
    # Fill NaNs with empty string to ensure len() works
    val_df["prompt"] = val_df["prompt"].fillna("")
    val_df["response_a"] = val_df["response_a"].fillna("")
    val_df["response_b"] = val_df["response_b"].fillna("")

    len_prompt = val_df["prompt"].apply(len).values
    len_resp_a = val_df["response_a"].apply(len).values
    len_resp_b = val_df["response_b"].apply(len).values
    diff_len = np.abs(len_resp_a - len_resp_b)

    # Calculate correlations
    correlations = {
        "Prompt Length": np.corrcoef(len_prompt, error_magnitude)[0, 1],
        "Response A Length": np.corrcoef(len_resp_a, error_magnitude)[0, 1],
        "Response B Length": np.corrcoef(len_resp_b, error_magnitude)[0, 1],
        "Response Length Difference": np.corrcoef(diff_len, error_magnitude)[0, 1],
    }

    print("Correlation between Error Magnitude and Input Features:")
    for feature, corr in correlations.items():
        print(f"{feature}: {corr}")

    # 5. Submission Generation
    threshold = 1.0005665522536111

    if metric < threshold:
        print(
            f"\nMetric ({metric}) is better than threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, config)
    else:
        print(
            f"\nMetric ({metric}) did not improve upon threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
