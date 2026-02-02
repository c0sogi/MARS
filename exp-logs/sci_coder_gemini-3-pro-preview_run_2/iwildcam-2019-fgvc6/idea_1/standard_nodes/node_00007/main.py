import os
import numpy as np
import torch
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_macro_f1
from library.dataset import create_dataloaders
from library.feature_extractor import get_features
from library.model import train_model, generate_submission


def run_failure_analysis(model, val_features, val_targets):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude (binary error) and input features.
    """
    print("\n--- Failure Analysis ---")
    device = torch.device(Config.DEVICE)
    model.eval()

    # Convert features to tensor for inference
    X_val = torch.from_numpy(val_features).float().to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(X_val)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Calculate binary error (1 if incorrect, 0 if correct)
    errors = (preds != val_targets).astype(int)
    error_rate = np.mean(errors)
    print(f"Validation Error Rate: {error_rate:.4f}")

    if error_rate == 0 or np.std(errors) == 0:
        print(
            "No variance in errors (perfect or total failure). Correlation analysis skipped."
        )
        return

    # Calculate Pearson correlation between Error vector and each Feature dimension
    # Vectorized implementation for speed

    # Center the data
    X_centered = val_features - val_features.mean(axis=0)
    e_centered = errors - errors.mean()

    # Compute Covariance: (E . X) / (N - 1)
    # e_centered shape: (N,), X_centered shape: (N, 4096)
    covariance = np.dot(e_centered, X_centered) / (len(errors) - 1)

    # Compute Standard Deviations
    X_std = val_features.std(axis=0)
    e_std = errors.std()

    # Avoid division by zero for constant features
    valid_mask = X_std > 1e-9

    correlations = np.zeros(val_features.shape[1])
    correlations[valid_mask] = covariance[valid_mask] / (X_std[valid_mask] * e_std)

    # Identify top correlations
    # Positive correlation: High feature value -> Higher probability of Error
    # Negative correlation: High feature value -> Lower probability of Error (Better performance)
    top_pos_idx = np.argsort(correlations)[-5:][::-1]
    top_neg_idx = np.argsort(correlations)[:5]

    print("\nTop 5 Input Features Positively Correlated with Error:")
    for idx in top_pos_idx:
        print(f"  Feature {idx}: {correlations[idx]:.4f}")

    print("\nTop 5 Input Features Negatively Correlated with Error:")
    for idx in top_neg_idx:
        print(f"  Feature {idx}: {correlations[idx]:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting execution...")

    # 2. Data Loading
    # Using full dataset as the strategy is efficient
    print("\n[1/5] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Feature Extraction (with Caching)
    print("\n[2/5] Extracting/Loading Features...")
    train_features, train_targets = get_features(
        train_loader, mode="train", load_cached_data=True
    )
    val_features, val_targets = get_features(
        val_loader, mode="val", load_cached_data=True
    )
    test_features, test_ids = get_features(
        test_loader, mode="test", load_cached_data=True
    )

    # 4. Model Training
    print("\n[3/5] Training Linear Probe...")
    model = train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
    )

    # 5. Validation Assessment & Failure Analysis
    print("\n[4/5] Validating and Analyzing Failures...")

    # Re-run inference on validation set to get final predictions for metric calculation
    device = torch.device(Config.DEVICE)
    model.eval()
    X_val = torch.from_numpy(val_features).float().to(device)

    with torch.no_grad():
        outputs = model(X_val)
        val_preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Calculate and Print Metric
    final_metric = calculate_macro_f1(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Run Failure Analysis
    run_failure_analysis(model, val_features, val_targets)

    # 6. Submission Generation
    print("\n[5/5] Generating Submission...")

    baseline_metric = 0.2412
    if final_metric > baseline_metric:
        print(
            f"Validation metric ({final_metric:.6f}) improved over baseline ({baseline_metric:.6f}). Generating submission."
        )
        generate_submission(
            model, test_features, test_ids, output_path=Config.SUBMISSION_PATH
        )
    else:
        print(
            f"Validation metric ({final_metric:.6f}) did not improve over baseline ({baseline_metric:.6f}). Skipping submission."
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
