import os
import numpy as np
import torch
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_macro_f1
from library.dataset import create_dataloaders
from library.model import train_model, generate_submission


def run_failure_analysis(model, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude (binary error) and input features.
    """
    print("\n--- Failure Analysis ---")
    device = torch.device(Config.DEVICE)
    model.eval()

    all_features = []
    all_preds = []
    all_targets = []

    # Run inference to get features and predictions
    print("Extracting features and predictions for failure analysis...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Get embeddings using the helper method
            features = model.extract_features(inputs)
            outputs = model.fc(features)
            preds = torch.argmax(outputs, dim=1)

            all_features.append(features.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    val_features = np.concatenate(all_features, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

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

    return val_targets, preds


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting execution...")

    # 2. Data Loading
    print("\n[1/4] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Training
    print("\n[2/4] Training AnimalModel (Partial Fine-Tuning)...")
    model = train_model(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
    )

    # 4. Validation Assessment & Failure Analysis
    print("\n[3/4] Validating and Analyzing Failures...")

    # Run Failure Analysis (which also computes metrics)
    val_targets, val_preds = run_failure_analysis(model, val_loader)

    # Calculate and Print Metric
    final_metric = calculate_macro_f1(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Submission Generation
    # Only generate submission if metric is improved
    baseline_metric = 0.3674631922482031

    if final_metric > baseline_metric:
        print(
            f"\n[4/4] Generating Submission (Metric {final_metric:.6f} > {baseline_metric:.6f})..."
        )
        generate_submission(model, test_loader, output_path=Config.SUBMISSION_PATH)
    else:
        print(
            f"\n[4/4] Skipping Submission: Metric {final_metric:.6f} did not improve baseline {baseline_metric:.6f}."
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
