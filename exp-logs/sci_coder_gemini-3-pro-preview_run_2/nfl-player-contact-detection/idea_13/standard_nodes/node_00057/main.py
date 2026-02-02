import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.train import Trainer
from library.feature_engineering import (
    generate_train_val_features,
    generate_test_features,
)
from library.model import WideResNetMLP
from library.dataset import get_dataloader


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    # Reducing epochs ensures the run completes well within the time limit
    Config.EPOCHS = 10

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 2. Training
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    # Execute training (handles data loading, training loop, and saving best model)
    trainer.fit(load_cached_data=True)

    # 3. Final Evaluation on Validation Set
    print("\n=== Starting Final Evaluation ===")

    # Reload validation data to ensure we have the exact set used for metrics
    # We ignore training data here to save memory
    _, _, X_val, y_val = generate_train_val_features(load_cached_data=True)

    device = torch.device(Config.DEVICE)
    input_dim = X_val.shape[1]

    # Load the best model saved by the Trainer
    model = WideResNetMLP(input_dim=input_dim)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Create DataLoader for inference
    val_loader = get_dataloader(
        X_val,
        y_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    all_probs = []
    all_targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            # Forward pass
            logits = model(features).squeeze(1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    # Optimize Threshold on full validation set
    best_threshold, best_mcc = optimize_threshold(y_true, y_probs)

    # Print Required Metric
    print(f"Final Validation Metric: {best_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_true - y_probs)

    # Calculate correlation between features and error
    # We approximate correlation for efficiency: Cov(X, E) / (Std(X)*Std(E))
    # Since X is standardized (Std(X)~1), Corr ~ Cov(X, E) / Std(E)

    print("Calculating feature-error correlations...")
    error_mean = np.mean(errors)
    error_std = np.std(errors)

    correlations = []
    num_features = X_val.shape[1]

    # Vectorized correlation calculation
    # Center the errors
    errors_centered = errors - error_mean

    # Calculate covariance: mean(X * centered_error)
    # We do this in chunks to avoid creating a massive matrix if X_val is huge,
    # though 800k x 500 fits in memory.
    covariance = np.mean(X_val * errors_centered[:, np.newaxis], axis=0)

    # Correlation
    feature_correlations = covariance / (error_std + 1e-9)

    # Get feature names
    try:
        # Attempt to read column names from the cached parquet file
        df_cols = pd.read_parquet(Config.VAL_FEATURES_PATH).columns.tolist()
        if "target" in df_cols:
            df_cols.remove("target")
        feature_names = np.array(df_cols)
    except Exception as e:
        print(f"Could not load feature names: {e}. Using indices.")
        feature_names = np.array([f"feature_{i}" for i in range(num_features)])

    # Sort by absolute correlation
    sorted_indices = np.argsort(-np.abs(feature_correlations))

    print("Top 10 features correlated with prediction error:")
    for i in sorted_indices[:10]:
        print(f"  {feature_names[i]}: {feature_correlations[i]:.6f}")

    # 5. Submission Generation
    THRESHOLD_SCORE = 0.62458462731896

    if best_mcc > THRESHOLD_SCORE:
        print(
            f"\nMetric ({best_mcc}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load Test Data
        X_test, test_ids = generate_test_features(load_cached_data=True)

        test_loader = get_dataloader(
            X_test,
            targets=None,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_probs = []

        print("Running test inference...")
        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                logits = model(features).squeeze(1)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.cpu().numpy())

        test_probs = np.concatenate(test_probs)

        # Apply optimized threshold
        predictions = (test_probs >= best_threshold).astype(int)

        # Construct Submission DataFrame
        submission = test_ids.copy()
        submission["contact"] = predictions

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({best_mcc}) <= Threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
