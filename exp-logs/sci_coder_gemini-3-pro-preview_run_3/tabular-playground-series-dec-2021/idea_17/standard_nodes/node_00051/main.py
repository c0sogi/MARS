import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.train import run_training, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_feature_names():
    """
    Reconstructs the feature names list corresponding to the processed numpy array
    to enable interpretable failure analysis.
    """
    try:
        # Load a small sample to get raw column names
        df = pd.read_parquet(Config.TRAIN_PATH)

        # Drop non-feature columns as done in data_loader
        df = df.drop(columns=["Id", "Cover_Type"], errors="ignore")

        # Define new engineered features as per library/data_loader.py
        new_continuous = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Euclidean_Distance_To_Hydrology",
            "Absolute_Hydrology_Elevation",
            "Mean_Distance_To_Amenities",
        ]

        # Construct list of continuous features (Original + New)
        continuous_cols = Config.CONTINUOUS_FEATURES + new_continuous

        # Identify binary columns (Raw columns that are not in the continuous list)
        raw_cols = df.columns.tolist()
        binary_cols = [c for c in raw_cols if c not in Config.CONTINUOUS_FEATURES]

        # The data_loader stacks continuous then binary
        feature_names = continuous_cols + binary_cols
        return feature_names

    except Exception as e:
        print(f"Could not reconstruct feature names: {e}")
        return None


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)
    device = get_device()

    print("Starting Runfile Execution...")

    # 2. Training
    # We use the full dataset (debug=False) to ensure we meet the high accuracy threshold.
    # The A100 GPU is sufficient to train 60 epochs on this tabular data within the time limit.
    print("Initiating Training Pipeline...")
    model, test_loader, test_ids = run_training(
        epochs=Config.EPOCHS, load_cached_data=True, debug=False
    )

    # 3. Validation Evaluation
    print("\nRunning Final Validation...")
    # Load validation data (using cache generated during training)
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_probs.append(probs.cpu())
            all_targets.append(labels.cpu())

    val_acc = correct / total

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    y_true = torch.cat(all_targets).numpy()
    y_probs = torch.cat(all_probs).numpy()

    # Error Magnitude = 1 - Probability assigned to the True Class
    # This provides a continuous signal of how "wrong" or "uncertain" the model was
    rows = np.arange(len(y_true))
    prob_correct = y_probs[rows, y_true]
    error_magnitude = 1.0 - prob_correct

    # Get Feature Matrix from the dataset
    X_val = val_loader.dataset.X.numpy()

    # Calculate Correlations between Features and Error Magnitude
    n_samples = len(error_magnitude)

    # Center X and error vector
    X_mean = X_val.mean(axis=0)
    X_centered = X_val - X_mean

    err_mean = error_magnitude.mean()
    err_centered = error_magnitude - err_mean

    # Covariance: (X_centered.T @ err_centered) / (n-1)
    covariance = np.dot(X_centered.T, err_centered) / (n_samples - 1)

    # Standard Deviations
    X_std = X_val.std(axis=0, ddof=1)
    err_std = error_magnitude.std(ddof=1)

    # Handle zero std dev to avoid division by zero
    X_std[X_std == 0] = 1e-10
    if err_std == 0:
        err_std = 1e-10

    correlations = covariance / (X_std * err_std)

    # Get Feature Names for display
    feature_names = get_feature_names()

    # Sort by absolute correlation magnitude
    abs_corrs = np.abs(correlations)
    top_indices = np.argsort(abs_corrs)[::-1][:10]  # Top 10 features

    print("Top Feature Correlations with Error Magnitude:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Feature':<40} {'Correlation':<12}")
    for i, idx in enumerate(top_indices):
        fname = (
            feature_names[idx]
            if feature_names and idx < len(feature_names)
            else f"Feature_{idx}"
        )
        print(f"{i+1:<5} {fname[:39]:<40} {correlations[idx]:.6f}")
    print("-" * 60)

    # 5. Submission Generation
    # Threshold defined in task
    TARGET_THRESHOLD = 0.9625041666666667

    if val_acc > TARGET_THRESHOLD:
        print(
            f"\nValidation metric ({val_acc:.6f}) meets threshold ({TARGET_THRESHOLD:.6f})."
        )
        generate_submission(model, test_loader, test_ids)
    else:
        print(
            f"\nValidation metric ({val_acc:.6f}) does NOT meet threshold ({TARGET_THRESHOLD:.6f})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
