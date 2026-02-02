import sys
import os
import numpy as np
import random
from scipy.stats import pearsonr

# Ensure the current directory is in the python path to load library modules correctly
sys.path.append(os.getcwd())

from library.trainer import Trainer
from library.config import Config


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Set Random Seed
    set_seed(Config.RANDOM_SEED)

    # 2. Initialize the Trainer
    trainer = Trainer()

    # 3. Train the Model
    # We use the full dataset (max_samples=None) because the dataset is small (206 training samples).
    # Limiting samples here would degrade performance without providing meaningful speed gains.
    print("Starting training pipeline...")
    val_metric = trainer.train(load_cached_data=True, max_samples=None)

    # 4. Print Final Validation Metric
    # Strict requirement: Print full precision without rounding.
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Retrieve validation data manually to perform analysis
    # The trainer.loader has access to the split logic
    (_, _), (X_val, y_val), _ = trainer.loader.get_data_splits(load_cached_data=True)

    # Generate predictions on the validation set
    # The model is already trained from the previous step
    y_val_pred = trainer.model.predict_proba(X_val)

    # Calculate Error Magnitude
    # We define error magnitude as the Mean Absolute Error (MAE) per sample,
    # averaged across all 19 species.
    # y_val and y_val_pred are shape (n_samples, n_species)
    sample_errors = np.mean(np.abs(y_val - y_val_pred), axis=1)

    # Calculate Pearson Correlation between Error Magnitude and Input Features
    # X_val is shape (n_samples, n_features)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vec = X_val[:, i]

        # Check for constant features to avoid division by zero in correlation calculation
        if np.std(feature_vec) == 0 or np.std(sample_errors) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_vec, sample_errors)

        correlations.append((i, corr))

    # Sort features by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(
        "Top 5 Features correlated with Error Magnitude (indicates potential systematic failure modes):"
    )
    for feat_idx, corr in correlations[:5]:
        print(f"Feature {feat_idx}: Correlation = {corr}")

    # 6. Generate Submission
    print("\nGenerating submission file...")
    trainer.generate_submission(load_cached_data=True, max_samples=None)
    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()
