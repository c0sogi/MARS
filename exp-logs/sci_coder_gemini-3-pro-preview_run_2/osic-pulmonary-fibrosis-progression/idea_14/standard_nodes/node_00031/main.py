import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.train_eval import Trainer


def perform_failure_analysis(trainer, val_data):
    """
    Analyzes the model's performance on the validation set by correlating
    prediction errors with input features.
    """
    print("\n=== Failure Analysis ===")

    X_fvc = val_data["X_fvc"]
    y_true = val_data["y"]

    # Generate predictions
    y_pred = trainer.fvc_model.predict(X_fvc)

    # Calculate error magnitude (Absolute Error)
    error_magnitude = np.abs(y_true - y_pred)

    print(f"Mean Absolute Error on Validation: {np.mean(error_magnitude):.4f}")

    # Calculate correlations between features and error magnitude
    # X_fvc shape: (n_samples, n_features)
    n_features = X_fvc.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_fvc[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feature_col) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(feature_col, error_magnitude)
        correlations.append((i, corr))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 Features correlated with Error Magnitude:")
    print("(Positive correlation implies high feature value -> high error)")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # Also check correlation with metadata 'Weeks' if available separately
    if "weeks" in val_data:
        weeks = val_data["weeks"]
        corr_weeks, _ = pearsonr(weeks, error_magnitude)
        print(f"Correlation with 'Weeks' (Time): {corr_weeks:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Initialize Trainer
    trainer = Trainer()

    # 3. Load Data
    # load_cached_data=True ensures we use pre-computed features if available
    train_data, val_data, test_data = trainer.load_data(load_cached_data=True)

    # 4. Train Models
    trainer.train(train_data)

    # 5. Evaluate
    # The evaluate method prints the score, but we also need to return it
    # to print it in the specific format required.
    val_score = trainer.evaluate(val_data)
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    perform_failure_analysis(trainer, val_data)

    # 7. Conditional Submission
    # Threshold defined in task description
    THRESHOLD = -6.805292148096688

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_data)
    else:
        print(
            f"\nValidation score ({val_score}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )

    # 8. Save Models
    trainer.save_models()


if __name__ == "__main__":
    main()
