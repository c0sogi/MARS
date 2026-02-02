import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import TARGET_COL, ID_COL, NUMERIC_COLS, RANDOM_STATE
from library.utils import set_seed, save_submission
from library.data_loader import load_data
from library.model import PizzaRandomForest


def run_failure_analysis(df, y_true, y_pred, numeric_cols):
    """
    Analyzes the correlation between prediction errors and numerical features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error (residuals)
    # y_true is boolean/int (0 or 1), y_pred is probability [0, 1]
    errors = np.abs(y_true - y_pred)

    correlations = {}

    # Calculate correlation for each numeric feature
    # We filter to ensure the column exists in the dataframe
    valid_cols = [col for col in numeric_cols if col in df.columns]

    for col in valid_cols:
        # Handle potential NaNs just in case, though preprocessor handles them for the model
        if df[col].isnull().any():
            feat_vals = df[col].fillna(df[col].median())
        else:
            feat_vals = df[col]

        # Compute correlation
        corr = np.corrcoef(feat_vals, errors)[0, 1]

        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation (descending)
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top features correlated with prediction error:")
    for name, val in sorted_corrs[:5]:
        print(f"{name:<50}: {val:.4f}")

    return sorted_corrs


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)
    print("Starting pipeline...")

    # 2. Data Loading
    # We use cached data if available for speed
    train_df, val_df, test_df = load_data(load_cached_data=False)

    print(
        f"Data Loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # 3. Model Initialization and Training
    # Using the wrapper class provided in library.model
    model = PizzaRandomForest(random_state=RANDOM_STATE)

    # Train the model
    # Note: The dataset is small enough (~2k rows) that we don't need to subsample for a fast baseline.
    model.train(train_df, TARGET_COL)

    # 4. Validation
    print("Evaluating on Validation set...")
    # evaluate() returns the predicted probabilities
    val_probs = model.evaluate(val_df, TARGET_COL, set_name="Validation")

    # Calculate final metric explicitly to ensure format compliance
    y_val = val_df[TARGET_COL]
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    # Identify which features correlate with model errors
    # We use the NUMERIC_COLS defined in config
    run_failure_analysis(val_df, y_val, val_probs, NUMERIC_COLS)

    # 6. Submission Generation
    print("\nGenerating predictions for Test set...")
    test_probs = model.predict_proba(test_df)

    # Extract IDs for submission
    test_ids = test_df[ID_COL]

    # Save submission
    save_submission(test_ids, test_probs)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
