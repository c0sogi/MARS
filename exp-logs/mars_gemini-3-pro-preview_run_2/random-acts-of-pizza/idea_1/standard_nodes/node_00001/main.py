import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from provided library modules
from library.config import RANDOM_SEED
from library.utils import set_seed, save_submission
from library.feature_engineering import get_processed_data
from library.model import PizzaRandomForest


def run_failure_analysis(X_val, y_val, y_pred):
    """
    Analyzes the correlation between prediction error and input features.
    """
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (Absolute Error for binary classification probabilities)
    # Error = |True Label - Predicted Probability|
    errors = np.abs(y_val - y_pred)

    # Calculate correlation between each feature and the error
    # X_val is a DataFrame, so we can use corrwith
    correlations = X_val.corrwith(pd.Series(errors, index=X_val.index))

    # Sort by absolute correlation to find most impactful features on error
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 10 features correlated with prediction error:")
    for feature_name, abs_corr in abs_corrs.head(10).items():
        actual_corr = correlations[feature_name]
        print(f"{feature_name}: {actual_corr:.6f}")
    print("========================\n")


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Load Data
    # load_cached_data=True will use parquet files if they exist in ./working
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = get_processed_data(
        load_cached_data=True
    )

    # 3. Model Training
    # The dataset size is small (~2300 samples), so we use the full set.
    # Training Random Forest on this size is extremely fast (< 10 seconds).
    print(f"Training model on {len(X_train)} samples...")
    model = PizzaRandomForest()
    model.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("Performing validation inference...")
    val_preds = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    run_failure_analysis(X_val, y_val, val_preds)

    # 6. Submission Generation
    print("Generating test predictions...")
    test_preds = model.predict_proba(X_test)

    print("Saving submission...")
    save_submission(test_ids, test_preds)


if __name__ == "__main__":
    main()
