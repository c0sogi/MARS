import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# Import functions and constants from the provided library files
from library.config import SEED
from library.utils import seed_everything
from library.data_loader import load_dataset
from library.model_trainer import run_stratified_cv, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup Environment
    seed_everything(SEED)

    # 2. Load Data
    # We load cached features to speed up the process.
    # The feature engineering pipeline (Shift-Invariant Orthogonal Decomposition)
    # has already processed these and stored them in ./working/idea_16/
    print("Loading datasets...")
    X_train, y_train = load_dataset("train", load_cached_data=True)
    X_val, y_val = load_dataset("val", load_cached_data=True)
    X_test, _ = load_dataset("test", load_cached_data=True)

    # 3. Train Models
    # We use Stratified K-Fold CV on the training set.
    # We pass X_test to generate test predictions efficiently within the loop.
    # The dataset size (~3200 samples) is small enough that we don't need to subsample
    # for speed; LightGBM will handle this very quickly.
    print("Training LightGBM ensemble with Stratified K-Fold CV...")
    _, test_preds_avg, models = run_stratified_cv(
        X_train, y_train, n_folds=5, test_X=X_test
    )

    # 4. Validation Assessment
    # We must evaluate on the specific hold-out validation set loaded from metadata/val.csv
    print("Evaluating on hold-out validation set...")

    # Generate predictions for the validation set using the ensemble
    val_preds = np.zeros(len(X_val))
    for model in models:
        # LightGBM inference
        val_preds += model.predict(X_val, num_iteration=model.best_iteration)

    # Average the predictions across all folds
    val_preds /= len(models)

    # Calculate and print the required metric
    final_metric = mean_absolute_error(y_val, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Identify which features correlate most with prediction errors
    print("Performing failure analysis...")
    abs_errors = np.abs(y_val - val_preds)

    correlations = []
    for col in X_val.columns:
        # Calculate Pearson correlation, handling potential constant columns
        if X_val[col].std() > 1e-9:
            corr, _ = pearsonr(X_val[col], abs_errors)
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error magnitude:")
    for feat, corr in correlations[:5]:
        print(f"{feat}: {corr:.4f}")

    # 6. Submission Generation
    # Only generate submission if the model meets the performance requirement
    THRESHOLD = 2617304.0647319085

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} meets threshold. Generating submission..."
        )
        generate_submission(X_test.index, test_preds_avg)
    else:
        print(
            f"Validation metric {final_metric} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
