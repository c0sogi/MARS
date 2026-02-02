import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import SEED
from library.feature_engineering import get_features, NUMERIC_FEATURES
from library.model_builder import create_bagged_linear_model
from library.training_utils import run_stratified_cv, train_and_predict_submission


def set_seed(seed=42):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    # 2. Feature Generation / Loading
    # We load features for all splits. The library handles caching and scaling.
    # X arrays contain [MPNet Embeddings (768) | Scaled Numerics]
    print("Loading/Generating Features...")
    X_train, y_train = get_features("train", load_cached_data=True)
    X_val, y_val = get_features("val", load_cached_data=True)
    # Test features are needed later for submission, loading now ensures cache is ready
    X_test, _ = get_features("test", load_cached_data=True)

    # 3. Hyperparameter Tuning (Cross-Validation on Train Split)
    # run_stratified_cv performs 5-fold CV on the X_train data to find best C
    print("\nRunning Cross-Validation to tune regularization...")
    best_C, cv_auc = run_stratified_cv(n_splits=5, load_cached_data=True)

    # 4. Train Final Model on Train Split and Evaluate on Hold-out Val Split
    print(f"\nTraining model with best C={best_C} on full training set...")
    model = create_bagged_linear_model(C=best_C)
    model.fit(X_train, y_train)

    print("Evaluating on hold-out validation set...")
    # Predict probabilities for the positive class
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Calculate absolute error: |y_true - y_pred|
    # Note: y_true is 0 or 1. If y=1 and pred=0.8, error=0.2. If y=0 and pred=0.8, error=0.8.
    errors = np.abs(y_val - y_val_pred_proba)

    # We correlate errors with the numerical features.
    # The numerical features are at the end of the X matrix.
    # X structure: [Text (768) | Numerics (N)]
    num_feat_count = len(NUMERIC_FEATURES)
    X_val_numerics = X_val[:, -num_feat_count:]

    # Create a DataFrame for correlation calculation
    df_analysis = pd.DataFrame(X_val_numerics, columns=NUMERIC_FEATURES)
    df_analysis["error"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Top Feature Correlations with Prediction Error:")
    print(correlations.head(5))

    # 6. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.6994047619047619

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # train_and_predict_submission handles retraining on 'train' and predicting on 'test'
        # and saving to ./submission/submission.csv
        train_and_predict_submission(best_C, load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
