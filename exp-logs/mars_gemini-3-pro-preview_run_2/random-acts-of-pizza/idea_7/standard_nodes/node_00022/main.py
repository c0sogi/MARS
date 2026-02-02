import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_and_merge_data
from library.feature_extraction import extract_features
from library.model_factory import build_bagged_logistic_regression


def get_best_estimator(X, y, param_grid, base_builder_func, random_state):
    """
    Helper function to perform GridSearchCV.
    Maps parameters to the inner estimator of the BaggingClassifier.
    """
    # Initialize the base bagged model
    base_model = base_builder_func(random_state=random_state)

    # Map grid parameters to the inner estimator (estimator__param)
    sklearn_grid = {}
    for key, values in param_grid.items():
        if key in ["C", "penalty", "solver", "class_weight"]:
            sklearn_grid[f"estimator__{key}"] = values
        else:
            sklearn_grid[key] = values

    # Run Grid Search
    # n_jobs=-1 here allows parallelizing the grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=sklearn_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)

    # 2. Load Data
    print("Loading Data...")
    df_train, df_val, df_test = load_and_merge_data()

    # 3. Extract Features
    print("Extracting Features...")
    X_text_train, X_meta_train, X_text_val, X_meta_val, X_text_test, X_meta_test = (
        extract_features(df_train, df_val, df_test, load_cached_data=True)
    )

    # Prepare targets
    y_train = df_train["requester_received_pizza"].values
    y_val = df_val["requester_received_pizza"].values

    # 4. Early Fusion (Concatenation)
    print("Concatenating Features (Early Fusion)...")
    # Convert metadata DataFrames to numpy for concatenation
    X_meta_train_np = X_meta_train.values
    X_meta_val_np = X_meta_val.values
    X_meta_test_np = X_meta_test.values

    X_train_full = np.hstack((X_text_train, X_meta_train_np))
    X_val_full = np.hstack((X_text_val, X_meta_val_np))
    X_test_full = np.hstack((X_text_test, X_meta_test_np))

    # 5. Cross-Validation Loop
    n_folds = Config.N_FOLDS
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED
    )

    val_preds_accum = np.zeros(len(y_val))
    test_preds_accum = np.zeros(len(df_test))

    print(f"Starting Training with {n_folds} folds on Combined Features...")

    for fold, (train_idx, holdout_idx) in enumerate(skf.split(X_train_full, y_train)):
        print(f"Processing Fold {fold + 1}/{n_folds}")

        # Create Fold Splits
        X_tr, X_ho = X_train_full[train_idx], X_train_full[holdout_idx]
        y_tr, y_ho = y_train[train_idx], y_train[holdout_idx]

        # Tune and Train on Combined Features
        # Using the Combined Grid to find optimal regularization for the hybrid feature set
        best_model = get_best_estimator(
            X_tr,
            y_tr,
            Config.COMBINED_GRID,
            build_bagged_logistic_regression,
            Config.RANDOM_SEED,
        )

        # Predict
        val_preds_accum += best_model.predict_proba(X_val_full)[:, 1]
        test_preds_accum += best_model.predict_proba(X_test_full)[:, 1]

    # Average predictions
    final_val_probs = val_preds_accum / n_folds
    final_test_probs = test_preds_accum / n_folds

    # 6. Validation Evaluation
    val_auc = roc_auc_score(y_val, final_val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_probs)

    # Create a DataFrame for correlation analysis using the validation metadata features
    analysis_df = X_meta_val.copy()
    analysis_df["error"] = errors

    # Compute correlation between features and error
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(key=abs, ascending=False)
    )

    print("Correlation between Error and Features (Top 10):")
    print(correlations.head(10))

    # 8. Submission Generation
    threshold = 0.7129152602593712

    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")

        # Load test metadata to get request_ids
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": df_test_meta["request_id"],
                "requester_received_pizza": final_test_probs,
            }
        )

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {val_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
