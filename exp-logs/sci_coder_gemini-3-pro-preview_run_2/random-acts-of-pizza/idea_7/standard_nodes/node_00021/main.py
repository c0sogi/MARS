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
    Helper function to perform GridSearchCV for a specific view.
    Maps parameters to the inner estimator of the BaggingClassifier.
    """
    # Initialize the base bagged model
    base_model = base_builder_func(random_state=random_state)

    # Map grid parameters to the inner estimator (estimator__param)
    # This is required because BaggingClassifier wraps the LogisticRegression
    sklearn_grid = {}
    for key, values in param_grid.items():
        if key in ["C", "penalty", "solver", "class_weight"]:
            sklearn_grid[f"estimator__{key}"] = values
        else:
            sklearn_grid[key] = values

    # Run Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=sklearn_grid,
        scoring="roc_auc",
        cv=3,  # Inner CV for hyperparameter tuning
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
    # load_cached_data=True ensures we use pre-computed features if available
    X_text_train, X_meta_train, X_text_val, X_meta_val, X_text_test, X_meta_test = (
        extract_features(df_train, df_val, df_test, load_cached_data=True)
    )

    # Prepare targets and feature arrays
    y_train = df_train["requester_received_pizza"].values
    y_val = df_val["requester_received_pizza"].values

    # Early Fusion: Concatenate Text and Metadata
    # Cite solution_lesson_node_00020: Early Fusion outperforms Late Fusion on small datasets
    print("Concatenating features for Early Fusion...")
    X_train = np.hstack([X_text_train, X_meta_train.values])
    X_val = np.hstack([X_text_val, X_meta_val.values])
    X_test = np.hstack([X_text_test, X_meta_test.values])

    # 4. Cross-Validation Loop
    n_folds = Config.N_FOLDS
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED
    )

    # Initialize prediction containers
    val_preds_accum = np.zeros(len(y_val))
    test_preds_accum = np.zeros(len(df_test))

    # Store OOF preds for diagnostic (optional, but good for consistency)
    oof_preds = np.zeros(len(y_train))

    print(f"Starting Training with {n_folds} folds...")

    for fold, (train_idx, holdout_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Processing Fold {fold + 1}/{n_folds}")

        # Create Fold Splits
        X_tr, X_ho = X_train[train_idx], X_train[holdout_idx]
        y_tr, y_ho = y_train[train_idx], y_train[holdout_idx]

        # Tune and Train Bagged Logistic Regression on Combined Features
        # Cite solution_lesson_node_00019: Bagged LR with RankGauss (applied in feature extraction)
        best_model = get_best_estimator(
            X_tr,
            y_tr,
            Config.COMBINED_GRID,
            build_bagged_logistic_regression,
            Config.RANDOM_SEED,
        )

        # Predict OOF
        oof_preds[holdout_idx] = best_model.predict_proba(X_ho)[:, 1]

        # Predict on Fixed Validation Set and Test Set
        val_preds_accum += best_model.predict_proba(X_val)[:, 1]
        test_preds_accum += best_model.predict_proba(X_test)[:, 1]

    # Average predictions from all folds
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
    threshold = 0.6994047619047619

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
