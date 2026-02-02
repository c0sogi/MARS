import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from library.config import C_GRID, SUBMISSION_DIR, SEED
from library.data_loader import load_dataset_with_metadata
from library.feature_engineering import get_features, NUMERIC_FEATURES
from library.model_builder import create_bagged_linear_model


def run_stratified_cv(n_splits=5, load_cached_data=True):
    """
    Performs Stratified Cross-Validation to tune the regularization parameter C.
    Ensures strict separation of tabular scaling to prevent data leakage.
    """
    print("Starting Stratified Cross-Validation...")

    # 1. Load Precomputed Features (Text + Scaled Numerics)
    # We load this primarily to get the expensive text embeddings without re-computing them.
    X_all, y = get_features("train", load_cached_data=load_cached_data)

    # 2. Load Raw Data to get Raw Numerics
    # We need raw numerics to fit the QuantileTransformer strictly on training folds.
    # get_features returns numerics scaled on the full dataset, which is leakage for CV.
    df_train = load_dataset_with_metadata("train", load_cached_data=load_cached_data)

    # Extract Raw Numerics (fill NaNs with 0 as per FeatureEngineer logic)
    X_num_raw = df_train[NUMERIC_FEATURES].fillna(0).values

    # Extract Text Embeddings
    # The feature matrix X_all is [Text_Embeddings | Scaled_Numerics]
    n_numeric = len(NUMERIC_FEATURES)
    n_text = X_all.shape[1] - n_numeric
    X_text = X_all[:, :n_text]

    print(f"Data Loaded: {X_all.shape[0]} samples.")
    print(f"Feature Dimensions -> Text: {n_text}, Numeric: {n_numeric}")

    # 3. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = {}

    for C in C_GRID:
        print(f"\nEvaluating C={C}...")
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
            # Split Data
            X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
            X_num_train_raw, X_num_val_raw = X_num_raw[train_idx], X_num_raw[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Process Numerics strictly on training fold
            # We use the same configuration as the main FeatureEngineer (Quantile, Normal)
            scaler = QuantileTransformer(
                output_distribution="normal", random_state=SEED
            )

            # Fit on train, transform train and val
            X_num_train_scaled = scaler.fit_transform(X_num_train_raw)
            X_num_val_scaled = scaler.transform(X_num_val_raw)

            # Concatenate Text and Scaled Numerics
            X_train_fold = np.hstack([X_text_train, X_num_train_scaled])
            X_val_fold = np.hstack([X_text_val, X_num_val_scaled])

            # Train Model
            model = create_bagged_linear_model(C=C)
            model.fit(X_train_fold, y_train_fold)

            # Predict
            # BaggingClassifier.predict_proba averages the probabilities of base estimators
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            # Evaluate
            auc = roc_auc_score(y_val_fold, y_pred_proba)
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"C={C} -> Mean AUC: {mean_auc:.10f} (Std: {std_auc:.10f})")
        results[C] = mean_auc

    # Find Best C
    best_C = max(results, key=results.get)
    print(f"\nBest C: {best_C} with AUC: {results[best_C]:.10f}")

    return best_C, results[best_C]


def train_and_predict_submission(best_C, load_cached_data=True):
    """
    Retrains the model on the full training set using the best C and generates
    predictions for the test set.
    """
    print(f"\nRetraining final model with C={best_C} on full dataset...")

    # 1. Load Full Train Data
    # For the final model, we want to scale numerics based on the full training set.
    # get_features('train') returns exactly this (Text + Numerics scaled on full train).
    X_train, y_train = get_features("train", load_cached_data=load_cached_data)

    # 2. Train Model
    model = create_bagged_linear_model(C=best_C)
    model.fit(X_train, y_train)

    # 3. Load Test Data
    # get_features('test') automatically loads train data to fit the scaler,
    # then transforms the test data. This ensures consistent scaling.
    X_test, _ = get_features("test", load_cached_data=load_cached_data)

    print(f"Predicting on {X_test.shape[0]} test samples...")

    # 4. Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 5. Create Submission DataFrame
    # We need to map predictions back to request_ids.
    # The test metadata file contains the correct order of request_ids.
    df_test_meta = pd.read_csv("./metadata/test.csv")

    submission = pd.DataFrame(
        {
            "request_id": df_test_meta["request_id"],
            "requester_received_pizza": y_pred_proba,
        }
    )

    # 6. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return submission
