import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_processor import process_data, engineer_static_features
from library.feature_learner import SupervisedProjector
from library.model_handler import ModelTrainer


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("Initializing Cover Type Prediction Pipeline...")

    # 2. Data Loading and Processing
    # Load Train (80%) and Test (Unlabeled) using the cached processor
    print("Loading training and test data...")
    train_df, test_df = process_data(load_cached_data=True)

    # Load Hold-out Validation (20%)
    # We must manually process this to ensure feature consistency
    print("Loading and processing hold-out validation data...")
    val_path = "./metadata/val.csv"
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"{val_path} not found. Metadata generation required.")

    val_df = pd.read_csv(val_path)
    val_df = engineer_static_features(val_df)

    # Define columns
    target_col = "Cover_Type"
    id_col = "Id"

    # Extract feature names (excluding Id and Target)
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    print(f"Feature count: {len(feature_cols)}")

    # Helper to convert DF to Numpy
    def get_data_arrays(df, features, target=None):
        X = df[features].values
        y = df[target].values if target in df.columns else None
        return X, y

    # Prepare Data Arrays
    X_train_full, y_train_full = get_data_arrays(train_df, feature_cols, target_col)
    X_holdout, y_holdout = get_data_arrays(val_df, feature_cols, target_col)
    X_test, _ = get_data_arrays(test_df, feature_cols, target_col)

    # Extract Test IDs for submission
    if id_col in test_df.columns:
        test_ids = test_df[id_col].values
    else:
        test_ids = np.arange(4000000, 4000000 + len(test_df))

    # release memory
    del train_df, val_df, test_df
    gc.collect()

    # 3. Target Encoding
    # Fit LabelEncoder on all available targets to ensure consistency
    le = LabelEncoder()
    all_targets = np.unique(np.concatenate([y_train_full, y_holdout]))
    le.fit(all_targets)

    y_train_enc = le.transform(y_train_full)
    y_holdout_enc = le.transform(y_holdout)
    n_classes = len(le.classes_)
    print(f"Target Classes: {le.classes_} (Internal: {np.arange(n_classes)})")

    # 4. Stratified K-Fold Ensemble Training
    # We split the 80% training data into K folds.
    # The models are ensembled to predict on the 20% holdout set and the test set.
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Probability Accumulators for Soft Voting
    holdout_probs_sum = np.zeros((len(X_holdout), n_classes), dtype=np.float32)
    test_probs_sum = np.zeros((len(X_test), n_classes), dtype=np.float32)

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_enc)):
        print(f"\n--- Fold {fold + 1} / {Config.N_FOLDS} ---")

        # Split Fold Data
        X_tr = X_train_full[train_idx]
        y_tr = y_train_enc[train_idx]
        X_va = X_train_full[val_idx]
        y_va = y_train_enc[val_idx]

        # --- Supervised Feature Learning (LDA) ---
        # Fit LDA only on the training portion of the fold to prevent leakage
        print("Fitting LDA projection...")
        lda = SupervisedProjector(n_components=Config.LDA_COMPONENTS)
        lda.fit(X_tr, y_tr)

        # Project all datasets using the learned transformation
        X_tr_lda = lda.transform(X_tr)
        X_va_lda = lda.transform(X_va)
        X_holdout_lda = lda.transform(X_holdout)
        X_test_lda = lda.transform(X_test)

        # Augment original features with LDA components
        X_tr_final = np.hstack([X_tr, X_tr_lda])
        X_va_final = np.hstack([X_va, X_va_lda])
        X_holdout_final = np.hstack([X_holdout, X_holdout_lda])
        X_test_final = np.hstack([X_test, X_test_lda])

        # --- Model Training ---
        print("Training XGBoost...")
        trainer = ModelTrainer()
        # Train with early stopping on the fold validation set
        trainer.train(X_tr_final, y_tr, X_va_final, y_va)

        # --- Inference ---
        print("Inference on Hold-out and Test sets...")
        holdout_probs_sum += trainer.predict_proba(X_holdout_final)
        test_probs_sum += trainer.predict_proba(X_test_final)

        # Cleanup to prevent OOM
        del X_tr, y_tr, X_va, y_va
        del X_tr_lda, X_va_lda, X_holdout_lda, X_test_lda
        del X_tr_final, X_va_final, X_holdout_final, X_test_final
        del trainer, lda
        gc.collect()

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Hold-out Set ---")
    # Soft Voting
    avg_holdout_probs = holdout_probs_sum / Config.N_FOLDS
    holdout_preds_idx = np.argmax(avg_holdout_probs, axis=1)

    final_metric = accuracy_score(y_holdout_enc, holdout_preds_idx)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = (holdout_preds_idx != y_holdout_enc).astype(int)
    print(f"Total Errors: {errors.sum()} / {len(errors)} (Rate: {errors.mean():.6f})")

    print("Calculating feature correlations with prediction error...")
    correlations = []
    # X_holdout contains the original static features
    for i, feature_name in enumerate(feature_cols):
        feat_values = X_holdout[:, i]
        # Skip constant features to avoid division by zero in correlation
        if np.std(feat_values) > 1e-9:
            # Point-biserial correlation
            corr = np.corrcoef(feat_values, errors)[0, 1]
            correlations.append((feature_name, corr))
        else:
            correlations.append((feature_name, 0.0))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Associated with Error:")
    for name, corr in correlations[:10]:
        print(f"{name:<40}: {corr:.6f}")

    # 7. Submission Generation
    threshold = 0.9625222092091004
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )

        avg_test_probs = test_probs_sum / Config.N_FOLDS
        test_preds_idx = np.argmax(avg_test_probs, axis=1)
        test_preds = le.inverse_transform(test_preds_idx)

        save_submission(test_ids, test_preds)
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
