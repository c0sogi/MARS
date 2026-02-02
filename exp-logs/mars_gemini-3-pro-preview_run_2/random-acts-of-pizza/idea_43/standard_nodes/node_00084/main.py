import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.feature_engineering import FeatureEngineer
from library.preprocessing import HAMFPreprocessor
from library.model_factory import ModelFactory


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    # 2. Load Features
    # We use load_cached_data=True to leverage pre-computed embeddings if available
    fe = FeatureEngineer()
    features = fe.build_feature_set(load_cached_data=True)

    # 3. Prepare Data for Cross-Validation
    # The FeatureEngineer returns fixed train/val splits based on metadata.
    # We combine them here to perform our own 5-Fold CV on the full available labeled data.
    train_feat = features["train"]
    val_feat = features["val"]

    # Identify feature keys (excluding target 'y')
    feature_keys = [k for k in train_feat.keys() if k != "y"]

    # Concatenate train and val data
    full_data = {}
    for k in feature_keys:
        if k in val_feat:
            full_data[k] = np.concatenate([train_feat[k], val_feat[k]], axis=0)
        else:
            full_data[k] = train_feat[k]

    y = np.concatenate([train_feat["y"], val_feat["y"]], axis=0)

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []
    oof_preds = np.zeros(len(y))
    model_factory = ModelFactory()

    logger.info("Starting 5-Fold CV Training...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Slice data
        X_train = {k: full_data[k][train_idx] for k in feature_keys}
        y_train = y[train_idx]
        X_val = {k: full_data[k][val_idx] for k in feature_keys}
        y_val = y[val_idx]

        # Preprocessing
        # Fit on train, transform train and val
        preprocessor = HAMFPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        # Model Creation & Tuning
        clf = model_factory.create_classifier()
        param_grid = model_factory.get_hyperparameter_grid()

        # Use GridSearchCV for hyperparameter tuning within the fold
        # n_jobs=-1 uses all CPUs, verbose=0 keeps it silent
        gs = GridSearchCV(
            clf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        gs.fit(X_train_proc, y_train)

        best_model = gs.best_estimator_
        logger.info(f"Best params: {gs.best_params_}")

        # Validation Prediction
        preds = best_model.predict_proba(X_val_proc)[:, 1]
        oof_preds[val_idx] = preds

        score = roc_auc_score(y_val, preds)
        fold_scores.append(score)
        logger.info(f"Fold {fold + 1} AUC: {score:.6f}")

        # Save Artifacts
        joblib.dump(
            best_model,
            os.path.join(Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"),
        )
        joblib.dump(
            preprocessor,
            os.path.join(Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"),
        )

    # 5. Global Evaluation
    avg_auc = np.mean(fold_scores)
    print(f"Final Validation Metric: {avg_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y - oof_preds)

    # Correlate errors with numeric metadata features
    meta_matrix = full_data["metadata"]
    correlations = {}

    for i, col_name in enumerate(Config.NUMERIC_COLS):
        col_values = meta_matrix[:, i]
        # Avoid correlation calculation if constant
        if np.std(col_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(col_values, errors)[0, 1]
            correlations[col_name] = corr
        else:
            correlations[col_name] = 0.0

    print("Failure Analysis - Correlation of Error with Features:")
    for k, v in sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        print(f"{k}: {v:.6f}")

    # 7. Conditional Submission
    threshold = 0.7201989696216022
    if avg_auc > threshold:
        logger.info(
            f"Validation metric {avg_auc} > {threshold}. Generating submission..."
        )

        test_feat = features["test"]
        request_ids = test_feat["request_id"]

        # Filter test features for valid keys
        X_test_raw = {k: test_feat[k] for k in feature_keys if k in test_feat}

        fold_test_preds = []

        # Predict using all fold models
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"
            )
            proc_path = os.path.join(
                Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"
            )

            model = joblib.load(model_path)
            proc = joblib.load(proc_path)

            X_test_proc = proc.transform(X_test_raw)
            p = model.predict_proba(X_test_proc)[:, 1]
            fold_test_preds.append(p)

        # Average predictions (CV-Bagging)
        avg_test_preds = np.mean(fold_test_preds, axis=0)

        # Create submission DataFrame
        df_sub = pd.DataFrame(
            {"request_id": request_ids, "requester_received_pizza": avg_test_preds}
        )

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(f"Validation metric {avg_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
