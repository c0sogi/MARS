import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from library.config import CFG
from library.utils import seed_everything, get_score
from library.data import process_data, get_cpc_texts


def get_stacking_features(train_df, test_df, oof_probs, test_probs):
    """
    Constructs the feature matrix for the stacking model.
    Combines Level 1 probabilities, structural features, and context.
    """
    # Ensure structural features exist in the dataframes
    # If the caller passed raw dataframes, we compute the features here.
    if "feat_lev" not in train_df.columns:
        cpc_map = get_cpc_texts()
        train_df = process_data(train_df, cpc_map)
    if "feat_lev" not in test_df.columns:
        cpc_map = get_cpc_texts()
        test_df = process_data(test_df, cpc_map)

    # Context Encoding
    # Concatenate to ensure consistent LabelEncoding across train and test
    # We use the raw 'context' code (e.g., 'A47') as a categorical feature
    all_contexts = pd.concat([train_df["context"], test_df["context"]], axis=0)
    le = LabelEncoder()
    le.fit(all_contexts.astype(str))

    train_ctx = le.transform(train_df["context"].astype(str))
    test_ctx = le.transform(test_df["context"].astype(str))

    # Extract Structural Features
    # Columns: feat_lev (Normalized Levenshtein), feat_jac (Jaccard), feat_len (Length Ratio)
    feat_cols = ["feat_lev", "feat_jac", "feat_len"]

    X_train_struct = train_df[feat_cols].values.astype(np.float32)
    X_test_struct = test_df[feat_cols].values.astype(np.float32)

    # Combine Features:
    # 1. Level 1 Probabilities (N, 5)
    # 2. Structural Features (N, 3)
    # 3. Context ID (N, 1)

    X_train = np.hstack([oof_probs, X_train_struct, train_ctx.reshape(-1, 1)])
    X_test = np.hstack([test_probs, X_test_struct, test_ctx.reshape(-1, 1)])

    y_train = train_df["score"].values.astype(np.float32)

    return X_train, y_train, X_test


def train_stacking_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    load_cached_data: bool = True,
):
    """
    Trains a LightGBM meta-learner on OOF predictions and generates the submission.

    Args:
        train_df: DataFrame containing training metadata and targets.
        test_df: DataFrame containing test metadata.
        oof_probs: (N_train, 5) array of Level 1 probability predictions.
        test_probs: (N_test, 5) array of Level 1 probability predictions.
        load_cached_data: Whether to load pre-computed feature matrices from disk.
    """
    seed_everything(CFG.seed)

    # Caching Logic
    cache_dir = CFG.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    x_train_path = os.path.join(cache_dir, "stacking_X_train.npy")
    y_train_path = os.path.join(cache_dir, "stacking_y_train.npy")
    x_test_path = os.path.join(cache_dir, "stacking_X_test.npy")

    # Check if we can load from cache
    # We also check if the shapes match the current inputs to avoid stale cache issues
    cache_valid = False
    if (
        load_cached_data
        and os.path.exists(x_train_path)
        and os.path.exists(x_test_path)
    ):
        try:
            X_train = np.load(x_train_path)
            if X_train.shape[0] == len(train_df):
                cache_valid = True
        except:
            cache_valid = False

    if cache_valid:
        print("Loading cached stacking features...")
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
    else:
        print("Computing stacking features...")
        X_train, y_train, X_test = get_stacking_features(
            train_df, test_df, oof_probs, test_probs
        )
        np.save(x_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path, X_test)

    # LightGBM Configuration
    # Feature indices: 0-4 (probs), 5-7 (struct), 8 (context)
    # We specify the context column index as categorical
    cat_features = [8]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": CFG.seed,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_jobs": CFG.num_workers,
        "min_child_samples": 20,
    }

    # Cross-Validation Strategy
    # We use StratifiedKFold based on the discrete score values (0.0, 0.25, ...)
    # This ensures the distribution of scores is consistent across folds.
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    # Create stratification labels (0, 1, 2, 3, 4) from scores
    y_stratify = (y_train * 4).round().astype(int)

    test_preds_accum = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    print(f"Starting LightGBM Training ({CFG.n_fold} folds)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_stratify)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        val_data = lgb.Dataset(
            X_val, label=y_val, categorical_feature=cat_features, reference=train_data
        )

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress verbose logging
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Validation Prediction
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # Test Prediction
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds_accum += test_pred

        # Fold Metrics
        score = get_score(y_val, val_pred)
        rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        print(f"Fold {fold} | Pearson: {score:.8f} | RMSE: {rmse:.8f}")

    # Average Test Predictions over folds
    avg_test_preds = test_preds_accum / CFG.n_fold

    # Clip predictions to valid range [0, 1]
    avg_test_preds = np.clip(avg_test_preds, 0, 1)
    oof_preds = np.clip(oof_preds, 0, 1)

    # Overall CV Score
    overall_score = get_score(y_train, oof_preds)
    print(f"Overall CV Pearson: {overall_score:.8f}")

    # Generate Submission File
    submission = pd.DataFrame({"id": test_df["id"], "score": avg_test_preds})

    submission.to_csv(CFG.submission_path, index=False)
    print(f"Submission saved to {CFG.submission_path}")

    return avg_test_preds
