import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.data import preprocess_data


def prepare_stacking_features(
    train_preds, val_preds, test_preds, load_cached_data=True
):
    """
    Prepares feature matrices for the LightGBM stacker by combining Level 1 predictions,
    structural features, and One-Hot encoded context information.

    Args:
        train_preds (np.array): OOF predictions for the training set (aligned with metadata/train.csv).
        val_preds (np.array): Predictions for the validation set (aligned with metadata/val.csv).
        test_preds (np.array): Predictions for the test set (aligned with metadata/test.csv).
        load_cached_data (bool): Whether to attempt loading pre-computed matrices from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Define cache paths
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "stacking_X_train.npy"),
        "y_train": os.path.join(cache_dir, "stacking_y_train.npy"),
        "X_val": os.path.join(cache_dir, "stacking_X_val.npy"),
        "y_val": os.path.join(cache_dir, "stacking_y_val.npy"),
        "X_test": os.path.join(cache_dir, "stacking_X_test.npy"),
        "test_ids": os.path.join(cache_dir, "stacking_test_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data and all(os.path.exists(p) for p in paths.values()):
        try:
            X_train = np.load(paths["X_train"])
            y_train = np.load(paths["y_train"])
            X_val = np.load(paths["X_val"])
            y_val = np.load(paths["y_val"])
            X_test = np.load(paths["X_test"])
            test_ids = np.load(paths["test_ids"], allow_pickle=True)
            print("Loaded stacking features from cache.")
            return X_train, y_train, X_val, y_val, X_test, test_ids
        except Exception as e:
            print(f"Stacking cache load failed: {e}. Recomputing...")

    # 2. Load DataFrames (includes structural features)
    print("Computing stacking features...")
    df_train = preprocess_data("train", load_cached_data=load_cached_data)
    df_val = preprocess_data("val", load_cached_data=load_cached_data)
    df_test = preprocess_data("test", load_cached_data=load_cached_data)

    # Validate input alignment
    if len(train_preds) != len(df_train):
        raise ValueError(
            f"Shape mismatch: train_preds {len(train_preds)} vs df_train {len(df_train)}"
        )
    if len(val_preds) != len(df_val):
        raise ValueError(
            f"Shape mismatch: val_preds {len(val_preds)} vs df_val {len(df_val)}"
        )
    if len(test_preds) != len(df_test):
        raise ValueError(
            f"Shape mismatch: test_preds {len(test_preds)} vs df_test {len(df_test)}"
        )

    # 3. Feature Engineering

    # Structural features defined in Config
    struct_cols = Config.structural_features

    # Context Features (One-Hot Encoding)
    # We concatenate all contexts to ensure consistent dummy variables across splits
    df_train["_source"] = "train"
    df_val["_source"] = "val"
    df_test["_source"] = "test"

    all_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    # Generate dummies for 'context' (e.g., A47, H04)
    context_dummies = pd.get_dummies(all_df["context"], prefix="ctx", dummy_na=False)

    # Helper to construct the feature matrix X
    def build_X(df_subset, preds, indices):
        # 1. Level 1 Predictions (Reshaped to column vector)
        p_feats = preds.reshape(-1, 1).astype(np.float32)

        # 2. Structural Features (Levenshtein, Jaccard, Length Ratio)
        struct_feats = df_subset[struct_cols].values.astype(np.float32)

        # 3. Context Features (One-Hot)
        ctx_feats = context_dummies.iloc[indices].values.astype(np.float32)

        # Concatenate horizontally
        return np.hstack([p_feats, struct_feats, ctx_feats])

    # Get indices for each split
    train_idx = all_df[all_df["_source"] == "train"].index
    val_idx = all_df[all_df["_source"] == "val"].index
    test_idx = all_df[all_df["_source"] == "test"].index

    # Build Matrices
    X_train = build_X(df_train, np.array(train_preds), train_idx)
    X_val = build_X(df_val, np.array(val_preds), val_idx)
    X_test = build_X(df_test, np.array(test_preds), test_idx)

    # Extract Targets
    y_train = df_train["score"].values.astype(np.float32)
    y_val = df_val["score"].values.astype(np.float32)

    # Extract IDs for submission
    test_ids = df_test["id"].values

    # 4. Save to Cache
    try:
        np.save(paths["X_train"], X_train)
        np.save(paths["y_train"], y_train)
        np.save(paths["X_val"], X_val)
        np.save(paths["y_val"], y_val)
        np.save(paths["X_test"], X_test)
        np.save(paths["test_ids"], test_ids)
        print("Saved stacking features to cache.")
    except Exception as e:
        print(f"Warning: Failed to save stacking cache: {e}")

    return X_train, y_train, X_val, y_val, X_test, test_ids


def train_lgbm_stacker(train_preds, val_preds, test_preds, load_cached_data=True):
    """
    Trains the Level 2 LightGBM Meta-Learner.

    Args:
        train_preds (np.array): Level 1 predictions for Train set.
        val_preds (np.array): Level 1 predictions for Validation set.
        test_preds (np.array): Level 1 predictions for Test set.
        load_cached_data (bool): Whether to use cached feature matrices.

    Returns:
        np.array: Final predictions for the test set.
    """
    # 1. Prepare Data
    X_train, y_train, X_val, y_val, X_test, test_ids = prepare_stacking_features(
        train_preds, val_preds, test_preds, load_cached_data
    )

    print(
        f"Stacking Data Prepared: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # 2. Configure LightGBM
    # Copy params to avoid modifying Config in place
    params = Config.lgb_params.copy()
    # Extract early_stopping_rounds to pass to callback
    es_rounds = params.pop("early_stopping_rounds", 100)

    model = lgb.LGBMRegressor(**params)

    # Define callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds),
        lgb.log_evaluation(period=100),
    ]

    # 3. Train
    print("Starting LightGBM training...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    # 4. Validate
    val_pred = model.predict(X_val)
    # Clip predictions to valid range [0, 1]
    val_pred = np.clip(val_pred, 0, 1)
    val_pearson = np.corrcoef(y_val, val_pred)[0, 1]
    print(f"Stacker Validation Pearson Correlation: {val_pearson:.10f}")

    # 5. Predict on Test
    print("Generating final test predictions...")
    test_pred = model.predict(X_test)
    test_pred = np.clip(test_pred, 0, 1)

    # 6. Save Submission
    submission = pd.DataFrame({"id": test_ids, "score": test_pred})
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved successfully to {Config.submission_path}")

    return test_pred
