import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from library.config import Config
from library.dataset import load_and_process_data


def prepare_stacking_data(load_cached_data=True):
    """
    Prepares the feature matrices for the Stage 2 Meta-Learner.
    Merges Stage 1 predictions with structural features and one-hot encoded contexts.
    Implements caching using .npy files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X_train = os.path.join(cache_dir, "meta_X_train.npy")
    path_y_train = os.path.join(cache_dir, "meta_y_train.npy")
    path_X_test = os.path.join(cache_dir, "meta_X_test.npy")
    path_test_ids = os.path.join(cache_dir, "meta_test_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(path_X_train)
            and os.path.exists(path_y_train)
            and os.path.exists(path_X_test)
            and os.path.exists(path_test_ids)
        ):
            try:
                X_train = np.load(path_X_train)
                y_train = np.load(path_y_train)
                X_test = np.load(path_X_test)
                test_ids = np.load(path_test_ids, allow_pickle=True)
                # print("Loaded meta-features from cache.")
                return X_train, y_train, X_test, test_ids
            except Exception:
                # print("Cache loading failed. Recomputing...")
                pass

    # 2. Compute from scratch
    # print("Computing meta-features...")

    # Load Stage 1 Predictions
    oof_path = os.path.join(Config.WORKING_DIR, "stage1_oof.csv")
    test_pred_path = os.path.join(Config.WORKING_DIR, "stage1_test.csv")

    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"Stage 1 OOF file not found at {oof_path}. Run Stage 1 first."
        )
    if not os.path.exists(test_pred_path):
        raise FileNotFoundError(
            f"Stage 1 Test file not found at {test_pred_path}. Run Stage 1 first."
        )

    df_oof = pd.read_csv(oof_path)
    df_test_preds = pd.read_csv(test_pred_path)

    # Load Metadata (includes structural features)
    # We load both train and val and combine them to match the OOF set
    df_train_meta = load_and_process_data("train", load_cached_data=load_cached_data)
    df_val_meta = load_and_process_data("val", load_cached_data=load_cached_data)
    df_full_train = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    df_test_meta = load_and_process_data("test", load_cached_data=load_cached_data)

    # Merge Stage 1 Predictions with Metadata
    # Rename 'pred' to 'stage1_pred' to avoid confusion
    df_oof = df_oof.rename(columns={"pred": "stage1_pred"})
    df_test_preds = df_test_preds.rename(columns={"pred": "stage1_pred"})

    # Merge on ID
    # Note: OOF dataframe has 'id', 'score', 'fold', 'stage1_pred'
    # Metadata has 'id', 'anchor', 'target', 'context', 'score', structural features...
    # We merge OOF into Metadata to keep Metadata's rich features
    train_merged = df_full_train.merge(
        df_oof[["id", "stage1_pred"]], on="id", how="inner"
    )
    test_merged = df_test_meta.merge(
        df_test_preds[["id", "stage1_pred"]], on="id", how="inner"
    )

    # Define Feature Columns
    numeric_features = [
        "stage1_pred",
        "levenshtein_dist",
        "levenshtein_norm",
        "jaccard_sim",
        "len_diff",
        "len_ratio",
        "word_len_diff",
    ]

    # Handle Missing Values (Safety check, though unlikely with correct upstream processing)
    train_merged[numeric_features] = train_merged[numeric_features].fillna(0.0)
    test_merged[numeric_features] = test_merged[numeric_features].fillna(0.0)

    # One-Hot Encoding for Context
    # We concat train and test to ensure consistent dummy variables
    # Add a marker to split later
    train_merged["_is_train"] = 1
    test_merged["_is_train"] = 0

    combined = pd.concat([train_merged, test_merged], ignore_index=True)

    # Generate dummies for 'context'
    context_dummies = pd.get_dummies(combined["context"], prefix="ctx", dummy_na=False)

    # Concatenate numeric features with dummies
    X_combined = pd.concat([combined[numeric_features], context_dummies], axis=1)

    # Split back into Train and Test
    X_train_df = X_combined[combined["_is_train"] == 1]
    X_test_df = X_combined[combined["_is_train"] == 0]

    # Extract Targets and IDs
    y_train = train_merged["score"].values.astype(np.float32)
    test_ids = test_merged["id"].values

    # Convert to Numpy
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    # 3. Save to cache
    try:
        np.save(path_X_train, X_train)
        np.save(path_y_train, y_train)
        np.save(path_X_test, X_test)
        np.save(path_test_ids, test_ids)
        # print("Saved meta-features to cache.")
    except Exception:
        pass

    return X_train, y_train, X_test, test_ids


def train_and_predict_stacker(load_cached_data=True):
    """
    Trains a Ridge Regression meta-learner on the OOF predictions and structural features,
    then generates the final submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature matrices.
    """
    print("Starting Stage 2 (Stacking)...")

    # 1. Prepare Data
    X_train, y_train, X_test, test_ids = prepare_stacking_data(
        load_cached_data=load_cached_data
    )

    print(f"Meta-Model Training Data Shape: {X_train.shape}")
    print(f"Meta-Model Test Data Shape: {X_test.shape}")

    # 2. Train Model
    # Ridge Regression is robust for stacking
    model = Ridge(alpha=1.0, random_state=Config.SEED)
    model.fit(X_train, y_train)

    # Evaluate on Training Data (Proxy for performance)
    train_preds = model.predict(X_train)

    # Compute Score manually to avoid circular imports if possible,
    # but we can use the library function if needed.
    # Here we implement simple Pearson for reporting.
    if len(y_train) > 1:
        corr_matrix = np.corrcoef(y_train, train_preds)
        train_score = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        train_score = 0.0

    print(f"Stage 2 Train Pearson Score: {train_score:.6f}")

    # 3. Predict on Test
    final_preds = model.predict(X_test)

    # Clip predictions to valid range [0, 1]
    final_preds = np.clip(final_preds, 0.0, 1.0)

    # 4. Save Submission
    submission_df = pd.DataFrame({"id": test_ids, "score": final_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Saved Final Submission to {Config.SUBMISSION_FILE}")
    print("Stage 2 Complete.")
