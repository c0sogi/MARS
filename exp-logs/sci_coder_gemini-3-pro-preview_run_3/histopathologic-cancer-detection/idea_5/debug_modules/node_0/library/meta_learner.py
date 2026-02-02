import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.modeling import StackingMetaLearner
from library.trainer import get_folds


def train_stacker(model_oof_preds=None, load_cached_data=True):
    """
    Trains the Stacking Meta-Learner using OOF predictions from base models.

    Args:
        model_oof_preds (dict, optional): Dictionary containing OOF predictions.
            Structure: { 'model_name': { fold_idx: np.array_of_preds } }
            Required if cache is not found or load_cached_data is False.
        load_cached_data (bool): Whether to attempt loading processed OOF data from cache.

    Returns:
        StackingMetaLearner: The trained meta-learner instance.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "stacked_oof_data.parquet")
    model_save_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")

    df_data = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading stacked OOF data from {cache_path}")
        try:
            df_data = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Generate data if not loaded
    if df_data is None:
        if model_oof_preds is None:
            raise ValueError("model_oof_preds is required when cache is not available.")

        print("Constructing Stacking Feature Matrix...")

        # Load folds to get ground truth and indices
        # get_folds handles its own caching
        folds_df = get_folds(load_cached_data=True)

        # Initialize DataFrame with targets
        # We sort by original index to ensure alignment, though folds_df should be aligned
        df_data = folds_df[["id", "label", "fold"]].copy()

        # Construct feature columns for each model in Config.MODELS
        for model_name in Config.MODELS:
            if model_name not in model_oof_preds:
                raise ValueError(f"Missing OOF predictions for model: {model_name}")

            # Initialize column with NaNs
            col_name = f"pred_{model_name}"
            df_data[col_name] = np.nan

            # Fill in predictions fold by fold
            fold_preds_map = model_oof_preds[model_name]

            for fold_idx in range(Config.N_FOLDS):
                if fold_idx not in fold_preds_map:
                    raise ValueError(
                        f"Missing predictions for {model_name} fold {fold_idx}"
                    )

                preds = fold_preds_map[fold_idx]

                # Get indices for this fold
                fold_indices = df_data[df_data["fold"] == fold_idx].index

                if len(preds) != len(fold_indices):
                    raise ValueError(
                        f"Shape mismatch for {model_name} fold {fold_idx}. "
                        f"Got {len(preds)}, expected {len(fold_indices)}"
                    )

                # Assign predictions
                df_data.loc[fold_indices, col_name] = preds

        # Verify no missing values
        feature_cols = [f"pred_{m}" for m in Config.MODELS]
        if df_data[feature_cols].isnull().any().any():
            raise ValueError("NaNs detected in constructed OOF matrix.")

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_data.to_parquet(cache_path)
        print(f"Saved stacked OOF data to {cache_path}")

    # 3. Prepare Training Data
    feature_cols = [f"pred_{m}" for m in Config.MODELS]
    X = df_data[feature_cols].values
    y = df_data["label"].values

    print(
        f"Training Stacking Meta-Learner on {len(X)} samples with features: {feature_cols}"
    )

    # 4. Train Meta-Learner
    meta_learner = StackingMetaLearner()
    meta_learner.fit(X, y)

    # 5. Evaluate
    # We evaluate on the same OOF data (standard practice for stacking to see fit)
    # Ideally, one might use nested CV, but standard stacking uses OOF for training the meta-learner.
    preds = meta_learner.predict(X)
    auc = roc_auc_score(y, preds)
    print(f"Stacking Ensemble OOF AUC: {auc}")

    # 6. Save Model
    joblib.dump(meta_learner, model_save_path)
    print(f"Saved Meta-Learner to {model_save_path}")

    return meta_learner


def generate_submission(meta_learner, base_test_preds, test_ids):
    """
    Generates the submission file using the trained meta-learner.

    Args:
        meta_learner (StackingMetaLearner): Trained meta-learner instance.
        base_test_preds (dict): Dictionary of test predictions.
            Structure: { 'model_name': np.array_of_preds }
        test_ids (list): List of test image IDs corresponding to the predictions.
    """
    print("Generating submission...")

    # 1. Construct Test Feature Matrix
    # Must use same order as training: Config.MODELS
    X_test_list = []

    for model_name in Config.MODELS:
        if model_name not in base_test_preds:
            raise ValueError(f"Missing test predictions for model: {model_name}")

        preds = base_test_preds[model_name]
        X_test_list.append(preds)

    # Stack columns: Shape (N_test, N_models)
    X_test = np.column_stack(X_test_list)

    # 2. Predict
    final_probs = meta_learner.predict(X_test)

    # 3. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "label": final_probs})

    # 4. Save
    # Ensure directory exists (Config handles this, but good to be safe)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Head:\n{submission_df.head()}")
