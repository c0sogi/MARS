import os
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    CACHE_HARD_NEGATIVES,
    SCOUT_NEG_RATIO,
    HARD_NEGATIVE_THRESHOLD,
    SCOUT_LGBM_PARAMS,
    EXPERT_LGBM_PARAMS,
    EXPERT_XGB_PARAMS,
    RANDOM_STATE,
    N_JOBS,
)
from library.utils import seed_everything, compute_mcc
from library.feature_engineering import generate_features
from library.model_zoo import LGBMWrapper, XGBWrapper, HeterogeneousEnsemble
from library.data_loader import load_sample_submission

# Set global seed
seed_everything(RANDOM_STATE)


def train_scout_model(df_train, df_val):
    """
    Trains the Scout LightGBM model on a balanced subset of the gated training data.
    """
    print("\n--- Training Scout Model ---")

    # 1. Create Balanced Dataset
    # Filter positives
    pos_mask = df_train["contact"] == 1
    df_pos = df_train[pos_mask]

    # Sample negatives
    neg_mask = df_train["contact"] == 0
    df_neg = df_train[neg_mask]

    n_pos = len(df_pos)
    n_neg_sample = int(n_pos * SCOUT_NEG_RATIO)

    print(
        f"Scout Data: {n_pos} Positives. Sampling {n_neg_sample} Negatives from {len(df_neg)} available."
    )

    if len(df_neg) > n_neg_sample:
        df_neg_sample = df_neg.sample(n=n_neg_sample, random_state=RANDOM_STATE)
    else:
        df_neg_sample = df_neg

    df_scout = (
        pd.concat([df_pos, df_neg_sample])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Prepare Features
    # Exclude metadata columns
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
    ]
    feature_cols = [c for c in df_scout.columns if c not in exclude_cols]

    X_train = df_scout[feature_cols]
    y_train = df_scout["contact"]

    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # Train
    scout_model = LGBMWrapper(SCOUT_LGBM_PARAMS, name="scout_lgbm")
    scout_model.train(X_train, y_train, X_val, y_val)

    # Save
    model_path = os.path.join(WORKING_DIR, "models", "scout_lgbm.joblib")
    scout_model.save(model_path)

    return scout_model, feature_cols


def mine_hard_negatives(scout_model, df_train, feature_cols, load_cached_data=True):
    """
    Runs Scout inference on the full gated training set to find Hard Negatives.
    """
    print("\n--- Mining Hard Negatives ---")

    # Check cache
    if load_cached_data and os.path.exists(CACHE_HARD_NEGATIVES):
        print(f"Loading hard negative indices from {CACHE_HARD_NEGATIVES}...")
        hard_neg_indices = np.load(CACHE_HARD_NEGATIVES)
        print(f"Loaded {len(hard_neg_indices)} hard negative indices.")
        return hard_neg_indices

    print("Running Scout inference on full training set...")

    # Prepare full X
    X_full = df_train[feature_cols]

    # Predict
    preds = scout_model.predict(X_full)

    # Identify Hard Negatives
    # Condition: Ground Truth = 0 AND Prediction > Threshold
    is_neg = (df_train["contact"] == 0).values
    is_hard = preds > HARD_NEGATIVE_THRESHOLD

    hard_neg_mask = is_neg & is_hard
    hard_neg_indices = df_train.index[hard_neg_mask].to_numpy()

    print(
        f"Found {len(hard_neg_indices)} hard negatives out of {np.sum(is_neg)} total negatives."
    )

    # Cache
    print(f"Caching indices to {CACHE_HARD_NEGATIVES}...")
    np.save(CACHE_HARD_NEGATIVES, hard_neg_indices)

    return hard_neg_indices


def train_expert_models(df_train, df_val, hard_neg_indices, feature_cols):
    """
    Trains the Expert Ensemble (LGBM + XGB) on the enriched dataset.
    """
    print("\n--- Training Expert Models ---")

    # 1. Construct Expert Dataset
    # All Positives
    df_pos = df_train[df_train["contact"] == 1]

    # Mined Hard Negatives
    df_hard_neg = df_train.loc[hard_neg_indices]

    # Buffer Random Negatives (Equal to Positives for regularization)
    # We exclude hard negatives from this sampling to avoid duplication
    neg_mask = df_train["contact"] == 0
    df_easy_pool = df_train[neg_mask].drop(index=hard_neg_indices, errors="ignore")

    n_buffer = len(df_pos)
    if len(df_easy_pool) > n_buffer:
        df_buffer = df_easy_pool.sample(n=n_buffer, random_state=RANDOM_STATE)
    else:
        df_buffer = df_easy_pool

    # Combine
    df_expert = (
        pd.concat([df_pos, df_hard_neg, df_buffer])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print(
        f"Expert Dataset: {len(df_pos)} Positives, {len(df_hard_neg)} Hard Negs, {len(df_buffer)} Buffer Negs."
    )
    print(f"Total Expert Samples: {len(df_expert)}")

    X_train = df_expert[feature_cols]
    y_train = df_expert["contact"]

    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # 2. Train Expert LightGBM
    print("Training Expert LightGBM...")
    lgbm_expert = LGBMWrapper(EXPERT_LGBM_PARAMS, name="expert_lgbm")
    lgbm_expert.train(X_train, y_train, X_val, y_val)

    lgbm_path = os.path.join(WORKING_DIR, "models", "expert_lgbm.joblib")
    lgbm_expert.save(lgbm_path)

    # 3. Train Expert XGBoost
    print("Training Expert XGBoost...")
    # Calculate scale_pos_weight
    # ratio = num_neg / num_pos
    n_total = len(y_train)
    n_pos_train = y_train.sum()
    n_neg_train = n_total - n_pos_train
    scale_weight = n_neg_train / n_pos_train if n_pos_train > 0 else 1.0

    xgb_params = EXPERT_XGB_PARAMS.copy()
    xgb_params["scale_pos_weight"] = scale_weight
    print(f"XGB scale_pos_weight set to: {scale_weight:.4f}")

    xgb_expert = XGBWrapper(xgb_params, name="expert_xgb")
    xgb_expert.train(X_train, y_train, X_val, y_val)

    xgb_path = os.path.join(WORKING_DIR, "models", "expert_xgb.joblib")
    xgb_expert.save(xgb_path)

    return HeterogeneousEnsemble([lgbm_expert, xgb_expert])


def evaluate_and_submit(ensemble, df_val, feature_cols, load_cached_data=True):
    """
    Optimizes threshold on validation set and generates submission for test set.
    """
    print("\n--- Evaluation and Submission ---")

    # 1. Optimize Threshold
    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    print("Optimizing Ensemble Threshold on Validation Set...")
    best_threshold = ensemble.optimize_threshold(X_val, y_val)

    # Save threshold
    thresh_path = os.path.join(WORKING_DIR, "models", "best_threshold.npy")
    np.save(thresh_path, np.array([best_threshold]))

    # 2. Inference on Test Set
    # Note: Test set must NOT use gating, we need predictions for all rows in sample_submission
    df_test = generate_features(
        split="test", load_cached_data=load_cached_data, gating=False
    )

    # Ensure columns match
    # Missing columns in test (if any) should be 0, extra columns ignored
    # But generate_features should produce consistent schema
    X_test = df_test[feature_cols]

    print(f"Predicting on Test Set ({len(X_test)} rows)...")
    probs = ensemble.predict(X_test)

    predictions = (probs >= best_threshold).astype(int)

    # 3. Create Submission
    df_sub = load_sample_submission()

    # We need to map predictions back to contact_id.
    # df_test comes from test_metadata which comes from sample_submission.
    # So the order should be preserved if generate_features preserves order.
    # However, generate_features merges tracking data which might reorder if not careful.
    # But our data_loader and feature_engineering keep 'contact_id' and we can join.

    # Let's be safe and merge on contact_id
    df_preds = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact_pred": predictions}
    )

    # Merge with sample submission template to ensure correct order and row count
    # sample_submission has 'contact_id', 'contact' (all 0)
    df_final = df_sub.drop(columns=["contact"]).merge(
        df_preds, on="contact_id", how="left"
    )

    # Fill missing (if any dropped due to tracking issues) with 0
    df_final["contact"] = df_final["contact_pred"].fillna(0).astype(int)
    df_final = df_final[["contact_id", "contact"]]

    print(f"Saving submission to {SUBMISSION_PATH}...")
    df_final.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_pipeline(load_cached_data=True, nrows=None):
    """
    Orchestrates the VFF-MC training pipeline.
    """
    # Create model directory
    os.makedirs(os.path.join(WORKING_DIR, "models"), exist_ok=True)

    # 1. Load Data
    # Train/Val with Gating enabled for efficient training
    df_train = generate_features(
        "train", load_cached_data=load_cached_data, nrows=nrows, gating=True
    )
    df_val = generate_features(
        "val", load_cached_data=load_cached_data, nrows=nrows, gating=True
    )

    # 2. Train Scout
    scout_model, feature_cols = train_scout_model(df_train, df_val)

    # Clean up memory
    gc.collect()

    # 3. Mine Hard Negatives
    hard_neg_indices = mine_hard_negatives(
        scout_model, df_train, feature_cols, load_cached_data=load_cached_data
    )

    # Clean up memory
    del scout_model
    gc.collect()

    # 4. Train Expert Ensemble
    ensemble = train_expert_models(df_train, df_val, hard_neg_indices, feature_cols)

    # Clean up memory
    del df_train
    gc.collect()

    # 5. Evaluate and Submit
    evaluate_and_submit(
        ensemble, df_val, feature_cols, load_cached_data=load_cached_data
    )
