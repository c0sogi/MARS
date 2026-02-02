import os
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import matthews_corrcoef
from library.config import PathConfig, TrainConfig, SEED
from library.utils import (
    setup_logging,
    save_numpy,
    load_numpy,
    save_artifact,
    load_artifact,
)
from library.data import get_data_split
from library.models import TriEnsemble

# Initialize logging
setup_logging()

# Columns to exclude from features
META_COLS = [
    "contact_id",
    "game_play",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
    "datetime",
]


def get_feature_cols(df):
    """Returns the list of feature columns by excluding metadata."""
    return [c for c in df.columns if c not in META_COLS]


def train_scouts(df_train, feature_cols):
    """
    Phase 1: Train Scouts on a balanced subset of the data.
    Positives: contact > 0 (due to smoothing)
    Negatives: contact == 0
    """
    logging.info("Phase 1: Preparing Scout Dataset (Balanced)...")

    # Split into Positives (Signal) and Negatives (Background)
    # With label smoothing, anything > 0 contains some signal
    mask_pos = df_train["contact"] > 0
    mask_neg = df_train["contact"] == 0

    df_pos = df_train[mask_pos]
    df_neg = df_train[mask_neg]

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    logging.info(f"Total Positives (Signal > 0): {n_pos}")
    logging.info(f"Total Negatives (Signal == 0): {n_neg}")

    # Downsample Negatives to match Positives
    if n_neg > n_pos:
        df_neg_sampled = df_neg.sample(n=n_pos, random_state=SEED)
    else:
        df_neg_sampled = df_neg

    df_balanced = (
        pd.concat([df_pos, df_neg_sampled])
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )

    X_scout = df_balanced[feature_cols]
    y_scout = df_balanced["contact"]

    logging.info(f"Training Scouts on {len(df_balanced)} samples...")
    scout_ensemble = TriEnsemble()
    scout_ensemble.fit(X_scout, y_scout)

    return scout_ensemble


def mine_hard_negatives(df_train, feature_cols, load_cached=True):
    """
    Phase 2: Use Scouts to find Hard Negatives in the full training set.
    Hard Negative: True Label == 0 BUT Scout Prediction > Threshold.
    """
    cache_path = PathConfig.CACHE_HARD_NEGATIVES

    if load_cached and os.path.exists(cache_path):
        logging.info(f"Loading cached Hard Negative indices from {cache_path}...")
        return load_numpy(cache_path)

    logging.info("Phase 2: Mining Hard Negatives...")

    # Train Scouts first
    scout_ensemble = train_scouts(df_train, feature_cols)

    # Save Scouts
    scout_dir = os.path.join(PathConfig.WORKING_DIR, "models", "scouts")
    scout_ensemble.save(scout_dir)

    # Predict on ALL Negatives (we only mine from true negatives)
    # We process in chunks to avoid OOM if dataset is huge, but here we fit in memory
    mask_neg = df_train["contact"] == 0
    df_neg_full = df_train[mask_neg].reset_index(
        drop=True
    )  # Reset index to align with predictions

    # We need the original indices to reference back to df_train
    # So let's filter indices instead
    neg_indices = df_train.index[mask_neg].to_numpy()

    X_neg = df_train.loc[neg_indices, feature_cols]

    logging.info(f"Scoring {len(X_neg)} negative samples for mining...")
    preds = scout_ensemble.predict(X_neg)

    # Identify Hard Negatives
    hard_mask = preds > TrainConfig.HARD_NEGATIVE_THRESHOLD
    hard_negative_indices = neg_indices[hard_mask]

    logging.info(
        f"Found {len(hard_negative_indices)} Hard Negatives (Prob > {TrainConfig.HARD_NEGATIVE_THRESHOLD})"
    )

    # Save to cache
    save_numpy(hard_negative_indices, cache_path)

    return hard_negative_indices


def construct_expert_dataset(df_train, hard_neg_indices):
    """
    Constructs the dataset for the Expert model.
    Composition:
    1. All Positives (contact > 0)
    2. All Mined Hard Negatives
    3. Random Anchors (Easy Negatives) based on ANCHOR_RATIO
    """
    logging.info("Phase 3: Constructing Expert Dataset...")

    # 1. All Positives
    pos_indices = df_train.index[df_train["contact"] > 0].to_numpy()

    # 2. Hard Negatives
    # Ensure they are unique and valid
    hard_neg_indices = np.intersect1d(hard_neg_indices, df_train.index)

    # 3. Anchors
    # Candidates are Negatives that are NOT Hard Negatives
    # We can use set difference
    all_neg_indices = df_train.index[df_train["contact"] == 0].to_numpy()
    easy_neg_indices = np.setdiff1d(all_neg_indices, hard_neg_indices)

    n_hard = len(hard_neg_indices)
    n_anchors = int(n_hard * TrainConfig.ANCHOR_RATIO)

    # Sample Anchors
    rng = np.random.default_rng(SEED)
    if len(easy_neg_indices) > n_anchors:
        anchor_indices = rng.choice(easy_neg_indices, size=n_anchors, replace=False)
    else:
        anchor_indices = easy_neg_indices

    logging.info(
        f"Components: Positives={len(pos_indices)}, HardNegs={len(hard_neg_indices)}, Anchors={len(anchor_indices)}"
    )

    # Combine
    final_indices = np.concatenate([pos_indices, hard_neg_indices, anchor_indices])
    # Shuffle
    rng.shuffle(final_indices)

    return df_train.loc[final_indices].reset_index(drop=True)


def optimize_threshold(y_true, y_pred_prob):
    """
    Finds the binary classification threshold that maximizes MCC.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_mcc = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred_bin = (y_pred_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_bin)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    return best_thresh, best_mcc


def run_training_pipeline(load_cached=True):
    """
    Main orchestrator for the Tri-Scout Anchored Mining Curriculum.
    """
    # 1. Load Data
    df_train = get_data_split("train", load_cached=load_cached)
    df_val = get_data_split("val", load_cached=load_cached)

    feature_cols = get_feature_cols(df_train)
    logging.info(f"Feature columns: {len(feature_cols)}")

    # 2. Mine Hard Negatives
    hard_neg_indices = mine_hard_negatives(
        df_train, feature_cols, load_cached=load_cached
    )

    # 3. Construct Expert Dataset
    df_expert = construct_expert_dataset(df_train, hard_neg_indices)

    X_train = df_expert[feature_cols]
    y_train = df_expert["contact"]

    X_val = df_val[feature_cols]
    # Validation targets are binary in the provided dataset, but let's ensure type
    y_val = df_val["contact"].astype(int)

    # 4. Train Expert Ensemble
    logging.info("Training Expert Ensemble...")
    expert_ensemble = TriEnsemble()

    # We pass validation data for early stopping
    expert_ensemble.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

    # 5. Evaluate and Optimize
    logging.info("Evaluating on Validation Set...")
    val_probs = expert_ensemble.predict(X_val)

    best_thresh, best_mcc = optimize_threshold(y_val, val_probs)

    logging.info(
        f"Validation Results: Best Threshold={best_thresh:.4f}, Best MCC={best_mcc}"
    )

    # 6. Save Artifacts
    expert_dir = os.path.join(PathConfig.WORKING_DIR, "models", "experts")
    expert_ensemble.save(expert_dir)

    thresh_path = os.path.join(PathConfig.WORKING_DIR, "best_threshold.npy")
    save_numpy(np.array([best_thresh]), thresh_path)

    return expert_ensemble, best_thresh
