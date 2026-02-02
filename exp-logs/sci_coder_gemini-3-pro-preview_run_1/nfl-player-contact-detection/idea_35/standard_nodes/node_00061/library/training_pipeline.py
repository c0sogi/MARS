import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import setup_logger, seed_everything, ensure_dir
from library.feature_engineering import generate_features
from library.model_factory import get_model

# Initialize Logger
logger = setup_logger("training_pipeline")


def get_feature_cols(df):
    """
    Identifies feature columns by excluding metadata and target columns.
    """
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "step_lookup",
    ]
    return [c for c in df.columns if c not in meta_cols]


def train_scouts(X, y):
    """
    Trains the Tri-Scout ensemble (LGBM, XGB, CatBoost) on a balanced subset.
    Used to identify hard negatives in the full dataset.
    """
    logger.info("--- Phase 1: Tri-Scout Training ---")

    # 1. Create Balanced Scout Dataset
    # We use a simple 1:1 undersampling of negatives for the scouts to get a coarse decision boundary
    pos_mask = y == 1
    neg_mask = y == 0

    X_pos, y_pos = X[pos_mask], y[pos_mask]
    X_neg, y_neg = X[neg_mask], y[neg_mask]

    # Sample negatives to match positive count
    n_pos = len(y_pos)
    if len(y_neg) > n_pos:
        indices = np.random.choice(len(X_neg), size=n_pos, replace=False)
        X_neg_sample = X_neg.iloc[indices]
        y_neg_sample = y_neg.iloc[indices]
    else:
        X_neg_sample = X_neg
        y_neg_sample = y_neg

    X_scout = pd.concat([X_pos, X_neg_sample])
    y_scout = pd.concat([y_pos, y_neg_sample])

    # Shuffle
    perm = np.random.permutation(len(X_scout))
    X_scout = X_scout.iloc[perm]
    y_scout = y_scout.iloc[perm]

    logger.info(f"Scout Training Data: {len(X_scout)} rows (Balanced)")

    # 2. Train Scouts
    scouts = []
    model_types = ["lgbm", "xgb", "catboost"]

    for m_name in model_types:
        try:
            logger.info(f"Training Scout: {m_name.upper()}")
            model = get_model(m_name)
            # Scouts don't use validation sets; they are just for mining
            model.fit(X_scout, y_scout)

            # Save scout for caching/debugging
            scout_path = os.path.join(Config.MODEL_DIR, f"scout_{m_name}.joblib")
            model.save(scout_path)

            scouts.append(model)
        except Exception as e:
            logger.warning(f"Failed to train scout {m_name}: {e}")

    return scouts


def mine_hard_negatives(scouts, X, y, load_cached_data=True):
    """
    Uses trained scouts to predict on the full dataset (negatives only).
    Identifies 'Hard Negatives' where any scout predicts prob > Threshold.
    Handles caching of indices.
    """
    logger.info("--- Phase 2: Diversity Mining (Hard Negatives) ---")

    # Check Cache
    if load_cached_data and os.path.exists(Config.CACHE_HARD_NEGATIVES):
        logger.info(
            f"Loading cached hard negative indices from {Config.CACHE_HARD_NEGATIVES}..."
        )
        return np.load(Config.CACHE_HARD_NEGATIVES)

    if not scouts:
        raise ValueError("No scouts available for mining and no cache found.")

    # We only care about mining from the Negative class
    neg_indices = np.where(y == 0)[0]
    X_neg = X.iloc[neg_indices]

    logger.info(f"Mining from {len(X_neg)} negative samples...")

    # Get max probability across all scouts (Union of difficulty)
    max_probs = np.zeros(len(X_neg))

    for model in scouts:
        probs = model.predict_proba(X_neg)
        max_probs = np.maximum(max_probs, probs)

    # Identify Hard Negatives
    hard_mask = max_probs > Config.HARD_NEGATIVE_THRESHOLD

    # Map back to original dataframe indices
    # neg_indices is an array of indices in the original X corresponding to y=0
    # hard_mask is boolean array corresponding to neg_indices
    hard_neg_indices = neg_indices[hard_mask]

    logger.info(
        f"Mined {len(hard_neg_indices)} Hard Negatives ({len(hard_neg_indices)/len(X_neg):.2%} of negatives)."
    )

    # Save Cache
    ensure_dir(Config.CACHE_HARD_NEGATIVES)
    np.save(Config.CACHE_HARD_NEGATIVES, hard_neg_indices)

    return hard_neg_indices


def train_experts(X, y, X_val, y_val, hard_neg_indices):
    """
    Trains the final Expert Ensemble on the composite dataset:
    Positives + Hard Negatives + Random Anchors (1:1 with Positives).
    """
    logger.info("--- Phase 3: Anchored Expert Training ---")

    # 1. Construct Expert Dataset
    # Positives
    pos_mask = y == 1
    X_pos = X[pos_mask]
    y_pos = y[pos_mask]
    n_pos = len(y_pos)

    # Hard Negatives
    X_hard = X.iloc[hard_neg_indices]
    y_hard = y.iloc[hard_neg_indices]

    # Anchors (Easy Negatives)
    # We want negatives that are NOT hard negatives
    # Create a mask for hard negatives
    full_indices = np.arange(len(X))
    is_hard = np.isin(full_indices, hard_neg_indices)

    # Candidates for anchors: Negatives that are not hard
    anchor_candidate_mask = (y == 0) & (~is_hard)
    X_anchors_pool = X[anchor_candidate_mask]
    y_anchors_pool = y[anchor_candidate_mask]

    # Sample Anchors (1:1 ratio with positives defined in Config)
    n_anchors = int(n_pos * Config.ANCHOR_RATIO)
    if len(X_anchors_pool) > n_anchors:
        anchor_indices = np.random.choice(
            len(X_anchors_pool), size=n_anchors, replace=False
        )
        X_anchors = X_anchors_pool.iloc[anchor_indices]
        y_anchors = y_anchors_pool.iloc[anchor_indices]
    else:
        X_anchors = X_anchors_pool
        y_anchors = y_anchors_pool

    # Combine
    X_expert = pd.concat([X_pos, X_hard, X_anchors])
    y_expert = pd.concat([y_pos, y_hard, y_anchors])

    # Shuffle
    perm = np.random.permutation(len(X_expert))
    X_expert = X_expert.iloc[perm]
    y_expert = y_expert.iloc[perm]

    logger.info(f"Expert Training Data: {len(X_expert)} rows")
    logger.info(
        f"Composition: {len(X_pos)} Pos, {len(X_hard)} Hard Neg, {len(X_anchors)} Anchors"
    )

    # 2. Train Experts
    experts = {}
    model_types = ["lgbm", "xgb", "catboost"]

    for m_name in model_types:
        try:
            logger.info(f"Training Expert: {m_name.upper()}")
            model = get_model(m_name)
            model.fit(X_expert, y_expert, X_val=X_val, y_val=y_val)

            # Save
            expert_path = os.path.join(Config.MODEL_DIR, f"expert_{m_name}.joblib")
            model.save(expert_path)

            experts[m_name] = model
        except Exception as e:
            logger.warning(f"Failed to train expert {m_name}: {e}")

    return experts


def find_best_threshold(y_true, y_prob):
    """
    Finds the threshold that maximizes MCC.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_mcc = -1
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    return best_thresh, best_mcc


def run_training_pipeline(debug_sample=None, load_cached_data=True):
    """
    Main orchestrator for the training pipeline.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    logger.info("Loading Training Data...")
    df_train = generate_features(
        "train", load_cached_data=load_cached_data, debug_sample=debug_sample
    )

    logger.info("Loading Validation Data...")
    df_val = generate_features(
        "val", load_cached_data=load_cached_data, debug_sample=debug_sample
    )

    # 2. Prepare Feature Matrices
    feature_cols = get_feature_cols(df_train)
    logger.info(f"Using {len(feature_cols)} features.")

    X_train = df_train[feature_cols]
    y_train = df_train["contact"]

    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # Clean up memory
    del df_train, df_val
    gc.collect()

    # 3. Scout Phase (Skip if hard negatives are already cached)
    hard_neg_indices = None
    scouts = []

    if load_cached_data and os.path.exists(Config.CACHE_HARD_NEGATIVES):
        logger.info("Cached hard negatives found. Skipping Scout training.")
        hard_neg_indices = np.load(Config.CACHE_HARD_NEGATIVES)
    else:
        # Train Scouts
        scouts = train_scouts(X_train, y_train)

        # Mine Hard Negatives
        hard_neg_indices = mine_hard_negatives(
            scouts, X_train, y_train, load_cached_data=False
        )

        # Free memory
        del scouts
        gc.collect()

    # 4. Expert Phase
    experts = train_experts(X_train, y_train, X_val, y_val, hard_neg_indices)

    # 5. Evaluation Phase
    logger.info("--- Phase 4: Evaluation & Threshold Optimization ---")

    val_probs = np.zeros(len(X_val))
    model_count = 0

    for name, model in experts.items():
        p = model.predict_proba(X_val)
        val_probs += p
        model_count += 1

    if model_count > 0:
        avg_val_probs = val_probs / model_count

        best_thresh, best_mcc = find_best_threshold(y_val, avg_val_probs)

        logger.info(f"Ensemble Validation MCC: {best_mcc}")
        logger.info(f"Optimal Threshold: {best_thresh}")

        # Save threshold
        thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
        np.save(thresh_path, np.array([best_thresh]))
    else:
        logger.error("No expert models were trained successfully.")

    logger.info("Training Pipeline Completed.")
