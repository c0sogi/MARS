import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library import config, utils, feature_engineering, models

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_balanced_subset(X, y, ratio=1.0, seed=config.SEED):
    """
    Returns a balanced subset of the data (all positives + random negatives).
    """
    pos_mask = y == 1
    neg_mask = y == 0

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    n_pos = len(pos_indices)
    n_neg_sample = int(n_pos * ratio)

    np.random.seed(seed)
    if len(neg_indices) > n_neg_sample:
        neg_sample_indices = np.random.choice(
            neg_indices, size=n_neg_sample, replace=False
        )
    else:
        neg_sample_indices = neg_indices

    sample_indices = np.concatenate([pos_indices, neg_sample_indices])
    np.random.shuffle(sample_indices)

    return X.iloc[sample_indices], y.iloc[sample_indices]


def _evaluate_mcc(y_true, y_pred_prob, threshold):
    """
    Computes MCC for a given threshold.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    return matthews_corrcoef(y_true, y_pred)


# =============================================================================
# STAGE 1: SCOUT TRAINING & MINING
# =============================================================================


def train_scouts(X_train, y_train):
    """
    Trains two lightweight 'Scout' models (LGBM and XGB) on a balanced subset.
    """
    print("Preparing balanced dataset for Scout training...")
    X_bal, y_bal = _get_balanced_subset(X_train, y_train, ratio=1.0)

    print(f"Training Scout A (LGBM) on {len(X_bal)} samples...")
    scout_a = models.LGBMExpert()
    scout_a.fit(X_bal, y_bal)

    print(f"Training Scout B (XGB) on {len(X_bal)} samples...")
    scout_b = models.XGBExpert()
    scout_b.fit(X_bal, y_bal)

    return scout_a, scout_b


def mine_hard_negatives(X_full, y_full, scout_a, scout_b, load_cached_data=True):
    """
    Runs Scouts on the full dataset to identify Hard Negatives.
    Hard Negative: Negative sample where P(Contact) > 0.05 by either Scout.
    """
    cache_path = config.CACHED_HARD_NEGATIVES

    # 1. Try Load
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached hard negative indices from {cache_path}...")
        return np.load(cache_path)

    # 2. Compute
    print("Mining Hard Negatives (Inference on full training set)...")

    # Filter to negatives only to save inference time?
    # Strategy says "Run Scouts on the Entire Gated Survivor Pool".
    # We only care about Negatives that look like Positives.
    # Positives are always kept.

    # Get indices of all negatives
    neg_indices = np.where(y_full == 0)[0]
    X_neg = X_full.iloc[neg_indices]

    # Predict
    prob_a = scout_a.predict(X_neg)
    prob_b = scout_b.predict(X_neg)

    # Union condition: Scout A > 0.05 OR Scout B > 0.05
    hard_mask = (prob_a > config.HARD_NEGATIVE_THRESHOLD) | (
        prob_b > config.HARD_NEGATIVE_THRESHOLD
    )

    hard_negative_indices = neg_indices[hard_mask]

    print(
        f"Mined {len(hard_negative_indices)} Hard Negatives out of {len(neg_indices)} total negatives."
    )

    # 3. Save
    print(f"Saving hard negative indices to {cache_path}...")
    np.save(cache_path, hard_negative_indices)

    return hard_negative_indices


# =============================================================================
# STAGE 2: EXPERT DATASET CONSTRUCTION
# =============================================================================


def construct_expert_dataset(X_full, y_full, hard_neg_indices):
    """
    Constructs the final training set:
    1. All Positives
    2. All Mined Hard Negatives
    3. Random Easy Negatives (Anchors) at 1:1 ratio with Positives
    """
    print("Constructing Expert Dataset...")

    # 1. Positives
    pos_indices = np.where(y_full == 1)[0]

    # 2. Hard Negatives
    # Ensure hard_neg_indices are valid

    # 3. Anchors
    # Pool for anchors: All negatives excluding hard negatives
    all_neg_indices = np.where(y_full == 0)[0]
    # Set difference to find easy negatives
    # Note: np.setdiff1d assumes unique arrays. Indices are unique.
    easy_neg_pool = np.setdiff1d(all_neg_indices, hard_neg_indices)

    n_pos = len(pos_indices)
    n_anchors = int(n_pos * config.ANCHOR_RATIO)

    np.random.seed(config.SEED)
    if len(easy_neg_pool) > n_anchors:
        anchor_indices = np.random.choice(easy_neg_pool, size=n_anchors, replace=False)
    else:
        anchor_indices = easy_neg_pool

    # Combine
    final_indices = np.concatenate([pos_indices, hard_neg_indices, anchor_indices])
    np.random.shuffle(final_indices)

    print(f"Expert Dataset Stats:")
    print(f"  Positives: {len(pos_indices)}")
    print(f"  Hard Negatives: {len(hard_neg_indices)}")
    print(f"  Anchors: {len(anchor_indices)}")
    print(f"  Total: {len(final_indices)}")

    return X_full.iloc[final_indices], y_full.iloc[final_indices]


# =============================================================================
# STAGE 3: EXPERT TRAINING & OPTIMIZATION
# =============================================================================


def train_experts(X_train, y_train, X_val, y_val):
    """
    Trains the Dual Ensemble Expert models.
    """
    ensemble = models.DualEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)
    return ensemble


def optimize_threshold(model, X_val, y_val):
    """
    Finds the decision threshold that maximizes MCC on the validation set.
    """
    print("Optimizing decision threshold...")
    probs = model.predict(X_val)

    best_threshold = 0.5
    best_score = -1.0

    # Search space
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        score = _evaluate_mcc(y_val, probs, thresh)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"Best Threshold: {best_threshold}")
    print(f"Best Validation MCC: {best_score}")

    # Save threshold
    thresh_path = os.path.join(config.MODEL_DIR, "best_threshold.npy")
    np.save(thresh_path, np.array([best_threshold]))

    return best_threshold


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def run_training_pipeline(debug=False, sample_size=10000, load_cached_data=True):
    """
    Executes the full DBRK-AME training pipeline.
    """
    utils.setup_logging(os.path.join(config.WORKING_DIR, "training.log"))
    utils.seed_everything()

    # 1. Load Features
    print("=== Loading Data ===")
    df_train = feature_engineering.generate_train_features(
        debug=debug, sample_size=sample_size, load_cached_data=load_cached_data
    )
    df_val = feature_engineering.generate_val_features(
        debug=debug, sample_size=sample_size, load_cached_data=load_cached_data
    )

    # Separate Features and Target
    # Drop metadata columns not used for training
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train_full = df_train[feature_cols]
    y_train_full = df_train["contact"]

    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    print(f"Training Features: {X_train_full.shape}")
    print(f"Validation Features: {X_val.shape}")

    # 2. Scout Training & Mining
    # Check if hard negatives are cached; if so, skip scout training unless forced
    hard_neg_path = config.CACHED_HARD_NEGATIVES

    if load_cached_data and os.path.exists(hard_neg_path):
        print("=== Loading Cached Hard Negatives ===")
        hard_neg_indices = np.load(hard_neg_path)
    else:
        print("=== Phase 1: Dual-Scout Training & Mining ===")
        scout_a, scout_b = train_scouts(X_train_full, y_train_full)
        hard_neg_indices = mine_hard_negatives(
            X_train_full, y_train_full, scout_a, scout_b, load_cached_data=False
        )

        # Save Scouts (Optional, but good for analysis)
        scout_a.save(os.path.join(config.MODEL_DIR, "scout_lgbm.joblib"))
        scout_b.save(os.path.join(config.MODEL_DIR, "scout_xgb.joblib"))

    # 3. Construct Expert Dataset
    print("=== Phase 2: Expert Dataset Construction ===")
    X_expert, y_expert = construct_expert_dataset(
        X_train_full, y_train_full, hard_neg_indices
    )

    # 4. Train Experts
    print("=== Phase 3: Anchored Expert Training ===")
    ensemble = train_experts(X_expert, y_expert, X_val, y_val)

    # 5. Optimize Threshold
    print("=== Phase 4: Threshold Optimization ===")
    best_threshold = optimize_threshold(ensemble, X_val, y_val)

    print("Training Pipeline Complete.")
    return ensemble, best_threshold


def inference(model=None, threshold=None, load_cached_data=True):
    """
    Generates submission for the test set.
    """
    utils.setup_logging(os.path.join(config.WORKING_DIR, "inference.log"))

    # 1. Load Model & Threshold if not provided
    if model is None:
        model = models.DualEnsemble()
        model.load()

    if threshold is None:
        thresh_path = os.path.join(config.MODEL_DIR, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = float(np.load(thresh_path)[0])
        else:
            print("Warning: Threshold file not found. Using default 0.5")
            threshold = 0.5

    # 2. Load Test Features
    print("Loading Test Features...")
    df_test = feature_engineering.generate_test_features(
        load_cached_data=load_cached_data
    )

    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]
    feature_cols = [c for c in df_test.columns if c not in meta_cols]

    X_test = df_test[feature_cols]
    contact_ids = df_test["contact_id"]

    # 3. Predict
    print("Generating Predictions...")
    probs = model.predict(X_test)
    preds = (probs >= threshold).astype(int)

    # 4. Create Submission
    submission = pd.DataFrame({"contact_id": contact_ids, "contact": preds})

    # Ensure all contact_ids from sample_submission are present
    # The feature generation might filter out rows (though test gating should be careful).
    # We merge with sample_submission to ensure completeness.
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Merge: Left join on sample submission to keep all required IDs
    # Fill missing predictions with 0 (No Contact)
    final_sub = sample_sub[["contact_id"]].merge(
        submission, on="contact_id", how="left"
    )
    final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

    print(f"Saving submission to {config.SUBMISSION_FILE}...")
    final_sub.to_csv(config.SUBMISSION_FILE, index=False)

    print("Inference Complete.")
