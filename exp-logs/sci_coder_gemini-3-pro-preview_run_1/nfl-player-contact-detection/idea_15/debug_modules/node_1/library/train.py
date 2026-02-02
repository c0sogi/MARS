import os
import numpy as np
import pandas as pd
from library import config, utils, data_loader, models


def train_scout(df_train, df_val):
    """
    Trains the Scout LightGBM model on a balanced subset of the training data.

    Args:
        df_train (pd.DataFrame): Full gated training dataset.
        df_val (pd.DataFrame): Validation dataset.

    Returns:
        models.LGBMHandler: Trained Scout model.
    """
    print("--- Starting Scout Training Phase ---")

    # Initialize DatasetBuilder
    builder = data_loader.DatasetBuilder()

    # Build balanced dataset for Scout (1:1 ratio)
    # This ensures the scout learns basic contact physics without being overwhelmed by negatives
    df_scout_train = builder.build_scout_dataset(df_train, negative_ratio=1.0)

    print(f"Scout Training Data Shape: {df_scout_train.shape}")

    # Initialize Scout model handler
    # We use a distinct filename to separate it from the expert model
    scout_model = models.LGBMHandler(model_name="scout_lgbm.joblib")

    # Train the model
    # The handler manages early stopping and metric logging internally
    scout_model.fit(df_scout_train, df_val)

    return scout_model


def mine_hard_negatives(scout_model, df_train, load_cached_indices=True):
    """
    Uses the Scout model to identify hard negatives in the full training set.

    Args:
        scout_model (models.LGBMHandler): Trained Scout model.
        df_train (pd.DataFrame): Full gated training dataset.
        load_cached_indices (bool): Whether to load indices from disk if available.

    Returns:
        np.ndarray: Array of indices corresponding to hard negatives in df_train.
    """
    print("--- Starting Hard Negative Mining Phase ---")

    cache_path = os.path.join(
        config.WORKING_DIR, "data_cache", "hard_negative_indices.npy"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Check cache
    if load_cached_indices and os.path.exists(cache_path):
        print(f"Loading hard negative indices from {cache_path}")
        return np.load(cache_path)

    print("Predicting on full training set with Scout model...")
    # Generate probabilities for the entire training set
    probs = scout_model.predict_proba(df_train)

    # Identify Hard Negatives
    # Definition: Ground Truth is 0 (No Contact) BUT Predicted Probability > Threshold
    # This captures "Close Calls" or "Confusing Non-Contacts"
    is_negative = df_train["contact"] == 0
    is_high_prob = probs > config.HARD_NEGATIVE_THRESHOLD

    hard_negative_mask = is_negative & is_high_prob
    hard_negative_indices = df_train.index[hard_negative_mask].values

    print(f"Mining complete. Found {len(hard_negative_indices)} hard negatives.")

    # Save to cache
    np.save(cache_path, hard_negative_indices)
    print(f"Hard negative indices saved to {cache_path}")

    return hard_negative_indices


def train_expert(df_train, df_val, hard_negative_indices):
    """
    Trains the Expert Ensemble (LGBM + XGB) on the hard-mined dataset.

    Args:
        df_train (pd.DataFrame): Full gated training dataset.
        df_val (pd.DataFrame): Validation dataset.
        hard_negative_indices (np.ndarray): Indices of hard negatives to include.

    Returns:
        tuple: (Trained LGBMHandler, Trained XGBHandler)
    """
    print("--- Starting Expert Training Phase ---")

    builder = data_loader.DatasetBuilder()

    # Build Expert Dataset
    # Composition: All Positives + All Hard Negatives + Random Buffer of Negatives
    # This curriculum forces the model to focus on the decision boundary
    df_expert_train = builder.build_expert_dataset(
        df_train, hard_negative_indices, random_negative_ratio=0.5
    )

    print(f"Expert Training Data Shape: {df_expert_train.shape}")

    # Train Expert LightGBM (Leaf-wise growth)
    print("Training Expert LightGBM...")
    expert_lgbm = models.LGBMHandler(model_name="expert_lgbm.joblib")
    expert_lgbm.fit(df_expert_train, df_val)

    # Train Expert XGBoost (Level-wise growth)
    print("Training Expert XGBoost...")
    expert_xgb = models.XGBHandler(model_name="expert_xgb.joblib")
    expert_xgb.fit(df_expert_train, df_val)

    return expert_lgbm, expert_xgb


def optimize_threshold(lgbm_model, xgb_model, df_val):
    """
    Finds the optimal decision threshold maximizing MCC on the validation set.

    Args:
        lgbm_model: Trained LightGBM model.
        xgb_model: Trained XGBoost model.
        df_val (pd.DataFrame): Validation dataset.

    Returns:
        float: The optimal threshold value.
    """
    print("--- Optimizing Decision Threshold ---")

    # Create ensemble predictor
    ensemble = models.EnsemblePredictor(lgbm_model, xgb_model)

    # Get ensemble probabilities
    probs = ensemble.predict_proba(df_val)
    y_true = df_val["contact"].values

    best_mcc = -1.0
    best_thresh = 0.5

    # Grid search for threshold
    # We scan a wide range to ensure we catch the optimal point
    thresholds = np.arange(0.1, 0.91, 0.01)

    for thresh in thresholds:
        mcc = utils.calc_mcc(y_true, probs, threshold=thresh)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    print(f"Best MCC: {best_mcc}")
    print(f"Best Threshold: {best_thresh}")

    # Save the best threshold for inference
    thresh_path = os.path.join(config.WORKING_DIR, "models", "best_threshold.npy")
    os.makedirs(os.path.dirname(thresh_path), exist_ok=True)
    np.save(thresh_path, np.array([best_thresh]))

    return best_thresh


def run_training_pipeline(load_cached_features=True, load_cached_mining=True):
    """
    Orchestrates the full training pipeline:
    1. Load Data (with Gating)
    2. Train Scout Model
    3. Mine Hard Negatives
    4. Train Expert Ensemble
    5. Optimize Threshold

    Args:
        load_cached_features (bool): Whether to use cached feature files.
        load_cached_mining (bool): Whether to use cached hard negative indices.

    Returns:
        tuple: (expert_lgbm, expert_xgb, best_threshold)
    """
    # Ensure reproducibility
    utils.seed_everything()
    utils.setup_logging()

    print("=== Initializing Training Pipeline ===")

    # 1. Load Data
    # DatasetBuilder calls features.generate_features which handles caching and gating
    print("Loading Training Data...")
    df_train = data_loader.DatasetBuilder().load_data(
        "train", load_cached=load_cached_features
    )

    print("Loading Validation Data...")
    df_val = data_loader.DatasetBuilder().load_data(
        "val", load_cached=load_cached_features
    )

    # 2. Scout Phase
    scout_model = train_scout(df_train, df_val)

    # 3. Mining Phase
    hard_indices = mine_hard_negatives(
        scout_model, df_train, load_cached_indices=load_cached_mining
    )

    # 4. Expert Phase
    expert_lgbm, expert_xgb = train_expert(df_train, df_val, hard_indices)

    # 5. Threshold Optimization
    best_thresh = optimize_threshold(expert_lgbm, expert_xgb, df_val)

    print("=== Training Pipeline Completed Successfully ===")

    return expert_lgbm, expert_xgb, best_thresh
