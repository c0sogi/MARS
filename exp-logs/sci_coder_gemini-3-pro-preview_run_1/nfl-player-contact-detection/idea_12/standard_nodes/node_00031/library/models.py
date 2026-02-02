import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from library.config import (
    LGBM_SCOUT_PARAMS,
    LGBM_EXPERT_PARAMS,
    XGB_EXPERT_PARAMS,
    TRAIN_CONFIG,
    WORKING_DIR,
    SEED,
    HARD_NEGATIVE_INDICES_PATH,
    SCOUT_PREDS_PATH,
)

# Ensure reproducibility
np.random.seed(SEED)


class LGBMModel:
    def __init__(self, params, name="lgbm"):
        self.params = params
        self.name = name
        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_rounds,
        early_stopping_rounds,
        verbose_eval,
    ):
        print(f"Training {self.name}...")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class XGBModel:
    def __init__(self, params, name="xgb"):
        self.params = params
        self.name = name
        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        num_rounds,
        early_stopping_rounds,
        verbose_eval,
    ):
        print(f"Training {self.name}...")

        # Create DMatrix
        # Note: XGBoost with device='cuda' handles pandas dataframes efficiently
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class EnsemblePredictor:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        preds = []
        for model in self.models:
            p = model.predict_proba(X)
            preds.append(p)

        # Simple average ensemble
        return np.mean(preds, axis=0)


def evaluate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    print(
        f"Metrics (Thresh={threshold}): MCC={mcc}, AP={ap}, AUC={auc}, Precision={prec}, Recall={rec}"
    )
    return mcc


def get_hard_negative_indices(X_train, y_train, X_val, y_val, load_cached_data=True):
    """
    Executes Phase 1 & 2: Scout Training and Hard Negative Mining.
    Returns indices of hard negatives in X_train.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(HARD_NEGATIVE_INDICES_PATH):
        print(f"Loading hard negative indices from cache: {HARD_NEGATIVE_INDICES_PATH}")
        return np.load(HARD_NEGATIVE_INDICES_PATH)

    print("Starting Scout Phase for Hard Negative Mining...")

    # 2. Create Balanced Scout Dataset
    # Filter Positives
    pos_mask = y_train == 1
    neg_mask = y_train == 0

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    n_pos = len(pos_indices)
    # Use negative_sampling_ratio for scout balance (usually 1:1 or similar)
    ratio = TRAIN_CONFIG["negative_sampling_ratio"]
    n_neg = int(n_pos * ratio)

    # Sample negatives
    rng = np.random.RandomState(SEED)
    sampled_neg_indices = rng.choice(
        neg_indices, size=min(n_neg, len(neg_indices)), replace=False
    )

    scout_indices = np.concatenate([pos_indices, sampled_neg_indices])
    rng.shuffle(scout_indices)

    X_scout = X_train.iloc[scout_indices]
    y_scout = y_train[scout_indices]

    print(
        f"Scout Dataset: {len(X_scout)} rows (Pos: {n_pos}, Neg: {len(sampled_neg_indices)})"
    )

    # 3. Train Scout Model
    scout_model = LGBMModel(LGBM_SCOUT_PARAMS, name="scout_lgbm")
    scout_model.fit(
        X_scout,
        y_scout,
        X_val,
        y_val,
        num_rounds=TRAIN_CONFIG["scout_rounds"],
        early_stopping_rounds=TRAIN_CONFIG["early_stopping_rounds"],
        verbose_eval=TRAIN_CONFIG["verbose_eval"],
    )

    # 4. Mine Hard Negatives
    print("Mining hard negatives on full training set...")
    # Predict on ALL training data
    # To save memory, we can batch predict if needed, but 3M rows is manageable for inference
    preds = scout_model.predict_proba(X_train)

    # Identify Hard Negatives: Actual is 0, Predicted Probability > Threshold
    threshold = TRAIN_CONFIG["hard_negative_threshold"]
    hard_neg_mask = (y_train == 0) & (preds > threshold)
    hard_neg_indices = np.where(hard_neg_mask)[0]

    print(f"Found {len(hard_neg_indices)} hard negatives (Prob > {threshold}).")

    # 5. Cache Results
    print(f"Saving hard negative indices to {HARD_NEGATIVE_INDICES_PATH}")
    np.save(HARD_NEGATIVE_INDICES_PATH, hard_neg_indices)

    # Optionally save scout predictions for analysis
    # np.save(SCOUT_PREDS_PATH, preds)

    return hard_neg_indices


def train_with_mining_curriculum(X_train, y_train, X_val, y_val, load_cached_data=True):
    """
    Orchestrates the PG-KME training strategy.
    """
    # ---------------------------------------------------------
    # Phase 1 & 2: Mining
    # ---------------------------------------------------------
    hard_neg_indices = get_hard_negative_indices(
        X_train, y_train, X_val, y_val, load_cached_data=load_cached_data
    )

    # ---------------------------------------------------------
    # Phase 3: Expert Training
    # ---------------------------------------------------------
    print("\nConstructing Expert Dataset...")

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Expert Set = All Positives + All Hard Negatives + Buffer of Random Negatives
    # Buffer size: Let's use 1x Positives count as random buffer to maintain diversity
    n_pos = len(pos_indices)
    rng = np.random.RandomState(SEED)

    # Exclude hard negatives from the random pool to avoid duplication
    # Set difference is expensive on large arrays, so we just sample and unique later or accept minor overlap
    # Ideally:
    # random_neg_pool = np.setdiff1d(neg_indices, hard_neg_indices) # Can be slow
    # Fast approximation: Sample from all negs, then union with hard negs.

    n_buffer = int(n_pos * 1.0)  # 1:1 ratio for buffer
    random_neg_indices = rng.choice(
        neg_indices, size=min(n_buffer, len(neg_indices)), replace=False
    )

    expert_indices = np.concatenate([pos_indices, hard_neg_indices, random_neg_indices])
    expert_indices = np.unique(expert_indices)  # Remove duplicates if any overlap
    rng.shuffle(expert_indices)

    X_expert = X_train.iloc[expert_indices]
    y_expert = y_train[expert_indices]

    print(f"Expert Dataset: {len(X_expert)} rows")
    print(f"  Positives: {len(pos_indices)}")
    print(f"  Hard Negatives: {len(hard_neg_indices)}")
    print(f"  Random Buffer: {len(random_neg_indices)}")

    # Train Expert LGBM
    lgbm_expert = LGBMModel(LGBM_EXPERT_PARAMS, name="expert_lgbm")
    lgbm_expert.fit(
        X_expert,
        y_expert,
        X_val,
        y_val,
        num_rounds=TRAIN_CONFIG["expert_rounds"],
        early_stopping_rounds=TRAIN_CONFIG["early_stopping_rounds"],
        verbose_eval=TRAIN_CONFIG["verbose_eval"],
    )

    # Train Expert XGB
    xgb_expert = XGBModel(XGB_EXPERT_PARAMS, name="expert_xgb")
    xgb_expert.fit(
        X_expert,
        y_expert,
        X_val,
        y_val,
        num_rounds=TRAIN_CONFIG["expert_rounds"],
        early_stopping_rounds=TRAIN_CONFIG["early_stopping_rounds"],
        verbose_eval=TRAIN_CONFIG["verbose_eval"],
    )

    # ---------------------------------------------------------
    # Evaluation & Ensemble
    # ---------------------------------------------------------
    print("\nEvaluating Ensemble on Validation Set...")
    ensemble = EnsemblePredictor([lgbm_expert, xgb_expert])

    val_preds = ensemble.predict_proba(X_val)

    # Find best threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_mcc = -1
    best_thresh = 0.5

    for t in thresholds:
        mcc = matthews_corrcoef(y_val, (val_preds >= t).astype(int))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    print(f"Best Validation MCC: {best_mcc} at Threshold: {best_thresh}")
    evaluate_metrics(y_val, val_preds, threshold=best_thresh)

    # Save best threshold
    np.save(os.path.join(WORKING_DIR, "best_threshold.npy"), np.array([best_thresh]))

    return ensemble
