import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.dataset_builder import DatasetBuilder
from library.utils import seed_everything


class DualStreamPredictor:
    """
    Wrapper class to hold trained models and optimal thresholds for both streams.
    """

    def __init__(self, model_a, model_b, threshold_a, threshold_b):
        self.model_a = model_a
        self.model_b = model_b
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b

    def predict_proba(self, X, stream):
        """
        Predicts probabilities for a specific stream.
        """
        if stream == "A":
            return self.model_a.predict_proba(X)[:, 1]
        elif stream == "B":
            return self.model_b.predict_proba(X)[:, 1]
        else:
            raise ValueError("Stream must be 'A' or 'B'")

    def predict(self, X, stream):
        """
        Predicts binary class based on the stream-specific optimized threshold.
        """
        proba = self.predict_proba(X, stream)
        threshold = self.threshold_a if stream == "A" else self.threshold_b
        return (proba >= threshold).astype(int)


def apply_undersampling(X, y, ids, ratio, seed=42):
    """
    Applies targeted majority undersampling.
    Retains 100% of positives and subsamples negatives to the specified ratio.
    """
    np.random.seed(seed)

    # Identify indices
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_pos = len(pos_indices)
    n_neg_keep = int(n_pos * ratio)

    # If we have fewer negatives than the ratio implies, keep all negatives
    if n_neg_keep > len(neg_indices):
        n_neg_keep = len(neg_indices)

    print(
        f"Undersampling: Positives={n_pos}, Negatives Original={len(neg_indices)}, Keeping={n_neg_keep}"
    )

    # Randomly sample negatives
    neg_indices_sampled = np.random.choice(neg_indices, size=n_neg_keep, replace=False)

    # Combine and shuffle
    all_indices = np.concatenate([pos_indices, neg_indices_sampled])
    np.random.shuffle(all_indices)

    # Filter data
    # X is a DataFrame, y and ids are numpy arrays
    X_resampled = X.iloc[all_indices].copy()
    y_resampled = y[all_indices]
    ids_resampled = ids[all_indices]

    return X_resampled, y_resampled, ids_resampled


def train_stream_model(stream_name, X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost classifier for a specific stream.
    """
    print(f"\n--- Training Stream {stream_name} Model ---")

    if stream_name == "A":
        params = Config.STREAM_A_MODEL_PARAMS
    elif stream_name == "B":
        params = Config.STREAM_B_MODEL_PARAMS
    else:
        raise ValueError("Unknown stream name")

    # Initialize model
    clf = xgb.XGBClassifier(
        **params, early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
    )

    # Train with early stopping
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=Config.VERBOSE_EVAL,
    )

    return clf


def optimize_threshold(model, X_val, y_val, stream_name):
    """
    Finds the probability threshold that maximizes MCC on the validation set.
    """
    print(f"\n--- Optimizing Threshold for Stream {stream_name} ---")

    # Get probabilities
    probas = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_mcc = -1.0

    # Search space
    start, stop, step = Config.THRESHOLD_SEARCH
    thresholds = np.arange(start, stop, step)

    for thresh in thresholds:
        preds = (probas >= thresh).astype(int)
        mcc = matthews_corrcoef(y_val, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    print(f"Stream {stream_name} Best Threshold: {best_threshold:.4f}")
    print(f"Stream {stream_name} Best Validation MCC: {best_mcc}")

    return best_threshold


def train_and_validate(load_cached_data=True):
    """
    Orchestrates the training pipeline:
    1. Loads data.
    2. Undersamples training data.
    3. Trains models.
    4. Optimizes thresholds.
    5. Returns the predictor.
    """
    seed_everything(Config.SEED)

    # Initialize Builders
    train_builder = DatasetBuilder("train", load_cached_data)
    val_builder = DatasetBuilder("validation", load_cached_data)

    # --- Stream A (Interaction) ---
    print("\nProcessing Stream A...")
    X_train_a, ids_train_a, y_train_a = train_builder.build_dataset("A")
    X_val_a, ids_val_a, y_val_a = val_builder.build_dataset("A")

    # Undersample Train A
    X_train_a_res, y_train_a_res, _ = apply_undersampling(
        X_train_a, y_train_a, ids_train_a, Config.NEGATIVE_SAMPLING_RATIO, Config.SEED
    )

    # Train A
    model_a = train_stream_model("A", X_train_a_res, y_train_a_res, X_val_a, y_val_a)

    # Optimize Threshold A
    thresh_a = optimize_threshold(model_a, X_val_a, y_val_a, "A")

    # Cleanup A
    del X_train_a, X_train_a_res, y_train_a, y_train_a_res, ids_train_a
    del X_val_a, y_val_a, ids_val_a
    gc.collect()

    # --- Stream B (Impact) ---
    print("\nProcessing Stream B...")
    X_train_b, ids_train_b, y_train_b = train_builder.build_dataset("B")
    X_val_b, ids_val_b, y_val_b = val_builder.build_dataset("B")

    # Undersample Train B
    X_train_b_res, y_train_b_res, _ = apply_undersampling(
        X_train_b, y_train_b, ids_train_b, Config.NEGATIVE_SAMPLING_RATIO, Config.SEED
    )

    # Train B
    model_b = train_stream_model("B", X_train_b_res, y_train_b_res, X_val_b, y_val_b)

    # Optimize Threshold B
    thresh_b = optimize_threshold(model_b, X_val_b, y_val_b, "B")

    # Cleanup B
    del X_train_b, X_train_b_res, y_train_b, y_train_b_res, ids_train_b
    del X_val_b, y_val_b, ids_val_b
    gc.collect()

    predictor = DualStreamPredictor(model_a, model_b, thresh_a, thresh_b)
    return predictor


def generate_submission(predictor, load_cached_data=True):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("\n--- Generating Submission ---")
    test_builder = DatasetBuilder("test", load_cached_data)

    # --- Stream A Predictions ---
    print("Predicting Stream A (Interactions)...")
    X_test_a, ids_test_a, _ = test_builder.build_dataset("A")

    if not X_test_a.empty:
        preds_a = predictor.predict(X_test_a, "A")
        df_a = pd.DataFrame({"contact_id": ids_test_a, "contact": preds_a})
    else:
        df_a = pd.DataFrame(columns=["contact_id", "contact"])

    del X_test_a
    gc.collect()

    # --- Stream B Predictions ---
    print("Predicting Stream B (Impacts)...")
    X_test_b, ids_test_b, _ = test_builder.build_dataset("B")

    if not X_test_b.empty:
        preds_b = predictor.predict(X_test_b, "B")
        df_b = pd.DataFrame({"contact_id": ids_test_b, "contact": preds_b})
    else:
        df_b = pd.DataFrame(columns=["contact_id", "contact"])

    del X_test_b
    gc.collect()

    # --- Combine and Format ---
    print("Combining predictions...")
    df_preds = pd.concat([df_a, df_b], ignore_index=True)

    # Load sample submission to ensure all IDs are present and order is preserved
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Map predictions to sample submission
    # We use map instead of merge to preserve exact order of sample_submission
    pred_map = df_preds.set_index("contact_id")["contact"].to_dict()

    # Fill missing with 0 (safe default)
    sample_sub["contact"] = sample_sub["contact_id"].map(pred_map).fillna(0).astype(int)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(sample_sub)}")
    print(f"Positive predictions: {sample_sub['contact'].sum()}")
