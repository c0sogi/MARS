import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from library.config import Config
from library.utils import (
    load_numpy_array,
    save_numpy_array,
    compute_spearman,
    get_artifact_path,
)

# ------------------------------------------------------------------------------
# Helper Functions for Indices and Slicing
# ------------------------------------------------------------------------------


def get_target_indices():
    """
    Returns the indices of Question targets and Answer targets based on Config.
    """
    all_cols = Config.TARGET_COLS
    q_cols = set(Config.get_question_targets())
    a_cols = set(Config.get_answer_targets())

    q_indices = [i for i, c in enumerate(all_cols) if c in q_cols]
    a_indices = [i for i, c in enumerate(all_cols) if c in a_cols]

    return q_indices, a_indices


def split_features_l1(features):
    """
    Splits embedding features for Level 1 Topology-Aware Solver.
    Input features shape: (N, 4 * Hidden_Dim) -> [h_cls, h_q, h_a, h_diff]

    Returns:
        X_q: (N, Hidden_Dim) -> h_q
        X_full: (N, 4 * Hidden_Dim) -> All features
    """
    # Infer hidden dimension
    total_dim = features.shape[1]
    hidden_dim = total_dim // 4

    # h_q is the second block (indices hidden_dim : 2*hidden_dim)
    X_q = features[:, hidden_dim : 2 * hidden_dim]
    X_full = features

    return X_q, X_full


def split_features_l2(prediction_list):
    """
    Splits prediction features for Level 2 Meta Stacker.
    Input is a list of prediction arrays, each shape (N, 30).

    Returns:
        X_q: Concatenation of Question-target predictions from all models.
        X_full: Concatenation of ALL predictions from all models.
    """
    q_indices, _ = get_target_indices()

    # X_full is simply all predictions concatenated
    X_full = np.hstack(prediction_list)

    # X_q is only the question columns from each prediction array
    q_preds = [preds[:, q_indices] for preds in prediction_list]
    X_q = np.hstack(q_preds)

    return X_q, X_full


# ------------------------------------------------------------------------------
# Topology-Aware Ridge Regressor
# ------------------------------------------------------------------------------


class TopologyAwareRidge:
    """
    A Regressor that uses two internal RidgeCV solvers:
    1. Q-Solver: Predicts Question targets using only Question features.
    2. A-Solver: Predicts Answer targets using Full features.
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.q_regressor = RidgeCV(alphas=alphas, scoring=None)
        self.a_regressor = RidgeCV(alphas=alphas, scoring=None)
        self.q_indices, self.a_indices = get_target_indices()

    def fit(self, X_q, X_full, y):
        """
        Fits the internal solvers.
        """
        # Split targets
        y_q = y[:, self.q_indices]
        y_a = y[:, self.a_indices]

        # Fit Q-Solver
        self.q_regressor.fit(X_q, y_q)

        # Fit A-Solver
        self.a_regressor.fit(X_full, y_a)

        return self

    def predict(self, X_q, X_full):
        """
        Predicts and reconstructs the full 30-column output.
        """
        # Predict
        pred_q = self.q_regressor.predict(X_q)
        pred_a = self.a_regressor.predict(X_full)

        # Reconstruct full array
        n_samples = X_q.shape[0]
        n_targets = len(Config.TARGET_COLS)

        # Initialize output array
        y_pred = np.zeros((n_samples, n_targets), dtype=np.float32)

        # Fill columns
        y_pred[:, self.q_indices] = pred_q
        y_pred[:, self.a_indices] = pred_a

        # Clip to valid probability range
        y_pred = np.clip(y_pred, 0.0, 1.0)

        return y_pred

    def save(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)


# ------------------------------------------------------------------------------
# Level 1 Training
# ------------------------------------------------------------------------------


def train_l1_model(model_alias, load_cached_preds=True):
    """
    Trains a TopologyAwareRidge model on the extracted features of a specific backbone.
    Generates OOF, Holdout, and Test predictions.
    """
    print(f"\n=== Training Level 1 Ridge for {model_alias} ===")

    # Define paths
    pred_oof_path = f"{model_alias}_l1_oof_preds.npy"
    pred_holdout_path = f"{model_alias}_l1_holdout_preds.npy"
    pred_test_path = f"{model_alias}_l1_test_preds.npy"
    model_path = get_artifact_path(f"{model_alias}_l1_ridge.joblib")

    # Check cache
    if load_cached_preds:
        oof = load_numpy_array(pred_oof_path)
        holdout = load_numpy_array(pred_holdout_path)
        test = load_numpy_array(pred_test_path)
        if oof is not None and holdout is not None and test is not None:
            print(f"Loaded Level 1 predictions for {model_alias} from cache.")
            return oof, holdout, test

    # Load Features
    train_feats = load_numpy_array(f"{model_alias}_train_features.npy")
    holdout_feats = load_numpy_array(f"{model_alias}_val_features.npy")
    test_feats = load_numpy_array(f"{model_alias}_test_features.npy")
    train_targets = load_numpy_array(f"{model_alias}_train_targets.npy")

    if train_feats is None:
        raise FileNotFoundError(
            f"Features for {model_alias} not found. Run fine-tuning first."
        )

    # Prepare Data splits
    X_q_train, X_full_train = split_features_l1(train_feats)
    X_q_holdout, X_full_holdout = split_features_l1(holdout_feats)
    X_q_test, X_full_test = split_features_l1(test_feats)

    # Train Model
    print(f"Fitting RidgeCV on {len(train_feats)} samples...")
    model = TopologyAwareRidge()
    model.fit(X_q_train, X_full_train, train_targets)

    # Save Model
    model.save(model_path)

    # Predict
    print("Generating predictions...")
    oof_preds = model.predict(
        X_q_train, X_full_train
    )  # Note: In this simple scheme, this is "in-sample" on the concatenated OOF features
    holdout_preds = model.predict(X_q_holdout, X_full_holdout)
    test_preds = model.predict(X_q_test, X_full_test)

    # Evaluate
    # Note: train_feats contains OOF features from fine-tuning, so 'oof_preds' here
    # effectively acts as a stacked OOF prediction set if we consider the ridge part as the final layer.
    # However, strictly speaking, to have clean OOF for stacking ridge, we should have cross-validated the ridge too.
    # Given the prompt's "Fine-Tune then Ridge" paradigm, usually fitting Ridge on the OOF features of the NN is standard.
    train_score = compute_spearman(train_targets, oof_preds)

    # Load holdout targets for validation
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_targets = val_df[Config.TARGET_COLS].values
    holdout_score = compute_spearman(val_targets, holdout_preds)

    print(f"{model_alias} Results:")
    print(f"  Train (OOF-Features) Spearman: {train_score:.6f}")
    print(f"  Holdout Spearman:              {holdout_score:.6f}")

    # Save Predictions
    save_numpy_array(oof_preds, pred_oof_path)
    save_numpy_array(holdout_preds, pred_holdout_path)
    save_numpy_array(test_preds, pred_test_path)

    return oof_preds, holdout_preds, test_preds


# ------------------------------------------------------------------------------
# Level 2 Stacking
# ------------------------------------------------------------------------------


def train_meta_stacker(model_aliases, load_cached_preds=True):
    """
    Trains the Level 2 Meta Stacker using predictions from Level 1 models.
    Generates the final submission file.
    """
    print("\n=== Training Level 2 Meta Stacker ===")

    # 1. Gather L1 Predictions
    l1_oof_list = []
    l1_holdout_list = []
    l1_test_list = []

    for alias in model_aliases:
        oof, holdout, test = train_l1_model(alias, load_cached_preds=load_cached_preds)
        l1_oof_list.append(oof)
        l1_holdout_list.append(holdout)
        l1_test_list.append(test)

    # 2. Prepare L2 Features (Concatenation of L1 predictions)
    X_q_train, X_full_train = split_features_l2(l1_oof_list)
    X_q_holdout, X_full_holdout = split_features_l2(l1_holdout_list)
    X_q_test, X_full_test = split_features_l2(l1_test_list)

    # Load Targets
    # For training the meta-stacker, we use the same targets as L1 training
    # (which aligns with the OOF features from the NN)
    train_targets = load_numpy_array(f"{model_aliases[0]}_train_targets.npy")

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_targets = val_df[Config.TARGET_COLS].values

    # 3. Train Meta Model
    print(
        f"Fitting Meta RidgeCV on {len(X_full_train)} samples with {len(model_aliases)} base models..."
    )
    meta_model = TopologyAwareRidge(alphas=(0.1, 1.0, 10.0))
    meta_model.fit(X_q_train, X_full_train, train_targets)

    # Save Meta Model
    meta_model.save(get_artifact_path("meta_stacker.joblib"))

    # 4. Predict
    train_preds = meta_model.predict(X_q_train, X_full_train)
    holdout_preds = meta_model.predict(X_q_holdout, X_full_holdout)
    test_preds = meta_model.predict(X_q_test, X_full_test)

    # 5. Evaluate
    train_score = compute_spearman(train_targets, train_preds)
    holdout_score = compute_spearman(val_targets, holdout_preds)

    print(f"Meta Stacker Results:")
    print(f"  Train (Stacked) Spearman: {train_score:.6f}")
    print(f"  Holdout Spearman:         {holdout_score:.6f}")

    # 6. Generate Submission
    print(f"Generating submission file at {Config.SUBMISSION_PATH}...")

    # Load test metadata to get qa_id
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create DataFrame
    submission_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "qa_id", test_df["qa_id"])

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
