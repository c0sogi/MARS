import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from library.config import Config


def prepare_meta_features(
    class_preds_dict,
    aux_preds_dict,
    actual_aux_targets,
    cache_name=None,
    load_cached_data=True,
):
    """
    Constructs the meta-feature matrix for the Trust-Aware Stacking Ensemble.

    Features for each base model:
    1. Predicted Class Probability (P_cls)
    2. Trust Score = |Predicted_Aux - Actual_Aux| (Absolute Error of file size prediction)

    Args:
        class_preds_dict (dict): Dictionary {model_name: np.array of shape (N,)} containing class probabilities.
        aux_preds_dict (dict): Dictionary {model_name: np.array of shape (N,)} containing auxiliary predictions.
        actual_aux_targets (np.array): Array of shape (N,) containing ground truth auxiliary targets (normalized log file sizes).
        cache_name (str, optional): Filename to cache the resulting matrix.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.array: The meta-feature matrix X of shape (N, 2 * num_models).
    """
    # 1. Caching Logic
    if cache_name:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        if load_cached_data and os.path.exists(cache_path):
            try:
                X_meta = np.load(cache_path)
                # print(f"Loaded meta-features from cache: {cache_path}")
                return X_meta
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}")

    # 2. Feature Construction
    # Ensure deterministic order of features based on model names
    model_names = sorted(class_preds_dict.keys())

    feature_list = []

    # Ensure actual targets are shaped correctly for broadcasting
    if actual_aux_targets.ndim == 1:
        actual_aux_targets = actual_aux_targets.reshape(-1, 1)

    for name in model_names:
        # -- Feature 1: Class Probability --
        p_cls = class_preds_dict[name]
        if p_cls.ndim == 1:
            p_cls = p_cls.reshape(-1, 1)
        feature_list.append(p_cls)

        # -- Feature 2: Trust Score (Auxiliary Error) --
        p_aux = aux_preds_dict[name]
        if p_aux.ndim == 1:
            p_aux = p_aux.reshape(-1, 1)

        # Trust Score is the absolute error.
        # The meta-learner will learn that higher error -> lower trust (negative coefficient likely).
        trust_score = np.abs(p_aux - actual_aux_targets)
        feature_list.append(trust_score)

    X_meta = np.hstack(feature_list)

    # 3. Save to Cache
    if cache_name:
        np.save(cache_path, X_meta)
        # print(f"Saved meta-features to cache: {cache_path}")

    return X_meta


def train_meta_learner(X_train, y_train, random_state=Config.SEED):
    """
    Trains the Logistic Regression meta-learner.

    Args:
        X_train (np.array): Training meta-features.
        y_train (np.array): Ground truth labels.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained model.
    """
    # L-BFGS is a good solver for this scale of data (~14k rows, small feature count)
    model = LogisticRegression(solver="lbfgs", random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def predict_meta_learner(model, X_test):
    """
    Generates calibrated probabilities using the meta-learner.

    Args:
        model: Trained LogisticRegression model.
        X_test (np.array): Test meta-features.

    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    # Return probability of class 1
    return model.predict_proba(X_test)[:, 1]


def save_submission(ids, preds, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): Image IDs.
        preds (list or np.array): Predicted probabilities.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame({"id": ids, "has_cactus": preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_stacking(
    train_class_preds,
    train_aux_preds,
    train_aux_targets,
    train_labels,
    test_class_preds,
    test_aux_preds,
    test_aux_targets,
    test_ids,
    submission_path=Config.SUBMISSION_PATH,
):
    """
    Orchestrates the full stacking pipeline:
    1. Prepare Train Meta-Features (from OOF predictions).
    2. Train Meta-Learner.
    3. Prepare Test Meta-Features.
    4. Predict on Test Set.
    5. Save Submission.

    Args:
        train_class_preds (dict): OOF class predictions per model.
        train_aux_preds (dict): OOF auxiliary predictions per model.
        train_aux_targets (np.array): Actual auxiliary targets for training set.
        train_labels (np.array): Ground truth labels for training set.
        test_class_preds (dict): Test class predictions per model.
        test_aux_preds (dict): Test auxiliary predictions per model.
        test_aux_targets (np.array): Actual auxiliary targets for test set (computed from files).
        test_ids (np.array): IDs for test set images.
        submission_path (str): Path to save the final submission.

    Returns:
        np.array: Final test predictions.
    """
    print("--- Stacking Phase Started ---")

    # 1. Prepare Training Data
    print("Preparing training meta-features...")
    X_train = prepare_meta_features(
        train_class_preds,
        train_aux_preds,
        train_aux_targets,
        cache_name="meta_X_train.npy",
        load_cached_data=True,
    )

    # 2. Train Model
    print(
        f"Training Meta-Learner on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    meta_model = train_meta_learner(X_train, train_labels)

    # Save the meta-model for future reference
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(Config.CHECKPOINT_DIR, "meta_model.joblib")
    joblib.dump(meta_model, model_path)
    print(f"Meta-learner saved to {model_path}")

    # 3. Prepare Test Data
    print("Preparing test meta-features...")
    X_test = prepare_meta_features(
        test_class_preds,
        test_aux_preds,
        test_aux_targets,
        cache_name="meta_X_test.npy",
        load_cached_data=True,
    )

    # 4. Predict
    print("Predicting on test set...")
    final_preds = predict_meta_learner(meta_model, X_test)

    # 5. Save Submission
    save_submission(test_ids, final_preds, submission_path)

    return final_preds
