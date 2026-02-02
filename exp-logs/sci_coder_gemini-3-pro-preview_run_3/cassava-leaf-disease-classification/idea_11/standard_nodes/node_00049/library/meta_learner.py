import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from library.config import Config


def fit_meta_learner(oof_preds_list, targets, save_model=True):
    """
    Trains a Logistic Regression meta-learner on concatenated OOF predictions from base models.
    This implements the 'Level 1' stacking stage.

    Args:
        oof_preds_list (list of np.ndarray): List of OOF probability arrays.
            Each element corresponds to the aggregated OOF predictions of one architecture.
            Shape of each array: (N_samples, N_classes).
        targets (np.ndarray): True labels for the training set. Shape: (N_samples,).
        save_model (bool): Whether to save the trained model to the working directory.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained meta-learner model.
    """
    # 1. Feature Engineering: Concatenate OOF probabilities
    # If we have 3 architectures and 5 classes, the feature vector size will be 15.
    X = np.concatenate(oof_preds_list, axis=1)
    y = targets

    print(f"Meta-Learner Training Data Shape: {X.shape}")

    # 2. Initialize Meta-Learner
    # We use Logistic Regression as it is robust and less prone to overfitting on meta-features
    meta_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=Config.SEED,
        C=1.0,
        max_iter=1000,
    )

    # 3. Train
    print("Training Meta-Learner (Logistic Regression)...")
    meta_model.fit(X, y)

    # 4. Evaluate on OOF Data
    # This score represents the cross-validated accuracy of the full ensemble
    preds = meta_model.predict(X)
    acc = accuracy_score(y, preds)

    # Print full precision as requested
    print(f"Meta-Learner OOF Accuracy: {acc}")

    # 5. Save Model
    if save_model:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(Config.OUTPUT_DIR, "meta_learner_logistic.joblib")
        joblib.dump(meta_model, model_path)
        print(f"Meta-Learner saved to {model_path}")

    return meta_model


def predict_meta_learner(test_preds_list, meta_model):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        test_preds_list (list of np.ndarray): List of test probability arrays.
            Must match the order and structure used in fit_meta_learner.
            Each array shape: (N_test_samples, N_classes).
        meta_model: The trained LogisticRegression model.

    Returns:
        tuple: (final_class_predictions, final_probabilities)
            - final_class_predictions (np.ndarray): Predicted class indices.
            - final_probabilities (np.ndarray): Predicted probabilities from the meta-learner.
    """
    # 1. Prepare Features
    # Concatenate inputs to match training feature space
    X_test = np.concatenate(test_preds_list, axis=1)

    # 2. Predict
    final_probs = meta_model.predict_proba(X_test)
    final_preds = np.argmax(final_probs, axis=1)

    return final_preds, final_probs


def generate_submission(test_preds_list, meta_model, image_ids):
    """
    Orchestrates the prediction generation and saves the submission file.

    Args:
        test_preds_list (list of np.ndarray): List of test probability arrays from base models.
        meta_model: Trained meta-learner.
        image_ids (list or np.ndarray): List of image IDs corresponding to the test set.
    """
    print("Generating final submission...")

    # 1. Get Predictions
    final_preds, _ = predict_meta_learner(test_preds_list, meta_model)

    # 2. Create DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "label": final_preds})

    # 3. Save to Disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Submission Head:")
    print(submission_df.head())
