import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from library import config
from library import feature_engine


def get_class_names():
    """
    Reconstructs the list of class names (breeds) exactly as done in the Dataset class.
    This ensures that the model's integer class indices map correctly to the breed names.
    """
    if os.path.exists(config.TRAIN_CSV):
        train_df = pd.read_csv(config.TRAIN_CSV)
        classes = sorted(train_df["breed"].unique().tolist())
        return classes
    else:
        raise FileNotFoundError(f"Training metadata not found at {config.TRAIN_CSV}")


def train_classifier(load_cached_data=True, save_model=True):
    """
    Trains a LogisticRegressionCV classifier on the extracted features.
    Evaluates performance on the validation set.

    Args:
        load_cached_data (bool): Whether to use cached features from feature_engine.
        save_model (bool): Whether to save the trained model to disk.

    Returns:
        model: The trained sklearn model.
        float: The validation log loss.
    """
    print("Loading training data...")
    X_train, y_train, _ = feature_engine.extract_features(
        "train", load_cached_data=load_cached_data
    )

    print("Loading validation data...")
    X_val, y_val, _ = feature_engine.extract_features(
        "val", load_cached_data=load_cached_data
    )

    print(f"Training LogisticRegressionCV with input shape {X_train.shape}...")

    # Initialize model with config parameters
    # Note: We must ensure random_state is passed if it's in the params, which it is.
    clf = LogisticRegressionCV(**config.LOGREG_PARAMS)

    # Fit the model
    clf.fit(X_train, y_train)

    # Evaluate on Validation set
    print("Evaluating on validation set...")
    y_pred_proba = clf.predict_proba(X_val)

    # Calculate Log Loss
    # labels are integers 0..N-1, predict_proba returns matrix (n_samples, n_classes)
    val_loss = log_loss(y_val, y_pred_proba, labels=clf.classes_)

    print(f"Validation Multi Class Log Loss: {val_loss}")

    if save_model:
        model_path = os.path.join(config.WORKING_DIR, "logreg_model.joblib")
        print(f"Saving model to {model_path}...")
        joblib.dump(clf, model_path)

    return clf, val_loss


def predict_submission(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained sklearn model.
        load_cached_data (bool): Whether to use cached features from feature_engine.
    """
    print("Loading test data...")
    X_test, _, test_ids = feature_engine.extract_features(
        "test", load_cached_data=load_cached_data
    )

    print(f"Predicting probabilities for {len(X_test)} test images...")
    y_pred_proba = model.predict_proba(X_test)

    # Get class names to use as column headers
    class_names = get_class_names()

    # Verify shape consistency
    if len(class_names) != y_pred_proba.shape[1]:
        raise ValueError(
            f"Mismatch between number of classes ({len(class_names)}) "
            f"and model output dimensions ({y_pred_proba.shape[1]})."
        )

    # Create Submission DataFrame
    # Format: id, breed1, breed2, ...
    submission_df = pd.DataFrame(y_pred_proba, columns=class_names)
    submission_df.insert(0, "id", test_ids)

    # Save to disk
    print(f"Saving submission to {config.SUBMISSION_FILE}...")
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)

    print("Submission generation complete.")
