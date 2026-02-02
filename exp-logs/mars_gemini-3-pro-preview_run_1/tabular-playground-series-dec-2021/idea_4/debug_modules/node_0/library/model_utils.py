import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from library.config import MODEL_PARAMS, PIPELINE_PARAMS, DATA_PATHS, ID_COL, TARGET_COL


def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Trains an XGBoost Classifier with the given parameters and data.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training targets.
        X_val (pd.DataFrame or np.ndarray, optional): Validation features.
        y_val (pd.Series or np.ndarray, optional): Validation targets.
        params (dict, optional): Hyperparameters overriding config.MODEL_PARAMS.

    Returns:
        xgb.XGBClassifier: The trained model.
    """
    # Merge default params with overrides
    clf_params = MODEL_PARAMS.copy()
    if params:
        clf_params.update(params)

    # Initialize classifier
    clf = xgb.XGBClassifier(**clf_params)

    fit_params = {}

    # Handle validation set and early stopping
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = False

        # Add early stopping if configured
        es_rounds = PIPELINE_PARAMS.get("early_stopping_rounds")
        if es_rounds:
            fit_params["early_stopping_rounds"] = es_rounds

    # Train
    clf.fit(X_train, y_train, **fit_params)

    # Evaluation
    if X_val is not None and y_val is not None:
        # Predict uses the best iteration automatically if early stopping was triggered
        val_preds = clf.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        print(f"Validation Accuracy: {acc}")

    return clf


def predict_proba(model, X):
    """
    Generates class probabilities for the input data.

    Args:
        model (xgb.XGBClassifier): Trained model.
        X (pd.DataFrame or np.ndarray): Input features.

    Returns:
        np.ndarray: Predicted probabilities (N_samples, N_classes).
    """
    return model.predict_proba(X)


def predict(model, X):
    """
    Generates class predictions for the input data.

    Args:
        model (xgb.XGBClassifier): Trained model.
        X (pd.DataFrame or np.ndarray): Input features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    return model.predict(X)


def generate_submission(models, X_test, output_path):
    """
    Generates a submission file by averaging predictions from multiple models.

    Args:
        models (list): List of trained xgb.XGBClassifier models.
        X_test (pd.DataFrame or np.ndarray): Test features.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating submission with {len(models)} models...")

    # Load sample submission to get correct IDs
    sample_sub = pd.read_csv(DATA_PATHS["sample_submission"])
    ids = sample_sub[ID_COL]

    # Ensure X_test aligns with sample_sub (assuming row order is preserved from test.csv)
    if len(X_test) != len(ids):
        raise ValueError(
            f"Shape mismatch: X_test has {len(X_test)} rows, but sample_submission has {len(ids)}."
        )

    # Average probabilities across all models (Homogeneous Ensemble)
    avg_probs = None
    for i, model in enumerate(models):
        probs = model.predict_proba(X_test)
        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs += probs

    avg_probs /= len(models)

    # Get class labels from the first model
    # XGBClassifier.classes_ maps the column index of predict_proba to the actual class label
    classes = models[0].classes_

    # Determine predicted class (argmax)
    pred_indices = np.argmax(avg_probs, axis=1)
    pred_labels = classes[pred_indices]

    # Create submission DataFrame
    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: pred_labels})

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
