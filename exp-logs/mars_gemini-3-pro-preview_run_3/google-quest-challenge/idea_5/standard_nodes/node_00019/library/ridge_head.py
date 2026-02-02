import os
import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from library.config import GlobalConfig


def train_ridge_head(train_features, train_targets, model_path, alphas=None):
    """
    Trains a Ridge Regression head on the extracted features using Leave-One-Out Cross-Validation.
    Includes StandardScaler in the pipeline.

    Args:
        train_features (np.ndarray): The input features for training. Shape (N, Feature_Dim).
        train_targets (np.ndarray): The target values. Shape (N, Num_Targets).
        model_path (str): The file path to save the trained model.
        alphas (list of float, optional): List of alpha values for regularization.
                                          Defaults to [0.1, 1.0, 10.0, 100.0].

    Returns:
        sklearn.pipeline.Pipeline: The trained pipeline (Scaler + Ridge).
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]

    print(f"Training RidgeCV with alphas: {alphas} and StandardScaler")
    print(f"Features shape: {train_features.shape}")
    print(f"Targets shape: {train_targets.shape}")

    # Create a pipeline with scaling and ridge regression
    # RidgeCV with cv=None performs efficient Leave-One-Out Cross-Validation (LOOCV)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, scoring=None))
    model.fit(train_features, train_targets)

    # Access the RidgeCV step to print best alpha
    ridge_step = model.named_steps["ridgecv"]
    print(f"Best Alpha: {ridge_step.alpha_}")

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained model using joblib
    joblib.dump(model, model_path)
    print(f"Ridge Pipeline saved to {model_path}")

    return model


def predict_ridge(test_features, model_path):
    """
    Generates predictions using a trained Ridge pipeline.

    Args:
        test_features (np.ndarray): The input features for prediction. Shape (N, Feature_Dim).
        model_path (str): The file path to the trained model.

    Returns:
        np.ndarray: Predicted probabilities clipped to the range [0, 1].
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ridge model not found at {model_path}")

    print(f"Loading Ridge pipeline from {model_path}...")
    model = joblib.load(model_path)

    print(f"Predicting for {test_features.shape[0]} samples...")
    preds = model.predict(test_features)

    # Ridge regression is a linear model and can produce values outside the [0, 1] range.
    # We clip the predictions to ensure valid probabilities.
    preds = np.clip(preds, 0.0, 1.0)

    return preds
