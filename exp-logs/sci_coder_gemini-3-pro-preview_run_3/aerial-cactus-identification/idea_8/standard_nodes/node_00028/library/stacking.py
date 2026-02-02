import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from library.utils import calculate_roc_auc


def train_meta_learner(
    oof_preds_dict, targets, output_dir=None, save_name="meta_learner.pkl", seed=42
):
    """
    Trains a Logistic Regression meta-learner on Out-Of-Fold (OOF) predictions.

    Args:
        oof_preds_dict (dict): Dictionary where keys are model names and values are
                               1D numpy arrays of OOF probabilities (shape: N_samples).
                               All arrays must be aligned to the same sample order.
        targets (np.array): 1D numpy array of ground truth labels (shape: N_samples).
        output_dir (str, optional): Directory to save the trained model.
        save_name (str, optional): Filename for the saved model.
        seed (int): Random seed for reproducibility.

    Returns:
        model: The trained LogisticRegression model.
        float: The ROC AUC score of the ensemble on the OOF data.
    """
    # Ensure consistency in feature order
    model_names = sorted(oof_preds_dict.keys())
    print(f"Training Meta-Learner on {len(model_names)} base models: {model_names}")

    # Construct feature matrix X
    # Shape: (N_samples, N_models)
    X = np.column_stack([oof_preds_dict[name] for name in model_names])
    y = np.array(targets)

    # Check shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in samples: X has {X.shape[0]}, y has {y.shape[0]}")

    # Initialize Logistic Regression
    # We use non-negative weights constraint if possible to avoid overfitting,
    # but standard LR is usually robust enough. Here we use standard LR.
    meta_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",  # Good for small datasets
        random_state=seed,
        fit_intercept=True,
    )

    # Fit model
    meta_model.fit(X, y)

    # Evaluate on OOF (Self-validation)
    # Note: Since these are OOF predictions, this is a valid estimate of generalization
    preds = meta_model.predict_proba(X)[:, 1]
    score = calculate_roc_auc(y, preds)

    print(f"Meta-Learner OOF ROC AUC: {score:.6f}")

    # Print coefficients to see model contributions
    print("Meta-Learner Coefficients:")
    for name, coef in zip(model_names, meta_model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {meta_model.intercept_[0]:.4f}")

    # Save model
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)

        # We also save the feature names to verify order during inference
        artifact = {"model": meta_model, "feature_names": model_names}
        joblib.dump(artifact, save_path)
        print(f"Meta-learner saved to {save_path}")

    return meta_model


def predict_stack(meta_model_artifact, test_preds_dict):
    """
    Generates predictions using the trained meta-learner.

    Args:
        meta_model_artifact (dict or object): The trained model artifact (dict with 'model' and 'feature_names')
                                              or just the sklearn model object.
        test_preds_dict (dict): Dictionary where keys are model names and values are
                                1D numpy arrays of test probabilities.

    Returns:
        np.array: Final ensemble probabilities.
    """
    # Handle artifact format
    if isinstance(meta_model_artifact, dict) and "model" in meta_model_artifact:
        model = meta_model_artifact["model"]
        expected_features = meta_model_artifact.get("feature_names", None)
    else:
        model = meta_model_artifact
        expected_features = None

    # Determine feature order
    if expected_features:
        feature_names = expected_features
        # Validate that all required models are present
        missing = [f for f in feature_names if f not in test_preds_dict]
        if missing:
            raise ValueError(
                f"Missing predictions for models required by meta-learner: {missing}"
            )
    else:
        # Fallback to sorting if no metadata provided (assumes same protocol as training)
        feature_names = sorted(test_preds_dict.keys())

    # Construct feature matrix X_test
    X_test = np.column_stack([test_preds_dict[name] for name in feature_names])

    # Predict
    # predict_proba returns [prob_class_0, prob_class_1]
    final_preds = model.predict_proba(X_test)[:, 1]

    return final_preds


def generate_submission(ids, preds, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs.
        preds (list or np.array): List of probabilities.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame({"id": ids, "has_cactus": preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(df.head())
