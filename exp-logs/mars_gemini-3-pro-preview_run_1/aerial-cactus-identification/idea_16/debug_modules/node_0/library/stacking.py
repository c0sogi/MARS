import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config


def prepare_meta_features(preds_dict, meta_features=None):
    """
    Constructs the feature matrix for the meta-learner.

    Args:
        preds_dict (dict): Dictionary where keys are model names and values are
                           numpy arrays of probabilities (N,).
        meta_features (np.ndarray, optional): Array of metadata features (N,).
                                              Expected to be normalized file sizes.

    Returns:
        tuple: (X, feature_names)
            X (np.ndarray): Feature matrix of shape (N, n_models + n_meta_feats).
            feature_names (list): List of feature names corresponding to columns.
    """
    # Sort keys to ensure consistent column order between train and test
    model_names = sorted(preds_dict.keys())

    # Stack model predictions into columns
    # Reshape (N,) arrays to (N, 1)
    feature_list = [preds_dict[name].reshape(-1, 1) for name in model_names]

    # Append metadata if provided
    feature_names = list(model_names)
    if meta_features is not None:
        feature_list.append(meta_features.reshape(-1, 1))
        feature_names.append("file_size")

    X = np.hstack(feature_list)
    return X, feature_names


def train_meta_learner(
    oof_preds_dict, y_true, meta_features=None, random_state=Config.SEED
):
    """
    Trains a Logistic Regression meta-learner using OOF predictions and metadata.

    Args:
        oof_preds_dict (dict): Dictionary of OOF predictions {model_name: probs_array}.
        y_true (np.ndarray): Ground truth binary labels.
        meta_features (np.ndarray, optional): Normalized file sizes for the training set.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained meta-model.
    """
    print("Training Meta-Learner (Stacking)...")

    # Prepare input matrix
    X, feature_names = prepare_meta_features(oof_preds_dict, meta_features)

    # Initialize Logistic Regression
    # 'liblinear' is a good choice for small-to-medium datasets and binary classification
    meta_model = LogisticRegression(
        solver="liblinear", C=1.0, random_state=random_state, fit_intercept=True
    )

    # Fit the model
    meta_model.fit(X, y_true)

    # Print learned coefficients for interpretation
    print("Meta-Learner Coefficients:")
    coefs = meta_model.coef_[0]
    for name, coef in zip(feature_names, coefs):
        print(f"  {name}: {coef:.6f}")
    print(f"  Intercept: {meta_model.intercept_[0]:.6f}")

    return meta_model


def generate_final_predictions(
    meta_model, test_preds_dict, test_ids, meta_features=None, output_path=None
):
    """
    Generates final predictions using the trained meta-learner and saves the submission file.

    Args:
        meta_model: Trained LogisticRegression model.
        test_preds_dict (dict): Dictionary of test predictions {model_name: probs_array}.
        test_ids (np.ndarray): Array of image IDs (filenames) for the test set.
        meta_features (np.ndarray, optional): Normalized file sizes for the test set.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_DIR.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print(f"Generating stacked predictions for {len(test_ids)} test samples...")

    # Prepare test feature matrix
    # Note: prepare_meta_features sorts keys, ensuring column alignment with training
    X_test, _ = prepare_meta_features(test_preds_dict, meta_features)

    # Predict probabilities for the positive class (1)
    final_probs = meta_model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Stacked submission saved to {output_path}")

    return submission_df
