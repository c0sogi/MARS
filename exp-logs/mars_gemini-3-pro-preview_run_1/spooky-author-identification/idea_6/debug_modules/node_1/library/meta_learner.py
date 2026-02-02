import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from library.configuration import Config
from library.utilities import seed_everything, compute_log_loss, extract_meta_features


def train_predict_xgboost(
    transformer_oof, transformer_test, linear_oof, linear_test, save_submission=True
):
    """
    Trains the Level 2 XGBoost Meta-Learner on combined Expert predictions and Meta-features.
    Generates the final submission file.

    Args:
        transformer_oof (np.array): OOF predictions from Transformer Expert (N_train, 3).
        transformer_test (np.array): Test predictions from Transformer Expert (N_test, 3).
        linear_oof (np.array): OOF predictions from Linear Expert (N_train, 3).
        linear_test (np.array): Test predictions from Linear Expert (N_test, 3).
        save_submission (bool): Whether to save the submission file to disk.

    Returns:
        pd.DataFrame: The submission dataframe containing IDs and predicted probabilities.
    """
    seed_everything(Config.SEED)

    print("Preparing data for Meta-Learner (XGBoost)...")

    # 1. Load Metadata for Labels and Feature Extraction
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 2. Extract Meta-Features (Char Len, Word Count, Punct Density)
    # These functions handle their own caching
    print("Extracting explicit meta-features...")
    train_meta = extract_meta_features(df_train, cache_name="train_final")
    test_meta = extract_meta_features(df_test, cache_name="test_final")

    # 3. Construct Feature Matrices
    # Concatenate: [Transformer Probs (3)] + [Linear Probs (3)] + [Meta Features (3)]

    # Training Matrix
    X_train = np.hstack([transformer_oof, linear_oof, train_meta.values])

    # Test Matrix
    X_test = np.hstack([transformer_test, linear_test, test_meta.values])

    # Target Vector
    y_train = df_train["author"].map(Config.LABEL2ID).values

    print(f"Meta-Learner Input Shape: {X_train.shape}")

    # 4. Internal Split for Early Stopping
    # We split the OOF data to give XGBoost a validation set to monitor convergence
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=Config.SEED, stratify=y_train
    )

    # 5. Initialize and Train XGBoost
    print("Training XGBoost Meta-Learner...")

    # Prepare parameters
    xgb_params = Config.XGB_PARAMS.copy()

    # Instantiate Classifier
    model = xgb.XGBClassifier(**xgb_params)

    # Fit with Early Stopping
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # 6. Evaluation
    # Predict on the internal validation set to check performance
    val_preds = model.predict_proba(X_val)
    val_loss = compute_log_loss(y_val, val_preds)
    print(f"Meta-Learner Internal Validation LogLoss: {val_loss}")

    # 7. Generate Final Test Predictions
    print("Generating final predictions on Test set...")
    final_test_probs = model.predict_proba(X_test)

    # 8. Create Submission DataFrame
    submission = pd.DataFrame(final_test_probs, columns=list(Config.LABEL2ID.keys()))
    # Insert ID column from test metadata
    submission.insert(0, "id", df_test["id"])

    # 9. Save Submission
    if save_submission:
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    return submission
