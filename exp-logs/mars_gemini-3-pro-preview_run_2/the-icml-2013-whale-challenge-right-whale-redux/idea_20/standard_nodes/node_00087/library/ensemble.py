import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import seed_everything


def load_oof_features(config: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and aggregates Out-Of-Fold predictions from base models to create
    the training set for the meta-learner. Reconstructs the full dataset
    prediction vector by mapping fold predictions to their validation indices.
    """
    cache_path = os.path.join(config.WORKING_DIR, "oof_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached OOF features from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Constructing OOF features from base model predictions...")

    # 1. Reconstruct the full labeled dataset order to match StratifiedKFold
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    # Concatenate exactly as done in library.data to ensure index alignment
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    y = full_df["label"].values

    # Initialize feature columns
    feature_cols = []
    for model in config.MODELS:
        for metric in ["best_auc", "best_loss"]:
            feature_cols.append(f"{model}_{metric}")

    # Initialize feature matrix with NaNs
    X = pd.DataFrame(np.nan, index=range(len(full_df)), columns=feature_cols)

    # 2. Iterate over folds to fill OOFs
    # Must use the same seed and shuffle as the training loop
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold, (_, val_idx) in enumerate(skf.split(full_df, y)):
        for model in config.MODELS:
            for metric in ["best_auc", "best_loss"]:
                col_name = f"{model}_{metric}"

                # Expected filename format from training loop
                fname = f"{model}_{metric}_fold_{fold}.npy"
                fpath = os.path.join(config.OOF_DIR, fname)

                if os.path.exists(fpath):
                    preds = np.load(fpath)

                    # Validate length
                    if len(preds) == len(val_idx):
                        X.iloc[val_idx, X.columns.get_loc(col_name)] = preds
                    else:
                        print(
                            f"Warning: Length mismatch for {fname}. "
                            f"Expected {len(val_idx)}, got {len(preds)}. Skipping."
                        )
                else:
                    print(
                        f"Warning: OOF file {fpath} not found. "
                        f"Feature {col_name} will have NaNs for fold {fold}."
                    )

    # Add label column for training
    X["label"] = y

    # Handle missing values if any files were missing (Simple Imputation)
    if X.isnull().values.any():
        print("Warning: NaNs detected in OOF features. Imputing with column means.")
        X = X.fillna(X.mean())

    # Cache the result
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    X.to_parquet(cache_path)
    print(f"OOF features cached to {cache_path}")

    return X


def load_test_features(config: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and aggregates Test predictions from base models.
    Performs Bagging (averaging across folds) for each model/metric configuration
    to produce robust feature vectors for the meta-learner.
    """
    cache_path = os.path.join(config.WORKING_DIR, "test_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached Test features from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Constructing Test features from base model predictions...")

    test_meta = pd.read_csv(config.TEST_CSV)
    num_test = len(test_meta)

    feature_data = {}

    for model in config.MODELS:
        for metric in ["best_auc", "best_loss"]:
            col_name = f"{model}_{metric}"
            fold_preds_list = []

            for fold in range(config.NUM_FOLDS):
                # Expected filename format from inference loop
                fname = f"{model}_{metric}_fold_{fold}.npy"
                fpath = os.path.join(config.PREDS_DIR, fname)

                if os.path.exists(fpath):
                    preds = np.load(fpath)
                    if len(preds) == num_test:
                        fold_preds_list.append(preds)
                    else:
                        print(
                            f"Warning: Length mismatch for {fname}. "
                            f"Expected {num_test}, got {len(preds)}."
                        )
                else:
                    # File might be missing if run didn't complete
                    pass

            if fold_preds_list:
                # Bagging: Average across available folds
                avg_preds = np.mean(fold_preds_list, axis=0)
                feature_data[col_name] = avg_preds
            else:
                print(
                    f"Warning: No predictions found for {col_name}. Filling with 0.5 (neutral)."
                )
                feature_data[col_name] = np.full(num_test, 0.5)

    X_test = pd.DataFrame(feature_data)

    # Cache the result
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    X_test.to_parquet(cache_path)
    print(f"Test features cached to {cache_path}")

    return X_test


def train_meta_learner(
    X: pd.DataFrame, target_col: str = "label", random_state: int = 42
):
    """
    Trains a Logistic Regression meta-learner on the OOF features.
    """
    seed_everything(random_state)

    feature_cols = [c for c in X.columns if c != target_col]
    y = X[target_col].values
    X_features = X[feature_cols].values

    print(
        f"Training Meta-Learner on {len(X)} samples with {len(feature_cols)} features..."
    )

    # Logistic Regression
    # 'liblinear' is a good choice for small datasets / binary classification
    clf = LogisticRegression(solver="liblinear", random_state=random_state)

    # Fit the model
    clf.fit(X_features, y)

    # Evaluate on the training set (OOF predictions are unbiased, so this is a valid check)
    probs = clf.predict_proba(X_features)[:, 1]
    auc = roc_auc_score(y, probs)

    print(f"Meta-Learner OOF AUC: {auc:.8f}")
    print("Feature Coefficients:")
    for name, coef in zip(feature_cols, clf.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    return clf


def generate_submission(config: Config, meta_learner, X_test: pd.DataFrame):
    """
    Generates the final submission file using the trained meta-learner.
    """
    print("Generating final submission...")

    # Ensure we use the values in the correct order
    probs = meta_learner.predict_proba(X_test.values)[:, 1]

    # Load Test Metadata to get clip IDs
    test_meta = pd.read_csv(config.TEST_CSV)

    if len(test_meta) != len(probs):
        raise ValueError(
            f"Length mismatch: Test meta {len(test_meta)} vs Preds {len(probs)}"
        )

    # Create prediction dataframe
    pred_df = pd.DataFrame({"clip": test_meta["clip"], "probability": probs})

    # Load Sample Submission to ensure correct order and format
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION)

    # Merge to enforce sample submission order
    # Left join on 'clip' ensures we keep all rows from sample_sub in order
    submission = sample_sub[["clip"]].merge(pred_df, on="clip", how="left")

    # Fill missing values (should not happen if data is complete)
    submission["probability"] = submission["probability"].fillna(0)

    # Save to submission directory
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {config.SUBMISSION_FILE}")
    print(submission.head())

    return submission
