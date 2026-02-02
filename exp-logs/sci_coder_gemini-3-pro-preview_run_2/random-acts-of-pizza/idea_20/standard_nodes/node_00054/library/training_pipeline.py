import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_loader import load_dataset
from library.text_processing import generate_embeddings
from library.feature_engineering import (
    SubredditPLSProjector,
    MetadataScaler,
    assemble_feature_matrix,
)
from library.model_factory import create_classifier


def run_stratified_cv(debug=False):
    """
    Performs 5-fold stratified cross-validation with nested grid search.
    Trains models, saves artifacts, and generates submission.

    Args:
        debug (bool): If True, uses a small subset of data for rapid testing.
    """
    print("Starting Stratified CV Pipeline...")

    # 1. Load Training Data
    # We use the 'train' split defined in metadata for CV.
    sample_size = 100 if debug else None

    # Load processed dataframe (includes text, subreddits, metadata, target)
    df_train = load_dataset("train", load_cached_data=True, sample_size=sample_size)

    # Load pre-computed SBERT embeddings (View 1)
    embeddings_train = generate_embeddings("train", load_cached_data=True)

    if debug:
        embeddings_train = embeddings_train[:sample_size]

    # Extract Feature Views for Indexing
    # View 2: Subreddits (List of strings)
    subreddits_train = df_train[Config.SUBREDDIT_COL].tolist()
    # View 3: Metadata (Numerical DataFrame/Array)
    metadata_train = df_train[Config.NUMERICAL_COLS].values
    # Target
    y = df_train[Config.TARGET_COL].values

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    # Create directory for saving fold-specific models
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\nProcessing Fold {fold + 1}/{Config.N_FOLDS}...")

        # Split Data Indices
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Slice Feature Views
        # View 1: SBERT
        emb_train_fold = embeddings_train[train_idx]
        emb_val_fold = embeddings_train[val_idx]

        # View 2: Subreddits
        sub_train_fold = [subreddits_train[i] for i in train_idx]
        sub_val_fold = [subreddits_train[i] for i in val_idx]

        # View 3: Metadata
        meta_train_fold = metadata_train[train_idx]
        meta_val_fold = metadata_train[val_idx]

        # Variables to track best hyperparameters for this fold
        best_fold_auc = -1.0
        best_fold_artifacts = {}
        best_fold_params = {}

        # Nested Grid Search
        # Outer Loop: PLS Components (Feature Engineering Hyperparameter)
        for n_pls in Config.PLS_N_COMPONENTS_GRID:
            # Initialize and Fit PLS Projector (Supervised)
            pls = SubredditPLSProjector(n_components=n_pls)
            try:
                pls.fit(sub_train_fold, y_train_fold)
            except Exception as e:
                print(f"  PLS fit failed for n={n_pls}: {e}")
                continue

            # Transform Subreddit Features
            pls_feat_train = pls.transform(sub_train_fold)
            pls_feat_val = pls.transform(sub_val_fold)

            # Initialize and Fit Metadata Scaler (RankGauss)
            scaler = MetadataScaler()
            scaler.fit(meta_train_fold)

            # Transform Metadata Features
            meta_feat_train = scaler.transform(meta_train_fold)
            meta_feat_val = scaler.transform(meta_val_fold)

            # Assemble Fused Feature Matrices
            X_train_fold = assemble_feature_matrix(
                emb_train_fold, pls_feat_train, meta_feat_train
            )
            X_val_fold = assemble_feature_matrix(
                emb_val_fold, pls_feat_val, meta_feat_val
            )

            # Inner Loop: Classifier Hyperparameters
            for C in Config.LR_C_GRID:
                for cw in Config.LR_CLASS_WEIGHTS:
                    # Create Bagged Logistic Regression Ensemble
                    clf = create_classifier(
                        C=C,
                        class_weight=cw,
                        random_state=Config.SEED,
                        n_jobs=Config.N_JOBS,
                    )

                    # Train
                    clf.fit(X_train_fold, y_train_fold)

                    # Validate
                    y_pred = clf.predict_proba(X_val_fold)[:, 1]
                    auc = roc_auc_score(y_val_fold, y_pred)

                    # Track Best Performance
                    if auc > best_fold_auc:
                        best_fold_auc = auc
                        best_fold_params = {"n_pls": n_pls, "C": C, "class_weight": cw}
                        best_fold_artifacts = {"pls": pls, "scaler": scaler, "clf": clf}

        print(f"  Best Fold AUC: {best_fold_auc} with params {best_fold_params}")
        fold_scores.append(best_fold_auc)

        # Save Best Artifacts for this Fold
        joblib.dump(
            best_fold_artifacts["pls"],
            os.path.join(models_dir, f"pls_fold_{fold}.joblib"),
        )
        joblib.dump(
            best_fold_artifacts["scaler"],
            os.path.join(models_dir, f"scaler_fold_{fold}.joblib"),
        )
        joblib.dump(
            best_fold_artifacts["clf"],
            os.path.join(models_dir, f"clf_fold_{fold}.joblib"),
        )

    avg_auc = np.mean(fold_scores)
    print(f"\nAverage CV AUC: {avg_auc}")

    # Generate Submission using the trained fold models
    generate_submission(models_dir, debug=debug)


def generate_submission(models_dir, debug=False):
    """
    Generates predictions for the test set by averaging outputs from all fold models.
    """
    print("\nGenerating Submission...")

    # Load Test Data
    sample_size = 100 if debug else None
    df_test = load_dataset("test", load_cached_data=True, sample_size=sample_size)
    embeddings_test = generate_embeddings("test", load_cached_data=True)

    if debug:
        embeddings_test = embeddings_test[:sample_size]

    # Extract Test Features
    subreddits_test = df_test[Config.SUBREDDIT_COL].tolist()
    metadata_test = df_test[Config.NUMERICAL_COLS].values
    request_ids = df_test["request_id"].values

    fold_preds = []

    # Iterate through all trained folds
    for fold in range(Config.N_FOLDS):
        # Load Fold Artifacts
        pls = joblib.load(os.path.join(models_dir, f"pls_fold_{fold}.joblib"))
        scaler = joblib.load(os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))
        clf = joblib.load(os.path.join(models_dir, f"clf_fold_{fold}.joblib"))

        # Transform Test Features
        pls_feat = pls.transform(subreddits_test)
        meta_feat = scaler.transform(metadata_test)

        # Assemble Test Matrix
        X_test = assemble_feature_matrix(embeddings_test, pls_feat, meta_feat)

        # Predict Probabilities
        preds = clf.predict_proba(X_test)[:, 1]
        fold_preds.append(preds)

    # Average Predictions (Ensemble of Ensembles)
    avg_preds = np.mean(fold_preds, axis=0)

    # Create Submission DataFrame
    df_sub = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": avg_preds}
    )

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
