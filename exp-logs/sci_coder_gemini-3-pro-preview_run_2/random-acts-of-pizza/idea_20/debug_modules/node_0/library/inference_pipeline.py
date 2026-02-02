import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import load_dataset
from library.text_processing import generate_embeddings
from library.feature_engineering import (
    SubredditPLSProjector,
    MetadataScaler,
    assemble_feature_matrix,
)


def generate_submission(models_dir, debug=False):
    """
    Generates predictions for the test set by averaging outputs from all fold models.

    Args:
        models_dir (str): Directory containing the trained model artifacts.
        debug (bool): If True, processes a small subset of data for debugging.
    """
    print("Starting Inference Pipeline...")
    print(f"Loading models from: {models_dir}")

    # 1. Load Test Data
    # We load the test split defined in metadata.
    sample_size = 100 if debug else None

    # Load processed dataframe (includes text, subreddits, metadata)
    df_test = load_dataset("test", load_cached_data=True, sample_size=sample_size)

    # Load pre-computed SBERT embeddings (View 1)
    # Note: generate_embeddings handles caching internally
    embeddings_test = generate_embeddings("test", load_cached_data=True)

    # Ensure alignment if debugging (generate_embeddings might return full set if cached)
    if debug and len(embeddings_test) > sample_size:
        embeddings_test = embeddings_test[:sample_size]

    # 2. Extract Raw Feature Views
    # View 2: Subreddits (List of strings)
    subreddits_test = df_test[Config.SUBREDDIT_COL].tolist()

    # View 3: Metadata (Numerical DataFrame/Array)
    metadata_test = df_test[Config.NUMERICAL_COLS].values

    # Identifiers for submission
    request_ids = df_test["request_id"].values

    # 3. Ensemble Inference
    fold_preds = []

    # Iterate through all trained folds defined in Config
    for fold in range(Config.N_FOLDS):
        # Construct paths for artifacts
        pls_path = os.path.join(models_dir, f"pls_fold_{fold}.joblib")
        scaler_path = os.path.join(models_dir, f"scaler_fold_{fold}.joblib")
        clf_path = os.path.join(models_dir, f"clf_fold_{fold}.joblib")

        # Check if artifacts exist
        if not (
            os.path.exists(pls_path)
            and os.path.exists(scaler_path)
            and os.path.exists(clf_path)
        ):
            raise FileNotFoundError(
                f"Artifacts for fold {fold} not found in {models_dir}"
            )

        # Load Fold Artifacts
        # We import the classes from library.feature_engineering to ensure joblib can deserialize them
        pls = joblib.load(pls_path)
        scaler = joblib.load(scaler_path)
        clf = joblib.load(clf_path)

        # Transform Test Features using this fold's transformers
        # View 2: Project Subreddits using PLS
        pls_feat = pls.transform(subreddits_test)

        # View 3: Scale Metadata using RankGauss
        meta_feat = scaler.transform(metadata_test)

        # Assemble Fused Feature Matrix
        # Concatenates: [SBERT Embeddings (384d) | PLS Features (~10d) | Metadata (~10d)]
        X_test = assemble_feature_matrix(embeddings_test, pls_feat, meta_feat)

        # Predict Probabilities (Class 1: Received Pizza)
        preds = clf.predict_proba(X_test)[:, 1]
        fold_preds.append(preds)

    # 4. Aggregate Predictions
    # Average predictions across all folds (Ensemble of Ensembles)
    avg_preds = np.mean(fold_preds, axis=0)

    # 5. Generate Submission File
    df_sub = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": avg_preds}
    )

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print first few rows for verification
    print("\nSubmission Head:")
    print(df_sub.head())
