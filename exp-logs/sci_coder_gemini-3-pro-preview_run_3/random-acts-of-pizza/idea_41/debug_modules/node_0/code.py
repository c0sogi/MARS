import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
import sys

# Import library modules
from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import load_and_preprocess_data, DataPreprocessor
from library.feature_engineering import (
    LatentUserClusterer,
    SparseFeaturizer,
    TextEmbedder,
    MetadataSelector,
    get_features,
)
from library.model_factory import get_base_models
from library.stacking_manager import StackingPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    logger = get_logger("DemoScript")
    logger("Starting demonstration script...")

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    logger("Overriding Config parameters for fast execution...")

    # Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # Reduce Random Forest Estimators
    Config.RF_LEXICAL_PARAMS["n_estimators"] = 10
    Config.RF_COMMUNITY_PARAMS["n_estimators"] = 10
    Config.RF_SEMANTIC_PARAMS["n_estimators"] = 10

    # Reduce Boosting Estimators
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["n_estimators"] = 10

    # Reduce Dimensionality for Feature Engineering
    Config.SUBREDDIT_TFIDF_PARAMS["max_features"] = 50
    Config.TEXT_TFIDF_PARAMS["max_features"] = 50
    Config.SVD_COMPONENTS = 5
    Config.N_CLUSTERS = 3

    # Ensure Working Directory is clean for fresh run (optional, but good for demo)
    # We keep the directory but remove specific cache files to force re-computation logic
    # where appropriate for demonstration.
    if os.path.exists(Config.WORKING_DIR):
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".parquet") or f.endswith(".npy") or f.endswith(".npz"):
                try:
                    os.remove(os.path.join(Config.WORKING_DIR, f))
                except OSError:
                    pass

    set_seed(Config.RANDOM_STATE)

    # =========================================================================
    # 2. Data Loading and Preprocessing Demonstration
    # =========================================================================
    logger("--- Demo: Data Loading and Preprocessing ---")

    # Load data (force reload to demonstrate logic)
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=False)

    # Verification
    assert (
        "text_concat" in train_df.columns
    ), "Preprocessing failed: 'text_concat' column missing."
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train."
    assert (
        "requester_received_pizza" not in test_df.columns
    ), "Target column leaked into test."

    # Check if leakage columns were removed
    leakage_col = "number_of_downvotes_of_request_at_retrieval"
    assert (
        leakage_col not in train_df.columns
    ), f"Leakage column {leakage_col} not removed."

    logger(
        f"Data loaded successfully. Train shape: {train_df.shape}, Val shape: {val_df.shape}"
    )

    # =========================================================================
    # 3. Feature Engineering Component Demonstration
    # =========================================================================
    logger("--- Demo: Feature Engineering Components ---")

    # Subset for speed testing components
    subset_df = train_df.head(50).copy()
    subreddit_series = subset_df[Config.SUBREDDIT_COL]
    text_series = subset_df["text_concat"]

    # A. Latent User Clusterer
    logger("Testing LatentUserClusterer...")
    clusterer = LatentUserClusterer()
    clusterer.fit(subreddit_series)
    latent_feats = clusterer.transform(subreddit_series)

    assert latent_feats.shape == (
        50,
        Config.N_CLUSTERS,
    ), f"Latent features shape mismatch. Expected (50, {Config.N_CLUSTERS}), got {latent_feats.shape}"

    # B. Sparse Featurizer
    logger("Testing SparseFeaturizer...")
    featurizer = SparseFeaturizer()
    featurizer.fit(text_series, subreddit_series)
    sparse_feats = featurizer.transform(text_series, subreddit_series)

    assert "lexical" in sparse_feats and "behavioral" in sparse_feats
    assert sp.issparse(sparse_feats["lexical"]), "Lexical features should be sparse."
    assert sp.issparse(
        sparse_feats["behavioral"]
    ), "Behavioral features should be sparse."

    # C. Text Embedder
    logger("Testing TextEmbedder (MiniLM)...")
    embedder = TextEmbedder()
    # Run on very small batch to verify model loading and inference
    embeddings = embedder.transform(text_series.head(5))

    assert embeddings.shape == (
        5,
        Config.EMBEDDING_DIM,
    ), f"Embedding shape mismatch. Expected (5, {Config.EMBEDDING_DIM}), got {embeddings.shape}"

    # D. Metadata Selector
    logger("Testing MetadataSelector...")
    selector = MetadataSelector()
    selector.fit(subset_df, latent_feats)
    meta_feats = selector.transform(subset_df, latent_feats)

    expected_meta_dim = len(Config.NUMERICAL_COLS) + Config.N_CLUSTERS
    assert meta_feats.shape == (
        50,
        expected_meta_dim,
    ), f"Metadata shape mismatch. Expected (50, {expected_meta_dim}), got {meta_feats.shape}"

    # =========================================================================
    # 4. Model Factory Verification
    # =========================================================================
    logger("--- Demo: Model Factory ---")
    base_models = get_base_models()
    assert "lexical_bagger" in base_models
    assert "semantic_booster" in base_models
    assert base_models["lexical_bagger"]["sparse"] is True
    assert base_models["semantic_booster"]["sparse"] is False

    logger("Model factory configuration verified.")

    # =========================================================================
    # 5. Full Stacking Pipeline Execution
    # =========================================================================
    logger("--- Demo: Full Stacking Pipeline Execution ---")

    pipeline = StackingPipeline()

    # Run the pipeline
    # This will:
    # 1. Load data
    # 2. Compute embeddings (cached in memory)
    # 3. Run 2-Fold CV (Level 1)
    # 4. Train Meta-Learner (Level 2)
    # 5. Retrain on full train
    # 6. Predict on test and save submission
    pipeline.run()

    # =========================================================================
    # 6. Final Validation
    # =========================================================================
    logger("--- Demo: Final Validation ---")

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    submission_df = pd.read_csv(submission_path)
    assert list(submission_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns are incorrect."

    assert len(submission_df) == len(
        test_df
    ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(submission_df)}."

    # Check probability range
    probs = submission_df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range."

    logger(f"Submission generated successfully at: {submission_path}")
    logger("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
