import os
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset, NUMERIC_COLS
from library.feature_text import SBERTEmbedder
from library.feature_topic import LDATopicExtractor
from library.feature_meta import MetadataScaler
from library.model_builder import create_classifier
from library.engine import ModelEngine


def main():
    # --------------------------------------------------------------------------
    # 0. Setup and Configuration for Demo
    # --------------------------------------------------------------------------
    print(">>> Setting up Demo Configuration...")

    # Override Config for a fast, isolated demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample size for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update paths in Config to point to the demo working directory
    Config.TRAIN_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "train_embeddings.npy"
    )
    Config.VAL_EMBEDDINGS_PATH = os.path.join(Config.WORKING_DIR, "val_embeddings.npy")
    Config.TEST_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "test_embeddings.npy"
    )
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_processed.parquet"
    )

    # Reduce CV folds for speed
    Config.N_FOLDS = 2

    # Ensure directories exist
    Config.ensure_directories()

    # Set seed for reproducibility
    set_seed(42)
    logger = setup_logger("demo_script")

    print("Configuration updated. Working directory:", Config.WORKING_DIR)

    # --------------------------------------------------------------------------
    # 1. Demonstrate Data Loading
    # --------------------------------------------------------------------------
    print("\n>>> Testing Data Loader...")

    # Force reload from raw JSON to demonstrate processing logic
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Assertions
    assert isinstance(df_train, pd.DataFrame)
    assert len(df_train) <= Config.DEBUG_SAMPLE_SIZE
    assert "combined_text" in df_train.columns
    assert "subreddit_list_str" in df_train.columns
    assert "requester_received_pizza" in df_train.columns

    # Check if files were saved
    assert os.path.exists(Config.TRAIN_FEATURES_PATH)
    assert os.path.exists(Config.VAL_FEATURES_PATH)

    print(f"Data Loaded Successfully. Train shape: {df_train.shape}")

    # --------------------------------------------------------------------------
    # 2. Demonstrate Text Embedding (SBERT)
    # --------------------------------------------------------------------------
    print("\n>>> Testing SBERT Embedder...")

    embedder = SBERTEmbedder()

    # Test direct encoding
    sample_texts = ["I need pizza", "Hungry student here"]
    embeddings = embedder.encode(sample_texts)
    assert embeddings.shape == (2, 384)
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)  # Check L2 normalization

    # Test processing and caching with DataFrame
    # We use the loaded df_train
    train_embeddings = embedder.process_and_cache(
        df_train, Config.TRAIN_EMBEDDINGS_PATH, load_cached_data=False
    )

    assert train_embeddings.shape == (len(df_train), 384)
    assert os.path.exists(Config.TRAIN_EMBEDDINGS_PATH)
    print("SBERT Embeddings generated and cached.")

    # --------------------------------------------------------------------------
    # 3. Demonstrate Topic Extraction (LDA)
    # --------------------------------------------------------------------------
    print("\n>>> Testing LDA Topic Extractor...")

    # Use small min_df because our sample size is tiny
    lda = LDATopicExtractor(n_components=3, min_df=1, random_state=42)

    # Fit on training data
    lda.fit(df_train["subreddit_list_str"])

    # Transform validation data
    topic_features = lda.transform(df_val["subreddit_list_str"])

    assert topic_features.shape == (len(df_val), 3)
    # Check for finite values (RankGauss should handle this, but good to verify)
    assert np.all(np.isfinite(topic_features))

    print("LDA Topic Extraction successful.")

    # --------------------------------------------------------------------------
    # 4. Demonstrate Metadata Scaling
    # --------------------------------------------------------------------------
    print("\n>>> Testing Metadata Scaler...")

    scaler = MetadataScaler(random_state=42)

    # Fit on training data
    scaler.fit(df_train)

    # Transform validation data
    meta_features = scaler.transform(df_val)

    assert meta_features.shape == (len(df_val), len(NUMERIC_COLS))
    assert np.all(np.isfinite(meta_features))

    print("Metadata Scaling successful.")

    # --------------------------------------------------------------------------
    # 5. Demonstrate Model Builder
    # --------------------------------------------------------------------------
    print("\n>>> Testing Model Builder...")

    # Create synthetic features for testing model fit
    n_samples = 20
    n_features = 10
    X_synth = np.random.randn(n_samples, n_features)
    y_synth = np.random.randint(0, 2, n_samples)

    clf = create_classifier(C=1.0, n_estimators=5, random_state=42)

    clf.fit(X_synth, y_synth)
    probs = clf.predict_proba(X_synth)[:, 1]

    assert len(probs) == n_samples
    assert np.all((probs >= 0) & (probs <= 1))

    print("Model Builder (Ensemble LR) test successful.")

    # --------------------------------------------------------------------------
    # 6. Demonstrate Full Engine Execution
    # --------------------------------------------------------------------------
    print("\n>>> Testing Full Model Engine (Integration Test)...")

    # The engine orchestrates everything: loading, embedding, fusion, CV, submission
    engine = ModelEngine()

    # We use load_cached_data=True to reuse the embeddings/features we just generated/verified
    # where possible, though the engine logic splits train/val differently (using StratifiedKFold)
    # so it might re-compute some things internally or use the cached files if paths align.
    # Given we set Config paths, it should pick them up.
    engine.run(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)

    # Check submission format
    assert "request_id" in submission_df.columns
    assert "requester_received_pizza" in submission_df.columns
    assert len(submission_df) > 0
    # In debug mode, we slice the test set too, so check length matches debug size
    assert len(submission_df) <= Config.DEBUG_SAMPLE_SIZE

    print(
        f"Engine execution complete. Submission generated at {Config.SUBMISSION_FILE_PATH}"
    )
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
