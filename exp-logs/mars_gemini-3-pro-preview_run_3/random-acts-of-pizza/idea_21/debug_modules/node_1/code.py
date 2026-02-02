import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from library import (
    config,
    utils,
    data_factory,
    feature_manager,
    model_definitions,
    stacking_engine,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed/Demo
    # -------------------------------------------------------------------------
    utils.set_seed(42)
    print("Initializing demonstration...")

    # Define temporary directories for this demo
    DEMO_DIR = "./working/demo_run"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_SUB_DIR = os.path.join(DEMO_DIR, "submission")

    for d in [DEMO_META_DIR, DEMO_CACHE_DIR, DEMO_SUB_DIR]:
        os.makedirs(d, exist_ok=True)

    # Override Config Paths to use our demo directories
    # This is crucial so that StackingTrainer reads the correct (sampled) test IDs
    config.METADATA_DIR = DEMO_META_DIR
    config.TRAIN_PATH = os.path.join(DEMO_META_DIR, "train.parquet")
    config.VAL_PATH = os.path.join(DEMO_META_DIR, "val.parquet")
    config.TEST_PATH = os.path.join(DEMO_META_DIR, "test.parquet")
    config.CACHE_DIR = DEMO_CACHE_DIR
    config.SUBMISSION_DIR = DEMO_SUB_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_SUB_DIR, "submission.csv")

    # Override Model Hyperparameters for fast execution
    print("Overriding hyperparameters for speed...")
    config.RF_PARAMS["n_estimators"] = 10
    config.RF_PARAMS["n_jobs"] = 1  # Reduce overhead for small data
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["n_jobs"] = 1
    config.N_FOLDS = 2  # Minimum for CV

    # -------------------------------------------------------------------------
    # 2. Create Sampled Datasets
    # -------------------------------------------------------------------------
    print("\n--- Creating Sampled Datasets (N=50) ---")
    SAMPLE_SIZE = 50

    # Load original files manually to sample them
    # We use the original paths hardcoded here just to bootstrap the demo data
    orig_train = pd.read_parquet("./metadata/train.parquet")
    orig_val = pd.read_parquet("./metadata/val.parquet")
    orig_test = pd.read_parquet("./metadata/test.parquet")

    # Sample and save to demo metadata location
    orig_train.head(SAMPLE_SIZE).to_parquet(config.TRAIN_PATH, index=False)
    orig_val.head(SAMPLE_SIZE).to_parquet(config.VAL_PATH, index=False)
    orig_test.head(SAMPLE_SIZE).to_parquet(config.TEST_PATH, index=False)

    print(f"Sampled data saved to {DEMO_META_DIR}")

    # -------------------------------------------------------------------------
    # 3. Test Data Cleaning
    # -------------------------------------------------------------------------
    print("\n--- Testing DataCleaner ---")
    cleaner = data_factory.DataCleaner()

    # Load the sampled train data via the factory (which now points to demo paths)
    df_train_raw = data_factory.DataLoader.load_train()
    df_train_clean = cleaner.clean_data(df_train_raw, "train", load_cached_data=False)

    # Verification
    assert len(df_train_clean) == SAMPLE_SIZE
    assert config.HISTORY_COL in df_train_clean.columns
    # Verify list-to-string conversion happened (should be string)
    assert pd.api.types.is_string_dtype(df_train_clean[config.HISTORY_COL])
    print("DataCleaner verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Test Feature Extraction
    # -------------------------------------------------------------------------
    print("\n--- Testing FeatureExtractor ---")
    extractor = feature_manager.FeatureExtractor()

    # Extract features (force recompute to test logic)
    # This will generate embeddings, TF-IDF, and metadata features
    features = extractor.extract_features(load_cached_data=False)

    # Verification of Structure
    for split in ["train", "val", "test"]:
        assert split in features
        assert "lexical" in features[split]
        assert "metadata" in features[split]
        assert "semantic_text" in features[split]

    # Verification of Shapes
    # Lexical should be sparse (N, Vocab)
    assert sp.issparse(features["train"]["lexical"])
    assert features["train"]["lexical"].shape[0] == SAMPLE_SIZE

    # Semantic should be dense (N, Embed_Dim)
    assert isinstance(features["train"]["semantic_text"], np.ndarray)
    assert features["train"]["semantic_text"].shape[0] == SAMPLE_SIZE

    # Metadata should be dense (N, Features)
    assert features["train"]["metadata"].shape[0] == SAMPLE_SIZE

    print(
        f"FeatureExtractor verified. Metadata shape: {features['train']['metadata'].shape}"
    )

    # -------------------------------------------------------------------------
    # 5. Test Model Preparation
    # -------------------------------------------------------------------------
    print("\n--- Testing ModelFactory & Feature Preparation ---")
    # Verify we can prepare features for a specific model type
    model_name = model_definitions.ModelFactory.LEXICAL_BAGGER
    X_prep = model_definitions.ModelFactory.prepare_features(
        model_name, features, "train"
    )

    # Lexical Bagger combines Sparse Lexical + Dense Metadata -> Sparse
    assert sp.issparse(X_prep)
    assert X_prep.shape[0] == SAMPLE_SIZE
    # Width = Lexical Vocab + Metadata Cols
    expected_width = (
        features["train"]["lexical"].shape[1] + features["train"]["metadata"].shape[1]
    )
    assert X_prep.shape[1] == expected_width

    print(f"ModelFactory prepared features for {model_name} with shape {X_prep.shape}")

    # -------------------------------------------------------------------------
    # 6. Test Stacking Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Testing StackingTrainer (Full Run) ---")
    trainer = stacking_engine.StackingTrainer(features)

    # Execute the pipeline:
    # 1. OOF Generation (2-Fold)
    # 2. Meta-Learner Training
    # 3. Retraining Base Models
    # 4. Prediction on Test
    # 5. Submission Generation
    trainer.run()

    # -------------------------------------------------------------------------
    # 7. Verify Submission
    # -------------------------------------------------------------------------
    print("\n--- Verifying Submission Output ---")
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Check columns
    assert config.ID_COL in sub_df.columns
    assert config.TARGET_COL in sub_df.columns

    # Check row count matches sample size
    assert len(sub_df) == SAMPLE_SIZE

    # Check probability range
    probs = sub_df[config.TARGET_COL]
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    # Check IDs match the test set
    expected_ids = pd.read_parquet(config.TEST_PATH)[config.ID_COL].values
    assert np.array_equal(sub_df[config.ID_COL].values, expected_ids)

    print("Submission verified successfully.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
