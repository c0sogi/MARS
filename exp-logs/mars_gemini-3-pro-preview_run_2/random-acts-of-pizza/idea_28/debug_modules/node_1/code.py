import os
import shutil
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.pipeline import Pipeline

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.feature_engine import EmbeddingGenerator
from library.dataset_builder import DatasetBuilder
from library.pipeline_factory import PipelineFactory
from library.trainer import Trainer
from library.predictor import Predictor


def run_demo():
    print("Starting Demo Execution of DRSEF Pipeline...")

    # =========================================================================
    # 1. Setup & Configuration Overrides (Optimize for Speed)
    # =========================================================================

    # Define temporary directories for the demo
    DEMO_WORKING_DIR = "./working/demo_execution"
    DEMO_METADATA_DIR = os.path.join(DEMO_WORKING_DIR, "metadata")

    # Clean up previous demo runs if they exist
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)

    print(f"Created temporary working directory: {DEMO_WORKING_DIR}")

    # Create subset metadata to speed up execution (20 samples for train, 10 for val, 10 for test)
    # We read the original metadata, slice it, and save it to the demo folder.
    # The 'sample_index' ensures we pull the correct corresponding raw JSON entries.
    original_train = pd.read_csv(Config.TRAIN_META_PATH)
    original_val = pd.read_csv(Config.VAL_META_PATH)
    original_test = pd.read_csv(Config.TEST_META_PATH)

    subset_size = 20
    demo_train_path = os.path.join(DEMO_METADATA_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_METADATA_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_METADATA_DIR, "test.csv")

    original_train.head(subset_size).to_csv(demo_train_path, index=False)
    original_val.head(subset_size // 2).to_csv(demo_val_path, index=False)
    original_test.head(subset_size // 2).to_csv(demo_test_path, index=False)

    print("Created subset metadata files.")

    # Override Config attributes globally
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.METADATA_DIR = DEMO_METADATA_DIR
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    # Update derived paths in Config (since they were initialized at import time)
    Config.TRAIN_EMBEDDINGS_HIGH_RES = os.path.join(
        DEMO_WORKING_DIR, "train_embeddings_high_res.npy"
    )
    Config.TEST_EMBEDDINGS_HIGH_RES = os.path.join(
        DEMO_WORKING_DIR, "test_embeddings_high_res.npy"
    )
    Config.TRAIN_EMBEDDINGS_LOW_RES = os.path.join(
        DEMO_WORKING_DIR, "train_embeddings_low_res.npy"
    )
    Config.TEST_EMBEDDINGS_LOW_RES = os.path.join(
        DEMO_WORKING_DIR, "test_embeddings_low_res.npy"
    )
    Config.TRAIN_METADATA_PATH = os.path.join(
        DEMO_WORKING_DIR, "train_metadata.parquet"
    )
    Config.TEST_METADATA_PATH = os.path.join(DEMO_WORKING_DIR, "test_metadata.parquet")
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load
    Config.N_FOLDS = 2
    Config.N_ESTIMATORS = 2  # Bagging estimators
    Config.PCA_DIMS = 5  # Ensure n_components < n_samples for demo
    # Simplify Grid Search to a single point
    Config.PARAM_GRID = {"C": [1.0], "solver": ["lbfgs"], "max_iter": [100]}

    # Set seed for reproducibility
    set_seed(Config.RANDOM_SEED)

    # Suppress verbose logging for the demo
    logging.getLogger("data_loader").setLevel(logging.WARNING)
    logging.getLogger("feature_engine").setLevel(logging.WARNING)
    logging.getLogger("dataset_builder").setLevel(logging.WARNING)
    logging.getLogger("pipeline_factory").setLevel(logging.WARNING)
    logging.getLogger("trainer").setLevel(
        logging.INFO
    )  # Keep trainer info to see progress
    logging.getLogger("predictor").setLevel(logging.INFO)

    # =========================================================================
    # 2. Test Data Loading
    # =========================================================================
    print("\n[1/6] Testing Data Loader...")
    loader = DataLoader()
    # Force reload from raw (using our new subset metadata)
    train_df, val_df, test_df = loader.load_data(load_cached_data=False)

    # Verification
    assert (
        len(train_df) == subset_size
    ), f"Expected {subset_size} train samples, got {len(train_df)}"
    assert (
        len(val_df) == subset_size // 2
    ), f"Expected {subset_size // 2} val samples, got {len(val_df)}"
    assert (
        "text_combined" in train_df.columns
    ), "Missing 'text_combined' column in processed data"
    print("Data Loader verification passed.")

    # =========================================================================
    # 3. Test Feature Engineering (Embedding Generation)
    # =========================================================================
    print("\n[2/6] Testing Embedding Generator...")
    embedder = EmbeddingGenerator()
    # This will download/load models and generate embeddings for the small subset
    embeddings = embedder.generate_embeddings(
        train_df, val_df, test_df, load_cached_data=False
    )
    train_high, val_high, test_high, train_low, val_low, test_low = embeddings

    # Verification
    # MiniLM-L6-v2 has dim 384
    assert train_high.shape == (
        subset_size,
        384,
    ), f"Unexpected High-Res shape: {train_high.shape}"
    # MPNet-base-v2 has dim 768
    assert train_low.shape == (
        subset_size,
        768,
    ), f"Unexpected Low-Res shape: {train_low.shape}"
    print("Embedding Generator verification passed.")

    # =========================================================================
    # 4. Test Dataset Builder
    # =========================================================================
    print("\n[3/6] Testing Dataset Builder...")
    builder = DatasetBuilder()
    # We rely on the cache created by previous steps or rebuild if needed.
    # Since we just generated embeddings in memory but the builder loads from disk/memory logic,
    # let's call build_datasets with load_cached_data=True (it will find the embeddings we just saved to disk in step 2)
    X_train, y_train, X_val, y_val, X_test, test_ids = builder.build_datasets(
        load_cached_data=True
    )

    # Verification
    # Features = Metadata cols + 384 (MiniLM) + 768 (MPNet)
    n_meta = len(Config.METADATA_COLS)
    expected_cols = n_meta + 384 + 768

    assert X_train.shape == (
        subset_size,
        expected_cols,
    ), f"Expected X_train shape {(subset_size, expected_cols)}, got {X_train.shape}"
    assert len(y_train) == subset_size
    assert len(test_ids) == subset_size // 2
    print("Dataset Builder verification passed.")

    # =========================================================================
    # 5. Test Pipeline Factory
    # =========================================================================
    print("\n[4/6] Testing Pipeline Factory...")
    dummy_params = {"C": 1.0}
    pipeline = PipelineFactory.create_pipeline(dummy_params)

    # Verification
    assert isinstance(pipeline, Pipeline), "Factory did not return a Pipeline object"
    assert "preprocessor" in pipeline.named_steps, "Pipeline missing preprocessor step"
    assert "classifier" in pipeline.named_steps, "Pipeline missing classifier step"
    print("Pipeline Factory verification passed.")

    # =========================================================================
    # 6. Test Trainer (Cross-Validation)
    # =========================================================================
    print("\n[5/6] Testing Trainer...")
    trainer = Trainer()
    # Run CV loop
    fold_scores = trainer.run_cross_validation(load_cached_data=True)

    # Verification
    assert (
        len(fold_scores) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} scores, got {len(fold_scores)}"
    models_path = os.path.join(Config.WORKING_DIR, "models")
    saved_models = os.listdir(models_path)
    assert len(saved_models) >= Config.N_FOLDS, "Not all fold models were saved"

    # Check if submission was generated by Trainer
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Trainer did not generate submission file"
    print("Trainer verification passed.")

    # =========================================================================
    # 7. Test Predictor (Inference)
    # =========================================================================
    print("\n[6/6] Testing Predictor...")
    # Delete the submission generated by trainer to verify Predictor generates it anew
    os.remove(Config.SUBMISSION_PATH)

    predictor = Predictor()
    predictor.generate_submission(load_cached_data=True)

    # Verification
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Predictor failed to generate submission file"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect"
    assert (
        len(df_sub) == subset_size // 2
    ), f"Submission length mismatch. Expected {subset_size // 2}, got {len(df_sub)}"

    # Check probabilities are valid
    assert df_sub["requester_received_pizza"].min() >= 0.0, "Probabilities < 0 found"
    assert df_sub["requester_received_pizza"].max() <= 1.0, "Probabilities > 1 found"

    print("Predictor verification passed.")

    print("\nAll demo steps completed successfully!")
    print(f"Final submission located at: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_demo()
