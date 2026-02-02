import os
import numpy as np
import pandas as pd
import torch
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model_trainer as model_trainer
from library.embedding_generator import EmbeddingService


def main():
    # 1. Setup
    print("Initializing Demo Script...")
    utils.set_seed(config.SEED)

    # Define a small subset size for rapid demonstration
    DEMO_SIZE = 50
    print(f"Running in DEMO mode with {DEMO_SIZE} samples.")

    # Define demo-specific paths to avoid conflicts with full-scale run artifacts
    demo_working_dir = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_working_dir, exist_ok=True)

    train_prim_path = os.path.join(demo_working_dir, "train_primary.npy")
    train_aux_path = os.path.join(demo_working_dir, "train_aux.npy")
    test_prim_path = os.path.join(demo_working_dir, "test_primary.npy")
    test_aux_path = os.path.join(demo_working_dir, "test_aux.npy")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # Load processed training data (subset)
    df_train = data_loader.get_processed_data(
        split="train",
        debug_size=DEMO_SIZE,
        load_cached_data=False,  # Force re-load for demo
    )

    # Load processed test data (subset)
    df_test = data_loader.get_processed_data(
        split="test", debug_size=DEMO_SIZE, load_cached_data=False
    )

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # Verify Data Loading
    assert (
        len(df_train) == DEMO_SIZE
    ), f"Expected {DEMO_SIZE} train samples, got {len(df_train)}"
    assert (
        "text_combined" in df_train.columns
    ), "Missing 'text_combined' column in train data"
    assert (
        config.TARGET_COL in df_train.columns
    ), f"Missing target column '{config.TARGET_COL}'"

    # 3. Embedding Generation
    print("\n--- Generating Embeddings ---")
    embedder = EmbeddingService()

    # Generate Primary Embeddings (MiniLM)
    print("Generating Primary Embeddings (Train)...")
    X_prim_train = embedder.get_embeddings(
        texts=df_train["text_combined"],
        model_name=config.PRIMARY_MODEL_NAME,
        cache_path=train_prim_path,
        load_cached_data=False,
    )

    print("Generating Primary Embeddings (Test)...")
    X_prim_test = embedder.get_embeddings(
        texts=df_test["text_combined"],
        model_name=config.PRIMARY_MODEL_NAME,
        cache_path=test_prim_path,
        load_cached_data=False,
    )

    # Generate Auxiliary Embeddings (MPNet)
    print("Generating Auxiliary Embeddings (Train)...")
    X_aux_train = embedder.get_embeddings(
        texts=df_train["text_combined"],
        model_name=config.AUX_MODEL_NAME,
        cache_path=train_aux_path,
        load_cached_data=False,
    )

    print("Generating Auxiliary Embeddings (Test)...")
    X_aux_test = embedder.get_embeddings(
        texts=df_test["text_combined"],
        model_name=config.AUX_MODEL_NAME,
        cache_path=test_aux_path,
        load_cached_data=False,
    )

    # Verify Embedding Shapes
    # MiniLM is 384d, MPNet is 768d
    assert X_prim_train.shape == (
        DEMO_SIZE,
        384,
    ), f"Unexpected Primary shape: {X_prim_train.shape}"
    assert X_aux_train.shape == (
        DEMO_SIZE,
        768,
    ), f"Unexpected Auxiliary shape: {X_aux_train.shape}"

    # 4. Prepare Numerical Metadata and Target
    print("\n--- Preparing Metadata & Target ---")
    # Extract numerical columns defined in config
    X_meta_train = df_train[config.NUMERICAL_COLS].values
    X_meta_test = df_test[config.NUMERICAL_COLS].values

    y_train = df_train[config.TARGET_COL].values

    # Verify Metadata
    assert X_meta_train.shape[1] == len(
        config.NUMERICAL_COLS
    ), "Incorrect number of metadata features"

    # 5. Run Cross-Validation Pipeline
    print("\n--- Running Cross-Validation ---")
    # This uses ADBEFTransformer internally (PCA -> Quantile -> Fusion)
    # and trains Bagging(LogisticRegression)
    trained_pipelines, oof_preds = model_trainer.run_cross_validation(
        X_primary=X_prim_train, X_aux=X_aux_train, X_meta=X_meta_train, y=y_train
    )

    assert len(trained_pipelines) == config.N_FOLDS, "Did not train all folds"
    assert len(oof_preds) == DEMO_SIZE, "OOF predictions shape mismatch"

    # 6. Generate Submission
    print("\n--- Generating Submission ---")
    test_ids = df_test["request_id"].values

    # Temporarily override submission path for demo safety
    original_sub_path = config.SUBMISSION_PATH
    demo_sub_path = os.path.join(demo_working_dir, "demo_submission.csv")
    config.SUBMISSION_PATH = demo_sub_path

    try:
        model_trainer.generate_submission(
            trained_pipelines=trained_pipelines,
            X_primary_test=X_prim_test,
            X_aux_test=X_aux_test,
            X_meta_test=X_meta_test,
            test_ids=test_ids,
        )
    finally:
        # Restore config path (good practice, though script ends here)
        config.SUBMISSION_PATH = original_sub_path

    # 7. Final Verification
    print("\n--- Verifying Output ---")
    assert os.path.exists(demo_sub_path), "Submission file was not created"

    df_sub = pd.read_csv(demo_sub_path)
    print(df_sub.head())

    # Check dimensions
    assert (
        len(df_sub) == DEMO_SIZE
    ), f"Submission has {len(df_sub)} rows, expected {DEMO_SIZE}"
    assert df_sub.shape[1] == 2, "Submission should have 2 columns"
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect columns"

    # Check probabilities
    preds = df_sub["requester_received_pizza"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    main()
