import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
import warnings
import joblib

# Import from provided library files
from library.utils import set_seed, setup_logger
from library.data_manager import DataManager
from library.embedding_engine import EmbeddingEngine
from library.custom_transformers import build_feature_pipeline
from library.trainer import ModelTrainer
from library.inference import InferenceManager

# Filter warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    DEMO_WORK_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    set_seed(42)

    # Configure logger to be less verbose for the demo
    logging.getLogger("DataManager").setLevel(logging.ERROR)
    logging.getLogger("EmbeddingEngine").setLevel(logging.ERROR)
    logging.getLogger("ModelTrainer").setLevel(logging.ERROR)
    logging.getLogger("InferenceManager").setLevel(logging.ERROR)
    logging.getLogger("CustomTransformers").setLevel(logging.ERROR)

    print("--- Starting Demo Execution ---")

    # ---------------------------------------------------------
    # 2. Data Manager Demo
    # ---------------------------------------------------------
    print("\n[1] Demonstrating DataManager...")
    dm = DataManager(cache_dir=DEMO_WORK_DIR)

    # Load data (this reads metadata and raw jsons)
    # We force processing raw data to verify logic
    train_df, val_df, test_df = dm.load_dataset(load_cached_data=False)

    # Verification
    assert not train_df.empty, "Train DataFrame is empty"
    assert "text_combined" in train_df.columns, "Missing text column"
    assert "requester_received_pizza" in train_df.columns, "Missing target column"

    print(f"    Loaded Train shape: {train_df.shape}")
    print(f"    Loaded Val shape: {val_df.shape}")
    print(f"    Loaded Test shape: {test_df.shape}")

    # OPTIMIZATION: Subsample data for speed in subsequent steps
    # We will use a tiny subset to make embedding generation and grid search fast
    N_SUB_TRAIN = 50
    N_SUB_TEST = 10

    train_subset = train_df.head(N_SUB_TRAIN).copy()
    test_subset = test_df.head(N_SUB_TEST).copy()

    print(f"    Subsampled Train for demo: {train_subset.shape}")

    # ---------------------------------------------------------
    # 3. Embedding Engine Demo
    # ---------------------------------------------------------
    print("\n[2] Demonstrating EmbeddingEngine...")
    ee = EmbeddingEngine(cache_dir=DEMO_WORK_DIR)

    sample_texts = train_subset["text_combined"].tolist()

    # Generate Anchor Embeddings (MiniLM)
    anchor_embs = ee.get_anchor_embeddings(
        sample_texts, "demo_train", load_cached_data=False
    )
    assert anchor_embs.shape == (
        N_SUB_TRAIN,
        384,
    ), f"Unexpected Anchor shape: {anchor_embs.shape}"

    # Generate Aux Embeddings (MPNet)
    aux_embs = ee.get_auxiliary_embeddings(
        sample_texts, "demo_train", load_cached_data=False
    )
    assert aux_embs.shape == (
        N_SUB_TRAIN,
        768,
    ), f"Unexpected Aux shape: {aux_embs.shape}"

    print("    Embedding generation successful. Shapes verified.")

    # ---------------------------------------------------------
    # 4. Custom Transformers & Pipeline Demo
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Feature Pipeline...")

    # Metadata dimensions
    meta_cols = dm.metadata_cols
    n_meta = len(meta_cols)

    # Construct a dummy input matrix [Anchor | Aux | Meta]
    # This simulates what the Trainer does internally
    dummy_meta = np.random.rand(N_SUB_TRAIN, n_meta).astype(np.float32)
    X_dummy = np.hstack([anchor_embs, aux_embs, dummy_meta])

    print(f"    Input Feature Matrix Shape: {X_dummy.shape}")

    # Build Pipeline
    pipeline = build_feature_pipeline(
        anchor_dim=384, aux_dim=768, meta_dim=n_meta, seed=42
    )

    # Fit Transform
    X_transformed = pipeline.fit_transform(X_dummy)

    # Check output
    # View 1 (384) + View 2 (50) + View 3 (10) + View 4 (10) = 454 features
    # Note: View 3 uses GMM(10) which outputs 10 probabilities.
    # View 2 uses PCA(50).
    # View 4 uses RankGauss on 10 meta features.
    # View 1 is 384 raw dims.
    expected_dim = 384 + 50 + 10 + 10

    assert X_transformed.shape == (
        N_SUB_TRAIN,
        expected_dim,
    ), f"Pipeline output shape mismatch. Got {X_transformed.shape}, expected ({N_SUB_TRAIN}, {expected_dim})"

    print("    Pipeline fit_transform successful.")

    # ---------------------------------------------------------
    # 5. Model Trainer Demo
    # ---------------------------------------------------------
    print("\n[4] Demonstrating ModelTrainer...")
    trainer = ModelTrainer(work_dir=DEMO_WORK_DIR)

    # PRE-COMPUTE CACHE FOR SPEED
    # Instead of letting trainer process the full dataset, we manually save the
    # subsampled features to the cache files it expects.

    print("    Pre-computing cache for fast training...")

    # Build Train Features (using the subset we created)
    # Re-use embeddings we already computed
    meta_data_train = train_subset[meta_cols].fillna(0).values.astype(np.float32)
    X_train_full = np.hstack([anchor_embs, aux_embs, meta_data_train])
    y_train_full = train_subset["requester_received_pizza"].values.astype(int)

    # Build Test Features (subset)
    test_texts = test_subset["text_combined"].tolist()
    test_anchor = ee.get_anchor_embeddings(
        test_texts, "demo_test", load_cached_data=False
    )
    test_aux = ee.get_auxiliary_embeddings(
        test_texts, "demo_test", load_cached_data=False
    )
    test_meta = test_subset[meta_cols].fillna(0).values.astype(np.float32)
    X_test_sub = np.hstack([test_anchor, test_aux, test_meta])
    test_ids_sub = test_subset["request_id"].values

    # Save to locations ModelTrainer looks for
    np.save(os.path.join(DEMO_WORK_DIR, "X_train_full.npy"), X_train_full)
    np.save(os.path.join(DEMO_WORK_DIR, "y_train_full.npy"), y_train_full)
    np.save(os.path.join(DEMO_WORK_DIR, "X_test.npy"), X_test_sub)
    np.save(os.path.join(DEMO_WORK_DIR, "test_ids.npy"), test_ids_sub)

    # Now run training with reduced folds
    # The trainer will find the cache and skip full data processing
    print("    Starting training loop (2 folds)...")
    trainer.train_loop(n_folds=2)

    # Verify models were saved
    assert os.path.exists(
        os.path.join(DEMO_WORK_DIR, "models", "model_fold_0.joblib")
    ), "Model fold 0 missing"
    assert os.path.exists(
        os.path.join(DEMO_WORK_DIR, "models", "model_fold_1.joblib")
    ), "Model fold 1 missing"

    # Generate submission (using the cached test set)
    trainer.generate_submission(n_folds=2)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not created by Trainer"
    print("    Trainer run complete.")

    # ---------------------------------------------------------
    # 6. Inference Manager Demo
    # ---------------------------------------------------------
    print("\n[5] Demonstrating InferenceManager...")
    # InferenceManager also looks for X_test.npy in work_dir, which we created above.
    inference = InferenceManager(work_dir=DEMO_WORK_DIR)

    # Run prediction
    # This will load the models trained by ModelTrainer and the X_test cache
    df_pred = inference.predict(load_cached_data=True, n_folds=2)

    # Verification
    assert (
        len(df_pred) == N_SUB_TEST
    ), f"Prediction count mismatch. Got {len(df_pred)}, expected {N_SUB_TEST}"
    assert "request_id" in df_pred.columns
    assert "requester_received_pizza" in df_pred.columns
    assert df_pred["requester_received_pizza"].min() >= 0.0
    assert df_pred["requester_received_pizza"].max() <= 1.0

    print("    Inference successful.")
    print(f"    Sample Predictions:\n{df_pred.head(3)}")

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    main()
