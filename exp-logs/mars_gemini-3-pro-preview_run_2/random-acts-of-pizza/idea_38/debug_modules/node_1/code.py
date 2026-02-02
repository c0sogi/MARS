import os
import sys
import numpy as np
import pandas as pd
import joblib
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_and_process_data
from library.embedding_manager import EmbeddingService
from library.pipeline_builder import PipelineBuilder
from library.trainer import Trainer
from library.inference_engine import InferenceService


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up configuration for fast demonstration...")

    # Monkey-patch Config for speed optimization
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample size for speed
    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.N_ESTIMATORS = 2  # Reduce bagging estimators
    Config.MAX_ITER = 100  # Reduce logistic regression iterations

    # Restrict Grid Search to a single parameter combination to avoid time consumption
    Config.GRID_SEARCH_PARAMS = {
        "estimator__C": [1.0],
        "estimator__class_weight": ["balanced"],
    }

    # Ensure clean state for working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    logger = setup_logger("demo_script")
    logger.info("Configuration patched for speed.")

    # ==========================================
    # 2. Test Data Loader
    # ==========================================
    logger.info(">>> Testing Data Loader...")

    # Force reload from raw data (ignore cache initially to prove processing works)
    df_train, df_val, df_test = load_and_process_data(
        debug=True, load_cached_data=False
    )

    # Assertions
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(df_train)}"
    assert len(df_val) == Config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(df_val)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(df_test)}"

    # Check for required columns
    expected_cols = Config.TEXT_COLS + Config.NUMERICAL_COLS + [Config.ID_COL]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing column {col} in train"
        assert col in df_test.columns, f"Missing column {col} in test"

    assert Config.TARGET_COL in df_train.columns, "Target column missing in train"
    assert (
        Config.TARGET_COL not in df_test.columns
    ), "Target column present in test (should be absent)"

    logger.info("Data Loader verified successfully.")

    # ==========================================
    # 3. Test Embedding Manager
    # ==========================================
    logger.info(">>> Testing Embedding Manager...")

    emb_service = EmbeddingService()

    # Generate Anchor embeddings (MiniLM -> 384 dim)
    # We use the 'train' split name, which maps to Config.TRAIN_EMB_ANCHOR
    anchor_emb = emb_service.get_embeddings(
        df_train, "train", "anchor", load_cached_data=False
    )

    # Generate Aux embeddings (MPNet -> 768 dim)
    aux_emb = emb_service.get_embeddings(
        df_train, "train", "aux", load_cached_data=False
    )

    # Assertions
    assert anchor_emb.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        384,
    ), f"Anchor emb shape mismatch: {anchor_emb.shape}"
    assert aux_emb.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        768,
    ), f"Aux emb shape mismatch: {aux_emb.shape}"

    # Verify file creation
    assert os.path.exists(Config.TRAIN_EMB_ANCHOR), "Anchor embedding file not saved"
    assert os.path.exists(Config.TRAIN_EMB_AUX), "Aux embedding file not saved"

    logger.info("Embedding Manager verified successfully.")

    # ==========================================
    # 4. Test Pipeline Builder (Unit Test)
    # ==========================================
    logger.info(">>> Testing Pipeline Builder...")

    # Construct dummy inputs matching the embedding shapes
    anchor_cols = [f"anchor_{i}" for i in range(384)]
    aux_cols = [f"aux_{i}" for i in range(768)]

    # Create a dummy dataframe combining metadata and embeddings
    # We reuse df_train's metadata columns
    meta_cols = pd.Index(Config.NUMERICAL_COLS + Config.DISCRETE_COLS).unique().tolist()
    X_dummy_meta = df_train[meta_cols].copy()

    # Create dummy embedding dataframes
    X_dummy_anchor = pd.DataFrame(
        np.random.randn(Config.DEBUG_SAMPLE_SIZE, 384),
        columns=anchor_cols,
        index=df_train.index,
    )
    X_dummy_aux = pd.DataFrame(
        np.random.randn(Config.DEBUG_SAMPLE_SIZE, 768),
        columns=aux_cols,
        index=df_train.index,
    )

    X_combined = pd.concat([X_dummy_meta, X_dummy_anchor, X_dummy_aux], axis=1)
    y_dummy = df_train[Config.TARGET_COL].values

    # Build Pipeline
    pipeline = PipelineBuilder.build_daadbe_pipeline(
        anchor_cols=anchor_cols,
        aux_cols=aux_cols,
        continuous_cols=Config.NUMERICAL_COLS,
        discrete_cols=Config.DISCRETE_COLS,
        pca_components=10,  # Reduced for test
        n_bins=3,
        n_estimators=2,
    )

    # Fit and Predict
    pipeline.fit(X_combined, y_dummy)
    preds = pipeline.predict_proba(X_combined)[:, 1]

    # Assertions
    assert len(preds) == Config.DEBUG_SAMPLE_SIZE
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of range"

    logger.info("Pipeline Builder verified successfully.")

    # ==========================================
    # 5. Test Trainer (Integration Test)
    # ==========================================
    logger.info(">>> Testing Trainer...")

    trainer = Trainer()

    # Run training loop (Debug mode uses subsampled data)
    # This will trigger data loading, embedding generation, and CV training
    trainer.train(debug=True, load_cached_data=True)

    # Verify output models
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.joblib")
        assert os.path.exists(model_path), f"Model for fold {fold} was not created"

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    logger.info("Trainer verified successfully.")

    # ==========================================
    # 6. Test Inference Engine (Integration Test)
    # ==========================================
    logger.info(">>> Testing Inference Engine...")

    inference_service = InferenceService()

    # Run inference
    # This loads test data, computes embeddings, loads models, and generates submission
    inference_service.predict(debug=True, load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows mismatch: {len(df_sub)}"
    assert list(df_sub.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch"

    # Check values
    assert df_sub[Config.TARGET_COL].min() >= 0.0
    assert df_sub[Config.TARGET_COL].max() <= 1.0

    logger.info("Inference Engine verified successfully.")

    print("\n>>> All demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
