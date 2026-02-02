import os
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.pipeline import Pipeline
from unittest.mock import patch

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.embedding_engine import EmbeddingEngine
from library.oasf_transformers import OASFPreprocessor
from library.model_builder import get_model_pipeline
from library.engine import Engine


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up Demo Configuration...")

    # Modify Config for Demo execution to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load for demo
    Config.N_SPLITS = 2  # Only 2 folds
    Config.N_ESTIMATORS = 2  # Only 2 base estimators for Bagging
    Config.PARAM_GRID = {"C": [1.0]}  # Single hyperparameter to skip grid search
    Config.PCA_COMPONENTS = 10  # Reduced components to fit small demo sample size

    # Ensure clean directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    set_seed(Config.RANDOM_SEED)

    # Setup Logger
    logger = setup_logger(
        "demo", os.path.join(Config.WORKING_DIR, "demo_execution.log")
    )
    logger.info("Starting Demo Execution")

    # ==========================================
    # 2. Data Loader Demonstration
    # ==========================================
    print("\n>>> Demonstrating DataLoader...")
    data_loader = DataLoader()

    # Load a small sample of training data
    # We will use this subset for the rest of the demo
    df_train_full = data_loader.load_dataset("train", load_cached_data=False)

    # Validate Data Loading
    assert isinstance(df_train_full, pd.DataFrame)
    assert Config.ID_COL in df_train_full.columns
    assert Config.TARGET_COL in df_train_full.columns
    assert "text_combined" in df_train_full.columns

    # Inspect Metadata Columns
    for col in Config.METADATA_COLS:
        assert col in df_train_full.columns, f"Missing metadata column: {col}"

    print(f"Successfully loaded {len(df_train_full)} training records.")
    print(f"Sample text: {df_train_full['text_combined'].iloc[0][:50]}...")

    # ==========================================
    # 3. Embedding Engine Demonstration
    # ==========================================
    print("\n>>> Demonstrating EmbeddingEngine...")
    embedder = EmbeddingEngine()

    # Select a tiny subset of texts for embedding demonstration
    demo_texts = df_train_full["text_combined"].iloc[:5].tolist()

    # Generate Anchor Embeddings (MiniLM)
    print(f"Generating Anchor embeddings for {len(demo_texts)} samples...")
    emb_anchor = embedder.get_anchor_embeddings(
        demo_texts, "demo_train", load_cached_data=False
    )

    # Generate Aux Embeddings (MPNet)
    print(f"Generating Aux embeddings for {len(demo_texts)} samples...")
    emb_aux = embedder.get_aux_embeddings(
        demo_texts, "demo_train", load_cached_data=False
    )

    # Validate Shapes
    assert emb_anchor.shape == (
        5,
        Config.ANCHOR_DIM,
    ), f"Anchor shape mismatch: {emb_anchor.shape}"
    assert emb_aux.shape == (5, Config.AUX_DIM), f"Aux shape mismatch: {emb_aux.shape}"

    print("Embedding generation successful.")

    # ==========================================
    # 4. OASF Preprocessor Demonstration
    # ==========================================
    print("\n>>> Demonstrating OASFPreprocessor Logic...")

    # Create synthetic data to verify OASF logic without relying on heavy embedding computation
    n_samples = 20
    X_anchor_dummy = np.random.rand(n_samples, Config.ANCHOR_DIM)
    X_aux_dummy = np.random.rand(n_samples, Config.AUX_DIM)
    X_meta_dummy = np.random.rand(n_samples, len(Config.METADATA_COLS))

    # Concatenate as expected by the transformer
    X_concat = np.hstack([X_anchor_dummy, X_aux_dummy, X_meta_dummy])

    preprocessor = OASFPreprocessor(
        anchor_dim=Config.ANCHOR_DIM,
        aux_dim=Config.AUX_DIM,
        pca_components=10,  # Reduced for demo
        random_state=42,
    )

    # Fit and Transform
    X_trans = preprocessor.fit_transform(X_concat)

    # Expected Output Dimension:
    # Anchor (384) + Residuals (10) + Meta (len(METADATA_COLS))
    expected_dim = Config.ANCHOR_DIM + 10 + len(Config.METADATA_COLS)

    assert X_trans.shape == (
        n_samples,
        expected_dim,
    ), f"OASF Output shape {X_trans.shape} != Expected ({n_samples}, {expected_dim})"

    # Verify Normalization of Anchor View (First 384 columns)
    anchor_view = X_trans[:, : Config.ANCHOR_DIM]
    norms = np.linalg.norm(anchor_view, axis=1)
    assert np.allclose(norms, 1.0), "Anchor view in OASF output is not L2 normalized!"

    print("OASF Preprocessor logic verified.")

    # ==========================================
    # 5. Model Pipeline Demonstration
    # ==========================================
    print("\n>>> Demonstrating Model Pipeline Construction...")

    pipeline, param_grid = get_model_pipeline(n_estimators=2, random_state=42)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["preprocessor"], OASFPreprocessor)

    print("Pipeline constructed successfully.")

    # ==========================================
    # 6. Engine Integration (End-to-End Subset Run)
    # ==========================================
    print("\n>>> Demonstrating Engine (End-to-End with Subset)...")

    # To make the engine run fast, we will monkey-patch the data_loader.load_dataset method
    # to return only a small subset (e.g., 50 samples) of the data.

    SUBSET_SIZE = 50

    original_load = DataLoader.load_dataset

    def mocked_load_dataset(self, split, load_cached_data=True):
        # Call original to get full df (or cached)
        df = original_load(self, split, load_cached_data)
        # Return subset
        print(f"   [Mock] Slicing {split} dataset to {SUBSET_SIZE} samples for speed.")
        return df.head(SUBSET_SIZE)

    # Patch the method on the class
    with patch.object(
        DataLoader, "load_dataset", side_effect=mocked_load_dataset, autospec=True
    ):

        engine = Engine()

        # A. Run Cross Validation
        print("Running Cross Validation on subset...")
        engine.run_cross_validation(load_cached_data=False)

        # Verify models were saved
        assert len(engine.models) == Config.N_SPLITS
        for i in range(Config.N_SPLITS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{i}.joblib")
            assert os.path.exists(model_path), f"Model file missing: {model_path}"

        # Verify OOF predictions
        oof_path = os.path.join(Config.WORKING_DIR, "oof_preds.npy")
        assert os.path.exists(oof_path)
        oof_preds = np.load(oof_path)
        # Since we mocked the data loader inside the engine, the internal y will be size SUBSET_SIZE * 2 (train+val merged)
        # Note: Engine merges train and val splits. If both are mocked to 50, total is 100.
        assert (
            len(oof_preds) == SUBSET_SIZE * 2
        ), f"OOF preds length {len(oof_preds)} mismatch."

        # B. Run Inference
        print("Running Inference on Test subset...")
        engine.predict_test(load_cached_data=False)

        # Verify Submission
        assert os.path.exists(Config.SUBMISSION_PATH)
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)

        assert len(df_sub) == SUBSET_SIZE
        assert Config.ID_COL in df_sub.columns
        assert Config.TARGET_COL in df_sub.columns

        # Verify Probabilities
        probs = df_sub[Config.TARGET_COL]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Predictions out of probability range [0, 1]"

        print(f"Submission generated with {len(df_sub)} rows.")

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    run_demo()
