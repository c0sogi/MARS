import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.ensemble import BaggingClassifier

# Import provided library components
from library.config import Config
from library.utils import set_seed, suppress_warnings, ensure_directory
from library.data_loader import DataLoader
from library.embedding_engine import EmbeddingEngine
from library.feature_engineer import CoherenceFeatureProcessor
from library.model_factory import ModelFactory
from library.execution_manager import ExecutionManager


def demo_data_loader():
    print("\n=== Demo: DataLoader ===")
    loader = DataLoader()

    # Test loading a small sample of the training split
    sample_size = 10
    df_train = loader.load_split("train", sample_size=sample_size)

    print(f"Loaded train sample shape: {df_train.shape}")

    # Assertions
    assert len(df_train) == sample_size, "Sample size mismatch in train split"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train split"
    assert Config.TEXT_COL_TITLE in df_train.columns, "Title column missing"
    assert Config.TEXT_COL_BODY in df_train.columns, "Body column missing"

    # Test loading test split
    df_test = loader.load_split("test", sample_size=sample_size)
    assert (
        "requester_received_pizza" not in df_test.columns
    ), "Target column should not be in test split"

    print("DataLoader verification passed.")
    return df_train


def demo_embedding_engine(df_sample):
    print("\n=== Demo: EmbeddingEngine ===")
    embedder = EmbeddingEngine()

    # Generate embeddings for the sample
    # Note: This will download models if not present, which might take a moment on the first run
    # We force load_cached_data=False to verify computation logic
    title_emb, body_emb, global_emb = embedder.generate_train_embeddings(
        df_sample, load_cached_data=False
    )

    print(f"Title Embeddings Shape: {title_emb.shape}")
    print(f"Body Embeddings Shape: {body_emb.shape}")
    print(f"Global Embeddings Shape: {global_emb.shape}")

    # Assertions
    # MiniLM output is 384 dims, MPNet is 768 dims
    assert title_emb.shape == (len(df_sample), 384), "Incorrect Title embedding shape"
    assert body_emb.shape == (len(df_sample), 384), "Incorrect Body embedding shape"
    assert global_emb.shape == (len(df_sample), 768), "Incorrect Global embedding shape"

    print("EmbeddingEngine verification passed.")
    return title_emb, body_emb, global_emb


def demo_feature_processor(title_emb, body_emb, global_emb, df_sample):
    print("\n=== Demo: CoherenceFeatureProcessor ===")
    processor = CoherenceFeatureProcessor()

    # Extract numeric metadata
    X_meta = df_sample[Config.NUMERIC_COLS].fillna(0).values

    # Fit the processor
    # Note: Config.PCA_COMPONENTS was reduced in setup to handle small sample size
    processor.fit(title_emb, body_emb, global_emb, X_meta)

    # Transform
    X_fused = processor.transform(title_emb, body_emb, global_emb, X_meta)

    print(f"Fused Feature Matrix Shape: {X_fused.shape}")

    # Expected dimensions:
    # 384 (Title) + 384 (Body) + PCA_COMPONENTS (Global) + 1 (Coherence) + N_Meta (RankGauss)
    expected_dim = 384 + 384 + Config.PCA_COMPONENTS + 1 + len(Config.NUMERIC_COLS)
    assert X_fused.shape == (
        len(df_sample),
        expected_dim,
    ), f"Expected {expected_dim} columns, got {X_fused.shape[1]}"

    print("CoherenceFeatureProcessor verification passed.")
    return X_fused


def demo_model_factory(X_train, y_train):
    print("\n=== Demo: ModelFactory ===")
    factory = ModelFactory()

    # Optimize and train
    # Config.PARAM_GRID was simplified in setup to make this fast
    model = factory.optimize_and_train(X_train, y_train)

    print(f"Returned Model Type: {type(model)}")

    # Assertions
    assert isinstance(
        model, BaggingClassifier
    ), "ModelFactory should return a BaggingClassifier"

    # Test prediction
    probs = model.predict_proba(X_train)
    assert probs.shape == (len(X_train), 2), "Predict proba output shape mismatch"

    print("ModelFactory verification passed.")


def demo_execution_manager():
    print("\n=== Demo: ExecutionManager (Full Pipeline) ===")
    manager = ExecutionManager()

    # Run pipeline with a small debug sample size
    # This integrates all steps: Data Load -> Embed -> Feature Eng -> CV Train -> Inference
    debug_size = 20  # Must be enough for StratifiedKFold (at least N_FOLDS * classes)
    manager.run_cv_and_inference(debug_sample_size=debug_size, load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")
    assert (
        len(df_sub) == debug_size
    ), f"Submission should have {debug_size} rows in debug mode"
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns

    print("ExecutionManager verification passed.")


if __name__ == "__main__":
    # 1. Setup Environment
    suppress_warnings()
    set_seed(42)

    # 2. Configure for Demo/Speed
    print("Configuring environment for fast demonstration...")

    # Set a separate working directory for the demo to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    ensure_directory(Config.SUBMISSION_PATH)

    # Reduce dimensionality and complexity for small sample tests
    Config.PCA_COMPONENTS = 2  # Must be < sample_size (10)
    Config.N_FOLDS = 2  # Minimum for CV
    Config.BAGGING_N_ESTIMATORS = 2

    # Simplify Grid Search to a single iteration
    Config.PARAM_GRID = {
        "estimator__C": [1.0],
        "estimator__class_weight": [None],
        "estimator__solver": ["lbfgs"],
        "estimator__max_iter": [100],
    }

    # Redirect cache paths to demo folder
    Config.CACHE_TRAIN_TITLE_MINILM = os.path.join(
        Config.WORKING_DIR, "train_emb_anchor.npy"
    )
    Config.CACHE_TRAIN_BODY_MINILM = os.path.join(
        Config.WORKING_DIR, "train_emb_aux.npy"
    )
    Config.CACHE_TRAIN_GLOBAL_MPNET = os.path.join(
        Config.WORKING_DIR, "train_emb_global.npy"
    )
    Config.CACHE_TEST_TITLE_MINILM = os.path.join(
        Config.WORKING_DIR, "test_emb_anchor.npy"
    )
    Config.CACHE_TEST_BODY_MINILM = os.path.join(Config.WORKING_DIR, "test_emb_aux.npy")
    Config.CACHE_TEST_GLOBAL_MPNET = os.path.join(
        Config.WORKING_DIR, "test_emb_global.npy"
    )

    # 3. Run Component Demos
    try:
        # Step 1: Data Loading
        df_sample = demo_data_loader()

        # Step 2: Embeddings
        title_emb, body_emb, global_emb = demo_embedding_engine(df_sample)

        # Step 3: Feature Engineering
        X_fused = demo_feature_processor(title_emb, body_emb, global_emb, df_sample)

        # Step 4: Model Training
        y_sample = df_sample["requester_received_pizza"].values
        demo_model_factory(X_fused, y_sample)

        # Step 5: Full Pipeline Execution
        demo_execution_manager()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nDemonstration FAILED with error: {e}")
        raise e
