import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import DataLoader
from library.embedder import EmbeddingGenerator
from library.feature_processor import WhitenedFusionPipeline
from library.model_builder import ModelBuilder
from library.engine import CrossValidationTrainer

if __name__ == "__main__":
    print(">>> Starting Demonstration of Random Acts of Pizza Pipeline")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # We override Config parameters to ensure the demo runs quickly
    # and validates logic without processing the entire dataset.
    print(">>> Configuring environment for fast demonstration...")

    set_seed(42)

    # Enable Debug mode to load only a small sample of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 40  # Small enough for speed, large enough for 2-fold CV

    # Reduce Model Complexity for Demo
    Config.N_FOLDS = 2
    Config.N_BAGGING_ESTIMATORS = 2

    # Reduce PCA components to fit within the small sample size (n_samples < n_components constraint)
    # With 40 samples and 2 folds, train size is 20. Components must be < 20.
    Config.PCA_N_COMPONENTS = 10

    # Simplify Grid Search to avoid long tuning times
    Config.LR_PARAM_GRID = {"C": [1.0], "solver": ["lbfgs"]}

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n>>> [Step 1] Demonstrating DataLoader...")
    loader = DataLoader()

    # load_cached=False forces the loader to process raw JSON files
    train_df, val_df, test_df = loader.load_data(load_cached=False)

    print(f"   Train Data Shape: {train_df.shape}")
    print(f"   Val Data Shape:   {val_df.shape}")
    print(f"   Test Data Shape:  {test_df.shape}")

    # Verification
    assert (
        len(train_df) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train data exceeds debug sample size"
    assert "request_text_edit_aware" in train_df.columns, "Missing text column"
    assert "text_concat" in train_df.columns, "Missing concatenated text column"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Missing target column in train"

    # ==========================================
    # 3. Embedding Generation
    # ==========================================
    print("\n>>> [Step 2] Demonstrating EmbeddingGenerator...")
    embedder = EmbeddingGenerator()

    # Generates embeddings using SentenceTransformers (MiniLM and MPNet)
    # load_cached=False forces generation (inference)
    embeddings = embedder.generate_embeddings(
        train_df, val_df, test_df, load_cached=False
    )

    # Verification
    expected_keys = [
        "train_anchor",
        "val_anchor",
        "test_anchor",
        "train_aux",
        "val_aux",
        "test_aux",
    ]
    for key in expected_keys:
        assert key in embeddings, f"Missing embedding key: {key}"

    # Check Dimensions
    # Anchor: [Title (384) | Body (384)] -> 768 dimensions
    assert embeddings["train_anchor"].shape == (
        len(train_df),
        768,
    ), "Incorrect Anchor embedding shape"
    # Aux: [Global (768)] -> 768 dimensions
    assert embeddings["train_aux"].shape == (
        len(train_df),
        768,
    ), "Incorrect Aux embedding shape"

    print("   Embeddings generated and verified.")

    # ==========================================
    # 4. Feature Processing (Fusion Pipeline)
    # ==========================================
    print("\n>>> [Step 3] Demonstrating WhitenedFusionPipeline...")
    pipeline = WhitenedFusionPipeline()

    # Fit pipeline on training data (PCA on Aux, Scaler on Metadata)
    pipeline.fit(embeddings["train_anchor"], embeddings["train_aux"], train_df)

    # Transform data
    X_train_fused = pipeline.transform(
        embeddings["train_anchor"], embeddings["train_aux"], train_df
    )

    print(f"   Fused Feature Matrix Shape: {X_train_fused.shape}")

    # Verification of Output Dimensions
    # Expected: Title(384) + Body(384) + Aux_PCA(10) + Metadata(10) = 788
    expected_dim = 384 + 384 + Config.PCA_N_COMPONENTS + 10
    assert (
        X_train_fused.shape[1] == expected_dim
    ), f"Expected {expected_dim} features, got {X_train_fused.shape[1]}"

    # ==========================================
    # 5. Model Building
    # ==========================================
    print("\n>>> [Step 4] Demonstrating ModelBuilder...")
    builder = ModelBuilder()
    optimizer = builder.get_bagged_lr_optimizer()

    # Verification
    assert isinstance(
        optimizer, GridSearchCV
    ), "ModelBuilder did not return a GridSearchCV object"
    print("   Optimizer object created successfully.")

    # ==========================================
    # 6. Full Engine Execution
    # ==========================================
    print("\n>>> [Step 5] Demonstrating CrossValidationTrainer (Full Pipeline Run)...")
    trainer = CrossValidationTrainer()

    # We use load_cached_data=True here because Step 2 (EmbeddingGenerator)
    # already saved the computed embeddings to disk. This saves re-computation time.
    trainer.run(load_cached_data=True)

    # Verification of Submission
    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), "Submission file was not created"

    submission_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"   Submission File Shape: {submission_df.shape}")

    assert len(submission_df) == len(test_df), "Submission row count mismatch"
    assert "request_id" in submission_df.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in submission_df.columns
    ), "Submission missing target column"

    # Check that predictions are probabilities (between 0 and 1)
    preds = submission_df["requester_received_pizza"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are not valid probabilities"

    print("\n>>> Demonstration Completed Successfully!")
