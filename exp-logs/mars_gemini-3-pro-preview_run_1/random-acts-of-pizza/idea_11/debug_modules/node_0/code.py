import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import Library Components
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_dataset
from library.feature_engineering import generate_features
from library.semantic_processing import SemanticEngine
from library.dataset import get_dataloaders
from library.models import TopicAugmentedRF, AttentionGatedMLP, train_mlp_model
from library.engine import run_pipeline, get_rf_features

if __name__ == "__main__":
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> [1/6] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples per split
    Config.MLP_PARAMS["epochs"] = 1
    Config.MLP_PARAMS["batch_size"] = 4
    Config.RF_PARAMS["n_estimators"] = 5

    # Use a separate working directory for this demo to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.SUBMISSION_DIR = "./working/demo_output"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # =========================================================================
    # 2. Data Loading & Feature Engineering
    # =========================================================================
    print("\n>>> [2/6] Demonstrating Data Loading & Feature Engineering...")

    # Load raw data (forcing reload to show logic)
    train_df, val_df, test_df = load_dataset(load_cached_data=False)

    # Validation
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), "Train set size mismatch for debug mode."
    print(f"Loaded Train Shape: {train_df.shape}")

    # Generate Features
    train_df, val_df, test_df = generate_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation
    expected_feats = ["text_len_char", "num_subreddits", "vote_balance_ratio"]
    for feat in expected_feats:
        assert feat in train_df.columns, f"Feature {feat} missing from dataframe."
    print("Feature engineering verified.")

    # =========================================================================
    # 3. Semantic Processing (Embeddings & Topics)
    # =========================================================================
    print("\n>>> [3/6] Demonstrating Semantic Processing...")

    engine = SemanticEngine()
    # Process text and history (SBERT, TF-IDF, Topics)
    semantic_data = engine.process(train_df, val_df, test_df, load_cached_data=False)

    # Validation
    assert "train" in semantic_data
    assert "sbert_request" in semantic_data["train"]
    # Check embedding dimension (MiniLM-L6-v2 is 384 dim)
    assert semantic_data["train"]["sbert_request"].shape == (
        Config.DEBUG_SAMPLE_SIZE,
        384,
    )
    print("Semantic features generated and shapes verified.")

    # =========================================================================
    # 4. Stream A: Topic-Augmented Random Forest
    # =========================================================================
    print("\n>>> [4/6] Demonstrating Stream A (Random Forest)...")

    # Identify numerical columns for RF
    drop_cols = set(Config.DROP_COLS)
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in drop_cols
    ]

    # Prepare Feature Matrix
    X_train_rf = get_rf_features(train_df, semantic_data["train"], numeric_cols)
    y_train_rf = train_df[Config.TARGET_COL].values.astype(int)

    # Instantiate and Fit
    rf_model = TopicAugmentedRF()
    rf_model.fit(X_train_rf, y_train_rf)

    # Predict
    X_test_rf = get_rf_features(test_df, semantic_data["test"], numeric_cols)
    rf_probs = rf_model.predict_proba(X_test_rf)

    # Validation
    assert len(rf_probs) == len(test_df)
    assert 0.0 <= rf_probs.min() <= 1.0
    print("Random Forest trained and predictions generated.")

    # =========================================================================
    # 5. Stream B: Attention-Gated MLP
    # =========================================================================
    print("\n>>> [5/6] Demonstrating Stream B (Attention-Gated MLP)...")

    # Get DataLoaders (this handles scaling/preprocessing)
    # We use load_cached_data=True because we generated the files in steps 2 & 3
    # and saved them to the demo WORKING_DIR.
    train_loader, val_loader, test_loader, meta_dim = get_dataloaders(
        load_cached_data=True
    )

    print(f"MLP Input Metadata Dimension: {meta_dim}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    required_keys = ["metadata", "request_emb", "history_emb", "history_mask", "label"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Instantiate Model
    mlp_model = AttentionGatedMLP(metadata_dim=meta_dim)

    # Forward Pass Verification
    with torch.no_grad():
        logits = mlp_model(
            batch["metadata"],
            batch["request_emb"],
            batch["history_emb"],
            batch["history_mask"],
        )
    assert logits.shape == (batch["metadata"].shape[0], 1), "MLP output shape mismatch."

    # Run Short Training Loop
    print("Running short training loop...")
    mlp_model, history = train_mlp_model(mlp_model, train_loader, val_loader)

    assert len(history["train_loss"]) == Config.MLP_PARAMS["epochs"]
    print("MLP training loop completed successfully.")

    # =========================================================================
    # 6. Full Pipeline Integration
    # =========================================================================
    print("\n>>> [6/6] Running Full Pipeline Integration...")

    # Execute the end-to-end pipeline (Load -> Train RF -> Train MLP -> Ensemble -> Submit)
    run_pipeline(load_cached_data=True)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        assert len(sub_df) == len(test_df)
        assert Config.ID_COL in sub_df.columns
        assert Config.TARGET_COL in sub_df.columns
        print(
            f"Submission file created at {Config.SUBMISSION_PATH} with {len(sub_df)} rows."
        )
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n>>> Demonstration completed successfully!")
