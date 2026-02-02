import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device, print_log
from library.text_encoder import SBERTEncoder, TFIDFEncoder
from library.feature_manager import FeatureManager
from library.model_rf import train_rf, predict_rf
from library.model_mlp import train_mlp, predict_mlp


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override
    # --------------------------------------------------------------------------
    # Modify Config attributes to ensure the demo runs quickly and uses a separate directory
    print_log("Configuring runtime parameters for speed...")

    Config.MAX_SAMPLES = 60  # Use only 60 samples from datasets
    Config.RF_ESTIMATORS = 5  # Fewer trees for RF
    Config.MLP_EPOCHS = 2  # Fewer epochs for MLP
    Config.MLP_BATCH_SIZE = 8  # Smaller batch size
    Config.MLP_HIDDEN_DIM = 16  # Smaller hidden dimension
    Config.TOP_K_SUBREDDITS = 5  # Fewer top-k indicators
    Config.TFIDF_VOCAB_SIZE = 50  # Smaller vocabulary

    # Setup a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to point to the new working directory
    # (Since these are initialized at import time, we must manually update them)
    Config.CACHE_RF_TRAIN = os.path.join(Config.WORKING_DIR, "rf_train.parquet")
    Config.CACHE_RF_VAL = os.path.join(Config.WORKING_DIR, "rf_val.parquet")
    Config.CACHE_RF_TEST = os.path.join(Config.WORKING_DIR, "rf_test.parquet")
    Config.CACHE_MLP_TRAIN = os.path.join(Config.WORKING_DIR, "mlp_train.npz")
    Config.CACHE_MLP_VAL = os.path.join(Config.WORKING_DIR, "mlp_val.npz")
    Config.CACHE_MLP_TEST = os.path.join(Config.WORKING_DIR, "mlp_test.npz")

    # --------------------------------------------------------------------------
    # 2. Utility Verification
    # --------------------------------------------------------------------------
    print_log("Verifying Utilities...")
    seed_everything(123)
    device = get_device()
    print(f"   -> Device detected: {device}")

    # --------------------------------------------------------------------------
    # 3. Text Encoder Verification
    # --------------------------------------------------------------------------
    print_log("Verifying Text Encoders...")

    # SBERT Encoder
    sbert = SBERTEncoder()
    dummy_texts = ["Pizza request", "Hungry student needs food"]
    embeddings = sbert.encode(dummy_texts, batch_size=2)

    assert embeddings.shape == (
        2,
        384,
    ), f"SBERT output shape mismatch: {embeddings.shape}"
    print("   -> SBERT Encoder: OK")

    # TFIDF Encoder
    tfidf = TFIDFEncoder(max_features=10)
    tfidf_mat = tfidf.fit_transform(dummy_texts)

    assert tfidf_mat.shape[0] == 2, "TFIDF row count mismatch"
    assert tfidf_mat.shape[1] <= 10, "TFIDF feature count mismatch"
    print("   -> TFIDF Encoder: OK")

    # --------------------------------------------------------------------------
    # 4. Feature Manager & Data Generation
    # --------------------------------------------------------------------------
    print_log("Verifying Feature Manager...")
    fm = FeatureManager()

    # Generate RF Data (Force re-compute to test logic)
    print_log("Generating RF Dataset (from scratch)...")
    X_rf_train, y_rf_train, X_rf_val, y_rf_val, X_rf_test, test_ids = fm.get_rf_dataset(
        load_cached_data=False
    )

    # Validation
    assert (
        len(X_rf_train) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} train samples, got {len(X_rf_train)}"
    assert len(y_rf_train) == Config.MAX_SAMPLES
    assert not X_rf_train.isnull().values.any(), "NaNs found in RF training data"
    print("   -> RF Data Generation: OK")

    # Generate MLP Data (Force re-compute)
    print_log("Generating MLP Dataset (from scratch)...")
    train_mlp, val_mlp, test_mlp = fm.get_mlp_dataset(load_cached_data=False)

    # Validation
    assert len(train_mlp["y"]) == Config.MAX_SAMPLES
    assert train_mlp["title_emb"].shape[1] == 384
    print("   -> MLP Data Generation: OK")

    # --------------------------------------------------------------------------
    # 5. Random Forest Model Workflow
    # --------------------------------------------------------------------------
    print_log("Verifying Random Forest Workflow...")

    # Train RF (using the cached data we just generated)
    rf_model, X_val_rf, y_val_rf, X_test_rf, _ = train_rf(load_cached_data=True)

    # Predict RF
    rf_probs = predict_rf(rf_model, X_val_rf)

    # Validation
    assert len(rf_probs) == len(y_val_rf)
    assert (
        0.0 <= rf_probs.min() and rf_probs.max() <= 1.0
    ), "RF probabilities out of range [0, 1]"
    print(f"   -> RF Sample Predictions: {rf_probs[:3]}")
    print("   -> RF Workflow: OK")

    # --------------------------------------------------------------------------
    # 6. MLP Model Workflow
    # --------------------------------------------------------------------------
    print_log("Verifying MLP Workflow...")

    # Train MLP (using cached data)
    mlp_model, val_data_mlp, test_data_mlp = train_mlp(load_cached_data=True)

    # Predict MLP (on test set)
    mlp_probs = predict_mlp(mlp_model, test_data_mlp)

    # Validation
    assert len(mlp_probs) == len(test_data_mlp["metadata"])
    assert (
        0.0 <= mlp_probs.min() and mlp_probs.max() <= 1.0
    ), "MLP probabilities out of range [0, 1]"
    print(f"   -> MLP Sample Predictions: {mlp_probs[:3]}")
    print("   -> MLP Workflow: OK")

    print_log("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
