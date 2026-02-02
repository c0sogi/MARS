import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# =============================================================================
# 1. CONFIGURATION OVERRIDES
# =============================================================================
# We must import library.config and modify its attributes BEFORE importing
# other modules from the library. This ensures that when other modules do
# "from library.config import VAR", they get our modified values.

import library.config

print("Overriding configuration for demonstration...")
library.config.DEBUG_MODE = True
library.config.DEBUG_SAMPLE_SIZE = 50  # Small sample for speed
library.config.EPOCHS = 1
library.config.RF_ESTIMATORS = 10
library.config.TFIDF_VOCAB_SIZE = 100
library.config.TOP_K_COMMUNITIES = 10
library.config.BATCH_SIZE = 8
library.config.CACHE_DIR = "./working/demo_cache"
library.config.SUBMISSION_DIR = "./working/demo_output"

# Ensure clean slate for demo directories
if os.path.exists(library.config.CACHE_DIR):
    shutil.rmtree(library.config.CACHE_DIR)
os.makedirs(library.config.CACHE_DIR, exist_ok=True)

if os.path.exists(library.config.SUBMISSION_DIR):
    shutil.rmtree(library.config.SUBMISSION_DIR)
os.makedirs(library.config.SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# 2. IMPORTS (After Config Override)
# =============================================================================

from library.utils import set_seed
from library.features import FeaturePipeline
from library.dataset import get_dataloaders
from library.model_tree import DualViewRandomForest
from library.model_nn import NeuralNetworkModel
from library.trainer import Trainer

# =============================================================================
# 3. DEMONSTRATION & VERIFICATION LOGIC
# =============================================================================


def test_feature_pipeline():
    print("\n--- Testing FeaturePipeline ---")
    pipeline = FeaturePipeline()

    # Run pipeline (this will generate and cache features)
    # load_cached_data=False forces re-computation for this demo
    rf_data, mlp_data = pipeline.run(load_cached_data=False)

    # Verify RF Data
    print("Verifying RF Data structure...")
    assert "X_train" in rf_data
    assert "y_train" in rf_data
    assert rf_data["X_train"].shape[0] == library.config.DEBUG_SAMPLE_SIZE
    assert rf_data["X_train"].shape[1] > 0

    # Verify MLP Data
    print("Verifying MLP Data structure...")
    assert "train_title_emb" in mlp_data
    assert "train_hist_seq" in mlp_data
    assert mlp_data["train_title_emb"].shape[0] == library.config.DEBUG_SAMPLE_SIZE
    # SBERT dimension is typically 384
    assert mlp_data["train_title_emb"].shape[1] == 384

    print("FeaturePipeline tests passed.")
    return rf_data, mlp_data


def test_dataloaders():
    print("\n--- Testing DataLoaders ---")
    # This relies on the cache generated in test_feature_pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=library.config.BATCH_SIZE, verbose=True
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    print("Verifying Train Batch keys and shapes...")
    expected_keys = [
        "title_emb",
        "body_emb",
        "history_seq",
        "history_mask",
        "dense_features",
        "label",
    ]
    for k in expected_keys:
        assert k in batch, f"Missing key {k} in batch"

    # Check batch size
    current_batch_size = batch["title_emb"].shape[0]
    assert current_batch_size <= library.config.BATCH_SIZE

    # Check tensor types
    assert isinstance(batch["title_emb"], torch.Tensor)
    assert batch["label"].dtype == torch.float32

    print("DataLoader tests passed.")


def test_rf_model():
    print("\n--- Testing DualViewRandomForest (Stream A) ---")
    rf_model = DualViewRandomForest()

    # Run the full RF pipeline
    test_ids, test_preds, val_auc = rf_model.run(load_cached_data=True)

    print(f"RF Validation AUC: {val_auc}")

    # Verify outputs
    assert len(test_ids) == library.config.DEBUG_SAMPLE_SIZE
    assert len(test_preds) == library.config.DEBUG_SAMPLE_SIZE
    assert isinstance(val_auc, float)
    assert 0.0 <= val_auc <= 1.0

    print("Random Forest model tests passed.")


def test_mlp_model():
    print("\n--- Testing NeuralNetworkModel (Stream B) ---")
    mlp_model = NeuralNetworkModel()

    # Run the full MLP pipeline
    test_ids, test_preds, val_auc = mlp_model.run(load_cached_data=True)

    print(f"MLP Validation AUC: {val_auc}")

    # Verify outputs
    assert len(test_ids) == library.config.DEBUG_SAMPLE_SIZE
    assert len(test_preds) == library.config.DEBUG_SAMPLE_SIZE
    assert isinstance(val_auc, float)

    print("Neural Network model tests passed.")


def test_trainer_integration():
    print("\n--- Testing Full Trainer Integration ---")
    trainer = Trainer()

    # Execute the full trainer run
    final_preds = trainer.run(load_cached_data=True)

    # Verify Submission File
    submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    assert len(df_sub) == library.config.DEBUG_SAMPLE_SIZE
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns
    assert df_sub["requester_received_pizza"].dtype == float

    print("Trainer integration tests passed.")


def run_demo():
    # Set seed for reproducibility
    set_seed(42)

    print("Starting Library Usage Demonstration...")
    print(f"Debug Mode: {library.config.DEBUG_MODE}")
    print(f"Cache Dir: {library.config.CACHE_DIR}")

    # 1. Test Feature Engineering
    test_feature_pipeline()

    # 2. Test Dataset Loading
    test_dataloaders()

    # 3. Test Random Forest Stream
    test_rf_model()

    # 4. Test Neural Network Stream
    test_mlp_model()

    # 5. Test Full Trainer (Ensemble)
    test_trainer_integration()

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    run_demo()
