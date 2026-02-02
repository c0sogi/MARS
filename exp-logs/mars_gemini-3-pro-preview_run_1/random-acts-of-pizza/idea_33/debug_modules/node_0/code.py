import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data_loader import DataLoader
from library.feature_engine import FeatureEngine
from library.dataset import PizzaDataset
from library.model_rf import RandomForestStream
from library.model_nn import NeuralNetworkStream
from library.engine import Engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configuration Override for Speed/Demo
    # ==========================================
    print("\n[1] Configuring environment for fast execution...")

    # Set a unique ID for this demo run to isolate cache
    Config.IDEA_ID = "demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_ID)
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_output")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    Config.setup()

    # Override Hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Only use 50 samples
    Config.RF_N_ESTIMATORS = 10  # Only 10 trees
    Config.MLP_MAX_EPOCHS = 2  # Only 2 epochs
    Config.MLP_BATCH_SIZE = 8  # Small batch size
    Config.SBERT_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.TFIDF_MAX_FEATURES = 100  # Reduce feature space

    # Set seed for reproducibility
    set_seed(Config.RANDOM_SEED)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Testing DataLoader...")
    loader = DataLoader()
    df_train, df_val, df_test = loader.load_data(debug_size=Config.DEBUG_SAMPLE_SIZE)

    # Verification
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(df_train)}"
    assert len(df_val) == Config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(df_val)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(df_test)}"
    assert "requester_subreddits_at_request" in df_train.columns
    assert isinstance(
        df_train.iloc[0]["requester_subreddits_at_request"], list
    ), "List parsing failed"
    print("DataLoader verification passed.")

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print("\n[3] Testing FeatureEngine...")
    fe = FeatureEngine()

    # Process data (force_retrain implicitly via load_cached_data=False or missing cache)
    # We pass load_cached_data=False to ensure we test the generation logic
    rf_data, mlp_data = fe.process_data(
        df_train, df_val, df_test, load_cached_data=False
    )

    # Verification - Random Forest Data
    assert "train" in rf_data and "val" in rf_data and "test" in rf_data
    assert rf_data["train"]["X"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert rf_data["train"]["y"].shape[0] == Config.DEBUG_SAMPLE_SIZE

    # Verification - MLP Data
    assert "train" in mlp_data
    assert "title_emb" in mlp_data["train"]
    assert mlp_data["train"]["title_emb"].shape == (
        Config.DEBUG_SAMPLE_SIZE,
        384,
    )  # SBERT dim
    print("FeatureEngine verification passed.")

    # ==========================================
    # 4. Dataset Class
    # ==========================================
    print("\n[4] Testing PizzaDataset...")
    train_dataset = PizzaDataset(mlp_data["train"])
    sample = train_dataset[0]

    # Verification
    assert isinstance(sample, dict)
    assert "title_emb" in sample
    assert isinstance(sample["title_emb"], torch.Tensor)
    assert sample["title_emb"].shape[0] == 384
    assert "target" in sample
    print("PizzaDataset verification passed.")

    # ==========================================
    # 5. Random Forest Stream
    # ==========================================
    print("\n[5] Testing RandomForestStream...")
    rf_stream = RandomForestStream()

    # Train
    rf_auc, rf_preds = rf_stream.train(rf_data, force_retrain=True)

    # Predict
    rf_test_preds = rf_stream.predict(rf_data["test"]["X"])

    # Verification
    assert isinstance(rf_auc, float)
    assert 0.0 <= rf_auc <= 1.0
    assert len(rf_preds) == Config.DEBUG_SAMPLE_SIZE
    assert len(rf_test_preds) == Config.DEBUG_SAMPLE_SIZE
    print(f"Random Forest Stream verification passed. Val AUC: {rf_auc:.4f}")

    # ==========================================
    # 6. Neural Network Stream
    # ==========================================
    print("\n[6] Testing NeuralNetworkStream...")
    nn_stream = NeuralNetworkStream()

    # Train
    nn_auc, nn_preds = nn_stream.train(mlp_data, force_retrain=True)

    # Predict
    nn_test_preds = nn_stream.predict(mlp_data["test"])

    # Verification
    assert isinstance(nn_auc, float)
    assert 0.0 <= nn_auc <= 1.0
    assert len(nn_preds) == Config.DEBUG_SAMPLE_SIZE
    assert len(nn_test_preds) == Config.DEBUG_SAMPLE_SIZE
    print(f"Neural Network Stream verification passed. Val AUC: {nn_auc:.4f}")

    # ==========================================
    # 7. Full Engine Integration
    # ==========================================
    print("\n[7] Testing Full Engine Pipeline...")
    # The Engine class orchestrates everything.
    # Since we've already populated the cache in step 3, this run will be fast
    # as it will load features from cache (unless we cleared it, but we kept it).
    # However, Engine.run() calls process_data with load_cached_data=True by default.

    engine = Engine()
    engine.run()

    # Verification
    if os.path.exists(Config.SUBMISSION_PATH):
        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        assert len(submission_df) == Config.DEBUG_SAMPLE_SIZE
        assert "request_id" in submission_df.columns
        assert "requester_received_pizza" in submission_df.columns
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print("Full Engine verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
