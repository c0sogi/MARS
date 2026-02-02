import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_data, save_submission
from library.feature_engineering import FeaturePipeline
from library.random_forest import RandomForestTrainer
from library.neural_net import NeuralTrainer


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. OVERRIDE CONFIGURATION FOR SPEED
    # We modify the Config class attributes directly to create a "mini" version of the task.
    print("Configuring environment for rapid demonstration...")

    # Use a specific demo directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "output")

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce model complexity and training duration
    Config.RF_N_ESTIMATORS = 5
    Config.RF_N_JOBS = 1
    Config.MLP_EPOCHS = 2
    Config.MLP_BATCH_SIZE = 4
    Config.MLP_HIDDEN_DIM = 32
    Config.MLP_PROJECTION_DIM = 16
    Config.TFIDF_VOCAB_SIZE = 50
    Config.TOP_K_MI_SUBREDDITS = 5

    # Set seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    # 2. DATA LOADING
    print("\n--- Loading Data ---")
    # Load full data using utility
    df_train_full = load_data("train")
    df_val_full = load_data("val")

    # Subsample for speed (20 train samples, 10 val samples)
    df_train = df_train_full.head(20).copy()
    df_val = df_val_full.head(10).copy()

    print(f"Train subset shape: {df_train.shape}")
    print(f"Val subset shape: {df_val.shape}")

    # Verify data loading logic
    assert Config.TARGET_COL in df_train.columns, "Target column missing in loaded data"
    assert Config.SUBREDDIT_COL in df_train.columns, "Subreddit column missing"
    assert isinstance(
        df_train[Config.SUBREDDIT_COL].iloc[0], list
    ), "Subreddit column not parsed as list"

    # 3. FEATURE ENGINEERING
    print("\n--- Running Feature Pipeline ---")
    pipeline = FeaturePipeline()

    # Fit and transform on train
    print("Fitting and transforming train data...")
    train_features = pipeline.fit_transform(df_train, split_name="demo_train")

    # Transform val
    print("Transforming val data...")
    val_features = pipeline.transform(df_val, split_name="demo_val")

    # Verify Feature Output
    expected_keys = [
        "rf_features",
        "mlp_metadata",
        "mlp_title_emb",
        "mlp_body_emb",
        "mlp_history_emb",
        "peak_relevance",
        "labels",
    ]
    for key in expected_keys:
        assert key in train_features, f"Missing key {key} in train features"
        assert key in val_features, f"Missing key {key} in val features"

    # Check shapes
    n_train = len(df_train)
    n_val = len(df_val)

    assert train_features["rf_features"].shape[0] == n_train
    assert val_features["rf_features"].shape[0] == n_val
    assert train_features["mlp_title_emb"].shape == (n_train, 384)  # SBERT dim

    print("Feature pipeline verification successful.")

    # 4. RANDOM FOREST TRAINING
    print("\n--- Running Random Forest Trainer ---")
    rf_trainer = RandomForestTrainer()

    # Train
    val_auc = rf_trainer.train(train_features, val_features)
    print(f"RF Validation AUC: {val_auc:.4f}")

    # Predict
    rf_preds = rf_trainer.predict(val_features)

    # Verify RF
    assert isinstance(val_auc, float)
    assert 0.0 <= val_auc <= 1.0
    assert len(rf_preds) == n_val
    assert np.all((rf_preds >= 0) & (rf_preds <= 1))
    print("Random Forest verification successful.")

    # 5. NEURAL NETWORK TRAINING
    print("\n--- Running Neural Network Trainer ---")
    # Calculate input dim for metadata (from pipeline output)
    input_dim_meta = train_features["mlp_metadata"].shape[1]

    nn_trainer = NeuralTrainer(input_dim_metadata=input_dim_meta)

    # Train
    nn_trainer.fit(train_features, val_features)

    # Predict
    nn_preds = nn_trainer.predict(val_features)

    # Verify NN
    assert len(nn_preds) == n_val
    assert np.all((nn_preds >= 0) & (nn_preds <= 1))
    print("Neural Network verification successful.")

    # 6. SUBMISSION GENERATION
    print("\n--- Generating Demo Submission ---")
    # Create dummy request IDs for the validation set subset
    dummy_ids = df_val["request_id"].values

    # Simple ensemble for demo
    ensemble_preds = (rf_preds + nn_preds) / 2

    save_submission(dummy_ids, ensemble_preds, filename="demo_submission.csv")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Check content
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (n_val, 2)
    assert Config.TARGET_COL in df_sub.columns
    print(f"Submission saved to {submission_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
