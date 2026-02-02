import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.models_mlp import train_model, predict
from library.models_rf import train_rf_model, predict_rf
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_reproducibility(seed=42):
    """Sets seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_config():
    """
    Overrides Config settings for a fast demonstration.
    Uses a small subset of data and minimal training iterations.
    """
    print("[Demo] Configuring environment for fast demonstration...")

    # Use a separate cache directory for the demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.DEBUG = True
    Config.MAX_SAMPLES = 50  # Use only 50 samples for speed
    Config.MLP_EPOCHS = 2  # Train for only 2 epochs
    Config.MLP_BATCH_SIZE = 8
    Config.RF_N_ESTIMATORS = 10  # Fewer trees for RF

    # Force device to CPU if GPU is not critical for this small test to avoid init overhead
    # (Optional, but keeping Config.DEVICE as is usually fine)

    print(f"[Demo] Cache Dir: {Config.CACHE_DIR}")
    print(f"[Demo] Max Samples: {Config.MAX_SAMPLES}")


def test_data_loader():
    """Validates the DataLoader class."""
    print("\n=== Testing DataLoader ===")
    dl = DataLoader()

    # Force reload to ensure we use the sample limit
    train_df, val_df, test_df = dl.load_dataset(load_cached_data=False)

    # Assertions
    assert (
        len(train_df) <= Config.MAX_SAMPLES
    ), f"Train size {len(train_df)} exceeds limit"
    assert len(val_df) <= Config.MAX_SAMPLES, f"Val size {len(val_df)} exceeds limit"
    assert len(test_df) <= Config.MAX_SAMPLES, f"Test size {len(test_df)} exceeds limit"

    required_cols = ["request_id", "request_text", "requester_received_pizza"]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in train_df"

    print(f"DataLoader successful. Train shape: {train_df.shape}")
    return train_df, val_df, test_df


def test_feature_engineering():
    """Validates the FeatureEngineer class."""
    print("\n=== Testing FeatureEngineer ===")
    fe = FeatureEngineer()

    # Generate features (this will trigger SBERT and TFIDF processing)
    # We pass load_cached_data=False to force generation on the small subset
    features = fe.create_features(load_cached_data=False)

    # Validate structure
    for split in ["train", "val", "test"]:
        assert split in features, f"Missing split {split} in features"
        assert "rf" in features[split], f"Missing 'rf' key in {split}"
        assert "mlp" in features[split], f"Missing 'mlp' key in {split}"

        # Check RF features
        assert "dense" in features[split]["rf"]
        assert "tfidf" in features[split]["rf"]

        # Check MLP features
        mlp_keys = ["title", "body", "history", "history_mask", "centroid", "metadata"]
        for key in mlp_keys:
            assert key in features[split]["mlp"], f"Missing MLP feature {key}"

    # Check shapes for Training set
    n_train = features["train"]["mlp"]["metadata"].shape[0]
    assert n_train <= Config.MAX_SAMPLES

    print("Feature Engineering successful. Feature dictionary structure validated.")
    return features


def test_mlp_model(features, train_df, val_df):
    """Validates MLP model training and inference."""
    print("\n=== Testing MLP Model ===")

    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    mlp_meta_dim = features["train"]["mlp"]["metadata"].shape[1]

    # Train
    model = train_model(
        features["train"]["mlp"], y_train, features["val"]["mlp"], y_val, mlp_meta_dim
    )

    assert isinstance(model, torch.nn.Module), "Model is not a PyTorch Module"

    # Predict
    preds = predict(model, features["test"]["mlp"])

    assert len(preds) == len(
        features["test"]["mlp"]["metadata"]
    ), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print(f"MLP Training successful. Test predictions mean: {preds.mean():.4f}")


def test_rf_model(features, train_df, val_df):
    """Validates Random Forest model training and inference."""
    print("\n=== Testing Random Forest Model ===")

    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # Train
    model = train_rf_model(
        features["train"]["rf"], y_train, features["val"]["rf"], y_val
    )

    # Predict
    preds = predict_rf(model, features["test"]["rf"])

    assert (
        len(preds) == features["test"]["rf"]["dense"].shape[0]
    ), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print(f"RF Training successful. Test predictions mean: {preds.mean():.4f}")


def test_full_trainer():
    """Validates the high-level Trainer class."""
    print("\n=== Testing Trainer Pipeline ===")

    trainer = Trainer()

    # Run the full pipeline
    # We use the overrides passed to the train method
    trainer.train(debug=True, max_samples=Config.MAX_SAMPLES, epochs=1)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns
    assert len(df_sub) <= Config.MAX_SAMPLES

    print(f"Trainer pipeline successful. Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # 1. Setup
    set_reproducibility()
    setup_demo_config()

    # 2. Test Data Loading
    train_df, val_df, test_df = test_data_loader()

    # 3. Test Feature Engineering
    # We pass the loaded DFs implicitly via the cache or by reloading in the class
    # For this test, we let FeatureEngineer handle loading internally using the Config settings
    features = test_feature_engineering()

    # 4. Test Models
    test_mlp_model(features, train_df, val_df)
    test_rf_model(features, train_df, val_df)

    # 5. Test Full Pipeline
    test_full_trainer()

    print("\nAll demonstrations completed successfully.")
