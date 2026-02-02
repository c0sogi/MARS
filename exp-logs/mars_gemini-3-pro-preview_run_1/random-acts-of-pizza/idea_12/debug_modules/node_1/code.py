import os
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Import Config and modify for fast demonstration
from library.config import Config

print("Configuring environment for fast demonstration...")
# Enable Debug mode to use a small subset of data
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 60  # Small sample size for speed

# Reduce Model Complexity for Speed
Config.RF_N_ESTIMATORS = 10
Config.RF_N_JOBS = 2
Config.MLP_EPOCHS = 2
Config.MLP_BATCH_SIZE = 8
Config.MLP_HIDDEN_DIMS = [32, 16]  # Smaller network
Config.MLP_PATIENCE = 1

# Reduce Feature Dimensionality
Config.TFIDF_TITLE_MAX_FEATURES = 50
Config.TFIDF_BODY_MAX_FEATURES = 50
Config.NUM_TOPIC_CLUSTERS = 3

# Redirect output directories to a demo folder to avoid conflicts
Config.WORKING_DIR = "./working/demo_execution"
Config.SUBMISSION_DIR = "./working/demo_output"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Set Seeds for Reproducibility
np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_SEED)

# Import library modules after Config modification
from library.feature_engineering import run_feature_engineering
from library.models_rf import StreamARF
from library.models_mlp import StreamBMLP
from library.train_eval import run_training_pipeline


def verify_feature_engineering():
    print("\n=== Verifying Feature Engineering ===")

    # Force re-computation by setting load_cached_data=False
    train_data, val_data, test_data = run_feature_engineering(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Check if data is a dictionary
    assert isinstance(
        train_data, (dict, np.lib.npyio.NpzFile)
    ), "Train data should be a dict or NpzFile"

    # Check for required keys
    required_keys = [
        "rf_meta",
        "rf_tfidf",
        "rf_topics",
        "mlp_meta",
        "mlp_title_emb",
        "mlp_body_emb",
        "mlp_hist_emb",
        "y",
        "ids",
    ]
    for key in required_keys:
        assert key in train_data, f"Missing key '{key}' in train data"

    # Verify shapes based on Config
    n_samples = len(train_data["y"])
    print(f"Train samples generated: {n_samples}")

    # Allow for slight deviation if data was filtered, but generally should match debug size
    assert n_samples <= Config.DEBUG_SAMPLE_SIZE, "Sample size exceeds debug limit"

    # Check TF-IDF shape
    # Title + Body features
    expected_tfidf_dim = (
        Config.TFIDF_TITLE_MAX_FEATURES + Config.TFIDF_BODY_MAX_FEATURES
    )
    assert (
        train_data["rf_tfidf"].shape[1] == expected_tfidf_dim
    ), f"RF TF-IDF dim mismatch. Expected {expected_tfidf_dim}, got {train_data['rf_tfidf'].shape[1]}"

    # Check MLP History Embedding shape: (N, SeqLen, EmbDim)
    assert (
        len(train_data["mlp_hist_emb"].shape) == 3
    ), "MLP History embeddings should be 3D"
    assert (
        train_data["mlp_hist_emb"].shape[2] == Config.EMBEDDING_DIM
    ), "Embedding dimension mismatch"

    print("Feature Engineering verification passed.")
    return train_data, val_data, test_data


def verify_stream_a_rf(train_data, val_data):
    print("\n=== Verifying Stream A (Random Forest) ===")

    model = StreamARF()

    # Test Fitting
    model.fit(train_data)

    # Test Prediction
    preds = model.predict_proba(val_data)

    # Verify Predictions
    assert len(preds) == len(val_data["y"]), "Prediction length mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities [0, 1]"

    # Test Evaluation
    auc = model.evaluate(val_data)
    assert isinstance(auc, float), "Evaluation should return a float"
    assert 0.0 <= auc <= 1.0, "AUC must be between 0 and 1"

    print(f"Stream A verification passed. AUC: {auc:.4f}")


def verify_stream_b_mlp(train_data, val_data):
    print("\n=== Verifying Stream B (MLP) ===")

    model = StreamBMLP()

    # Test Fitting
    model.fit(train_data, val_data)

    # Test Prediction
    preds = model.predict_proba(val_data)

    # Verify Predictions
    assert len(preds) == len(val_data["y"]), "Prediction length mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities [0, 1]"

    # Test Evaluation
    auc = model.evaluate(val_data)
    assert isinstance(auc, float), "Evaluation should return a float"
    assert 0.0 <= auc <= 1.0, "AUC must be between 0 and 1"

    print(f"Stream B verification passed. AUC: {auc:.4f}")


def verify_full_pipeline():
    print("\n=== Verifying Full Training Pipeline ===")

    # Run pipeline
    metrics = run_training_pipeline(load_cached_data=True, debug=Config.DEBUG)

    # Verify Metrics Dictionary
    assert "rf_auc" in metrics, "Missing RF AUC in metrics"
    assert "mlp_auc" in metrics, "Missing MLP AUC in metrics"
    assert "ensemble_auc" in metrics, "Missing Ensemble AUC in metrics"

    print(f"Pipeline Metrics: {metrics}")

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "request_id" in df_sub.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in df_sub.columns
    ), "Submission missing target column"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Submission file verified at {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    print("Starting Demo Execution...")

    # 1. Verify Data Processing
    # We keep the data in memory to pass to model verification steps
    train_data, val_data, test_data = verify_feature_engineering()

    # 2. Verify Stream A (RF)
    verify_stream_a_rf(train_data, val_data)

    # 3. Verify Stream B (MLP)
    verify_stream_b_mlp(train_data, val_data)

    # 4. Verify Full Pipeline Integration
    # This will re-load cached features and run everything end-to-end
    verify_full_pipeline()

    print("\nAll verifications passed successfully!")
