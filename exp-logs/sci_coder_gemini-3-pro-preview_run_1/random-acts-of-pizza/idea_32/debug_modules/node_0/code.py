import os
import shutil
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

# Import provided library modules
from library.config import Config
from library.data_utils import seed_everything, load_dataset, save_submission
from library.feature_engineering import FeatureEngineer
from library.dataset import PizzaDataset
from library.models import DualQueryGatedMLP, RFWrapper
from library.training import train_mlp_model, train_rf_model, predict_ensemble


def run_demo():
    print("=== Starting Demonstration of Library Usage ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Enable Debug mode to use a small subset of data (100 rows)
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Small enough for very fast execution

    # Reduce training complexity
    Config.NUM_EPOCHS = 1
    Config.RF_N_ESTIMATORS = 10
    Config.BATCH_SIZE = 16

    # Redirect working directory to avoid messing with real experiment cache
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)

    # Update paths dependent on WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.CACHE_TRAIN_PROCESSED = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.CACHE_VAL_PROCESSED = os.path.join(
        Config.WORKING_DIR, "val_processed.parquet"
    )
    Config.CACHE_TEST_PROCESSED = os.path.join(
        Config.WORKING_DIR, "test_processed.parquet"
    )
    Config.MODEL_RF_PATH = os.path.join(Config.WORKING_DIR, "rf_model.pkl")
    Config.MODEL_MLP_PATH = os.path.join(Config.WORKING_DIR, "nn_model.pth")

    # Set seed for reproducibility
    seed_everything(Config.RANDOM_SEED)
    print("Configuration updated: DEBUG=True, Epochs=1, Project=demo_execution")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Datasets...")
    train_df, val_df, test_df = load_dataset(debug=Config.DEBUG)

    # Verification
    assert len(train_df) == Config.DEBUG_SIZE, f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == Config.DEBUG_SIZE, f"Val size mismatch: {len(val_df)}"
    assert len(test_df) == Config.DEBUG_SIZE, f"Test size mismatch: {len(test_df)}"
    print(f"Data loaded successfully. Train shape: {train_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[3] Running Feature Engineering...")
    fe = FeatureEngineer()

    # Force re-computation (load_cached_data=False) to demonstrate logic
    train_feats, val_feats, test_feats = fe.process_data(load_cached_data=False)

    # Verification of RF Features (Sparse Matrix)
    assert sp.issparse(train_feats["X_rf"]), "X_rf should be a sparse matrix"
    assert train_feats["X_rf"].shape[0] == Config.DEBUG_SIZE, "X_rf row count mismatch"

    # Verification of MLP Features (Dictionary of Arrays)
    mlp_keys = ["title_emb", "body_emb", "history_emb", "metadata"]
    for key in mlp_keys:
        assert key in train_feats["X_mlp"], f"Missing key {key} in MLP features"
        assert (
            len(train_feats["X_mlp"][key]) == Config.DEBUG_SIZE
        ), f"Dimension mismatch for {key}"

    # Verify SBERT embedding dimension
    assert train_feats["X_mlp"]["title_emb"].shape[1] == Config.SBERT_EMBEDDING_DIM

    print("Feature engineering complete. Shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Dataset & Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset and Model Architecture...")

    # Instantiate Dataset
    ds = PizzaDataset(train_feats["X_mlp"], labels=train_feats["y"])
    sample = ds[0]

    # Verify Dataset Output
    expected_keys = [
        "title_emb",
        "body_emb",
        "history_emb",
        "metadata",
        "history_padding_mask",
        "label",
    ]
    for k in expected_keys:
        assert k in sample, f"Dataset sample missing key: {k}"

    print("Dataset sample keys verified.")

    # Instantiate MLP Model
    model = DualQueryGatedMLP(
        embedding_dim=Config.SBERT_EMBEDDING_DIM,
        metadata_dim=len(Config.NUMERIC_COLS),
        hidden_dim=Config.MLP_HIDDEN_DIM,
    )

    # Perform Dummy Forward Pass
    # Add batch dimension (unsqueeze)
    with torch.no_grad():
        output = model(
            sample["title_emb"].unsqueeze(0),
            sample["body_emb"].unsqueeze(0),
            sample["history_emb"].unsqueeze(0),
            sample["metadata"].unsqueeze(0),
            sample["history_padding_mask"].unsqueeze(0),
        )

    assert output.shape == (
        1,
        1,
    ), f"Model output shape mismatch. Expected (1, 1), got {output.shape}"
    print("MLP Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Models
    # -------------------------------------------------------------------------
    print("\n[5] Training Models...")

    # Train Random Forest
    print("-> Training Random Forest...")
    rf_model = train_rf_model(train_feats)
    assert os.path.exists(Config.MODEL_RF_PATH), "RF model file was not saved."

    # Train MLP
    print("-> Training MLP...")
    # Use CPU for demo stability/simplicity, or Config.DEVICE if available
    device = Config.DEVICE
    mlp_model = train_mlp_model(train_feats, val_feats, device=device)
    assert os.path.exists(Config.MODEL_MLP_PATH), "MLP model file was not saved."

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 6. Prediction & Ensemble
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions...")

    final_preds = predict_ensemble(rf_model, mlp_model, test_feats, device=device)

    # Verify Predictions
    assert len(final_preds) == Config.DEBUG_SIZE, "Prediction count mismatch"
    assert np.all(
        (final_preds >= 0) & (final_preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print(
        f"Generated {len(final_preds)} predictions. Mean prob: {np.mean(final_preds):.4f}"
    )

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    print("\n[7] Saving Submission...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    save_submission(final_preds, test_df, filename=submission_path)

    assert os.path.exists(submission_path), "Submission file not found."

    # Verify file content
    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (Config.DEBUG_SIZE, 2), "Submission CSV shape mismatch"
    assert (
        "request_id" in sub_df.columns and "requester_received_pizza" in sub_df.columns
    )

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
