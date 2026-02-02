import os
import sys
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import TrainingConfig, ModelConfig, FeatureConfig, PathConfig
from library.utils import seed_everything
from library.feature_engineering import run_feature_engineering
from library.data_factory import load_and_preprocess, get_pytorch_dataloaders
from library.models import TripleBranchMLP
from library.training_engine import CrossValidator, train_rf, train_mlp


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring for fast demonstration...")
    seed_everything(42)

    # Override Training Config to minimize runtime
    TrainingConfig.EPOCHS = 1
    TrainingConfig.NUM_FOLDS = 2
    TrainingConfig.BATCH_SIZE = 16

    # Override Model Config for Random Forest to be lightweight
    ModelConfig.RF_PARAMS["n_estimators"] = 5
    ModelConfig.RF_PARAMS["max_depth"] = 5

    # Override Feature Config to speed up text processing
    FeatureConfig.TFIDF_MAX_FEATURES = 100
    # We keep SBERT model as is; it is small enough for the A100 GPU.

    print("Configuration updated: Epochs=1, Folds=2, RF_Estimators=5")

    # ---------------------------------------------------------
    # 2. Feature Engineering & Data Preprocessing
    # ---------------------------------------------------------
    print("\n[Step 2] Running Feature Engineering...")
    # We set load_cached_data=False to force the pipeline to run and demonstrate functionality.
    # This processes metadata/train.csv, val.csv, and test.csv.
    stream_a_data, stream_b_data = load_and_preprocess(load_cached_data=False)

    # Unpack for verification
    # Stream A is for Random Forest (Lexical/Sparse)
    data_a_train, data_a_val, data_a_test = stream_a_data
    # Stream B is for MLP (Semantic/Dense)
    data_b_train, data_b_val, data_b_test = stream_b_data

    # Verify Stream A Data Structure
    print("Verifying Stream A (RF) data...")
    assert "tfidf" in data_a_train
    assert "community" in data_a_train
    assert "meta_num" in data_a_train
    assert "y" in data_a_train
    assert data_a_train["y"].shape[0] == data_a_train["tfidf"].shape[0]
    print(f" -> Stream A Train Samples: {data_a_train['y'].shape[0]}")

    # Verify Stream B Data Structure
    print("Verifying Stream B (MLP) data...")
    assert "sbert" in data_b_train
    assert "community" in data_b_train
    assert "meta_num" in data_b_train
    assert data_b_train["sbert"].shape[1] == ModelConfig.SEMANTIC_INPUT_DIM
    print(f" -> Stream B Train Samples: {data_b_train['y'].shape[0]}")

    # ---------------------------------------------------------
    # 3. Data Loading (PyTorch)
    # ---------------------------------------------------------
    print("\n[Step 3] Creating PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = get_pytorch_dataloaders(
        data_b_train, data_b_val, data_b_test, batch_size=TrainingConfig.BATCH_SIZE
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    print(" -> Batch keys:", list(batch.keys()))

    assert "semantic_input" in batch
    assert "community_input" in batch
    assert "meta_input" in batch
    assert "target" in batch
    assert batch["semantic_input"].shape[0] == TrainingConfig.BATCH_SIZE
    print(" -> Batch dimensions verified.")

    # ---------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 4] Testing TripleBranchMLP Architecture...")
    meta_dim = data_b_train["meta_num"].shape[1]
    model = TripleBranchMLP(meta_dim=meta_dim)
    model.to(TrainingConfig.DEVICE)

    # Move batch to device
    sem = batch["semantic_input"].to(TrainingConfig.DEVICE)
    comm = batch["community_input"].to(TrainingConfig.DEVICE)
    meta = batch["meta_input"].to(TrainingConfig.DEVICE)

    # Perform Forward pass
    logits = model(sem, comm, meta)
    print(f" -> Logits shape: {logits.shape}")

    # Assert output shape is (Batch_Size, 1)
    assert logits.shape == (TrainingConfig.BATCH_SIZE, 1)
    print(" -> Forward pass successful.")

    # ---------------------------------------------------------
    # 5. Running the Full Training Engine
    # ---------------------------------------------------------
    print("\n[Step 5] Running CrossValidator (Reduced Folds/Epochs)...")
    # This will:
    # 1. Load the data (using the cache we just generated)
    # 2. Run 2-Fold CV training RF and MLP
    # 3. Train a Stacking Meta-Learner
    # 4. Generate predictions on Test set
    cv = CrossValidator(k_folds=TrainingConfig.NUM_FOLDS)
    cv.run()

    # ---------------------------------------------------------
    # 6. Verify Submission
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying Submission...")
    submission_path = PathConfig.SUBMISSION_FILE

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f" -> Submission shape: {df_sub.shape}")
    print(df_sub.head(3))

    # Check row count (Test set size is 1162)
    expected_rows = 1162
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check values are valid probabilities
    probs = df_sub["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
