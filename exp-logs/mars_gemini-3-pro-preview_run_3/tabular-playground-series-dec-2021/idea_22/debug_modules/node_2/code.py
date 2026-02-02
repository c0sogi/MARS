import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.model_arch import VectorCrossLayer, ResNetBlock, HybridModel
from library.data_utils import get_dataloaders, engineer_features
from library.train_utils import train_model, generate_submission, set_seed


def main():
    print("Starting demonstration and verification script...")

    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    # We override the Config class attributes to use a temporary directory
    # and small subsets of data for a fast demonstration.

    WORKING_DIR = "./working/demo_execution"
    MINI_DATA_DIR = os.path.join(WORKING_DIR, "data")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Clear stale cache to prevent loading full dataset files
    if os.path.exists(CACHE_DIR):
        print(f"Clearing stale cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)

    os.makedirs(MINI_DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Creating mini-datasets for rapid testing...")
    # Load a small chunk of the actual data to ensure schema consistency
    # We use the metadata paths provided in the original Config
    df_train_full = pd.read_parquet(Config.TRAIN_PATH)
    df_val_full = pd.read_parquet(Config.VAL_PATH)
    df_test_full = pd.read_parquet(Config.TEST_PATH)

    # Create subsets (2000 train, 500 val, 500 test)
    mini_train = df_train_full.head(2000).copy()
    mini_val = df_val_full.head(500).copy()
    mini_test = df_test_full.head(500).copy()

    # Save mini parquets
    mini_train_path = os.path.join(MINI_DATA_DIR, "train.parquet")
    mini_val_path = os.path.join(MINI_DATA_DIR, "val.parquet")
    mini_test_path = os.path.join(MINI_DATA_DIR, "test.parquet")

    mini_train.to_parquet(mini_train_path, index=False)
    mini_val.to_parquet(mini_val_path, index=False)
    mini_test.to_parquet(mini_test_path, index=False)

    # Override Config attributes
    print("Overriding Config parameters...")
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path
    Config.CACHE_DIR = CACHE_DIR
    Config.SUBMISSION_DIR = SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Speed up training for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 64
    Config.EARLY_STOPPING_PATIENCE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.DEVICE = (
        "cpu"  # Force CPU for simple logic verification if GPU is busy, or leave as is.
    )
    # The library uses "cuda" if available. We'll stick to library logic.

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Verify Model Architecture Components
    # ---------------------------------------------------------
    print("\nVerifying Model Architecture Components...")

    BATCH_SIZE = 10
    INPUT_DIM = 64
    HIDDEN_DIM = 128

    # Test VectorCrossLayer
    vcl = VectorCrossLayer(INPUT_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    out = vcl(x, x)
    assert out.shape == (
        BATCH_SIZE,
        INPUT_DIM,
    ), f"VectorCrossLayer output shape mismatch: {out.shape}"
    print("  [OK] VectorCrossLayer passed.")

    # Test ResNetBlock
    res_block = ResNetBlock(HIDDEN_DIM, dropout=0.1)
    x_res = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    out_res = res_block(x_res)
    assert out_res.shape == (
        BATCH_SIZE,
        HIDDEN_DIM,
    ), f"ResNetBlock output shape mismatch: {out_res.shape}"
    print("  [OK] ResNetBlock passed.")

    # Test HybridModel
    model = HybridModel(
        input_dim=INPUT_DIM, num_classes=7, resnet_depth=2, resnet_width=HIDDEN_DIM
    )
    logits = model(x)
    assert logits.shape == (
        BATCH_SIZE,
        7,
    ), f"HybridModel output shape mismatch: {logits.shape}"
    print("  [OK] HybridModel passed.")

    # 3. Verify Feature Engineering Logic
    # ---------------------------------------------------------
    print("\nVerifying Feature Engineering...")
    # Create a dummy dataframe with required columns
    dummy_df = pd.DataFrame(
        {
            "Aspect": [0, 90, 180],
            "Horizontal_Distance_To_Hydrology": [0, 3, 4],
            "Vertical_Distance_To_Hydrology": [0, 4, 3],
            "Elevation": [100, 200, 300],
            "Horizontal_Distance_To_Roadways": [10, 20, 30],
            "Horizontal_Distance_To_Fire_Points": [5, 15, 25],
        }
    )

    processed_df = engineer_features(dummy_df)

    # Check new columns exist
    expected_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Dist_Hydro",
        "Abs_Hydro_Elev",
        "Mean_Dist_Amenities",
    ]
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing engineered feature: {col}"

    # Check calculation correctness (Spot check Euclidean Dist)
    # Row 1: sqrt(3^2 + 4^2) = 5
    assert np.isclose(
        processed_df.loc[1, "Euclidean_Dist_Hydro"], 5.0
    ), "Feature calculation error"
    print("  [OK] Feature Engineering passed.")

    # 4. Run Full Training Pipeline
    # ---------------------------------------------------------
    print("\nRunning Training Pipeline (train_model)...")

    # This will:
    # 1. Call get_dataloaders (which processes data and saves to cache)
    # 2. Initialize the model
    # 3. Train for Config.EPOCHS (2)
    trained_model, test_loader, test_ids = train_model()

    assert isinstance(
        trained_model, HybridModel
    ), "train_model did not return a HybridModel instance"
    assert len(test_ids) == 500, f"Expected 500 test IDs, got {len(test_ids)}"
    print("  [OK] Training pipeline completed successfully.")

    # 5. Generate Submission
    # ---------------------------------------------------------
    print("\nGenerating Submission...")

    generate_submission(trained_model, test_loader, test_ids)

    # Verify file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (500, 2), f"Submission shape mismatch: {df_sub.shape}"
    assert list(df_sub.columns) == ["Id", "Cover_Type"], "Submission columns mismatch"
    assert (
        df_sub["Cover_Type"].between(1, 7).all()
    ), "Predictions out of valid range (1-7)"

    print(f"  [OK] Submission generated at {Config.SUBMISSION_PATH}")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
