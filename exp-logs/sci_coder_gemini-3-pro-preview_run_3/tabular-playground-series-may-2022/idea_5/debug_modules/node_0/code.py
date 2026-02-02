import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
import string
import random

# Import library modules
# We import Config first to patch it before other modules might use it (though they access it at runtime)
import library.config
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import GlobalPreprocessor, ManufacturingDataset
from library.model import LNGatedFunnelNet
from library.engine import run_engine


def generate_dummy_data(num_rows, is_test=False):
    """Generates a dummy dataframe matching the competition schema."""
    data = {
        "id": np.arange(num_rows),
    }

    # Continuous features f_00 to f_26, and f_28
    # Note: f_27 is the string feature
    for i in range(29):
        if i == 27:
            continue
        col_name = f"f_{i:02d}"
        data[col_name] = np.random.randn(num_rows)

    # Categorical integer features
    data["f_29"] = np.random.randint(0, 2, size=num_rows)
    data["f_30"] = np.random.randint(0, 3, size=num_rows)

    # String feature f_27 (length 10)
    # Generate random strings like 'ABACAD...'
    chars = list(string.ascii_uppercase[:10])  # Use subset for lower cardinality
    f_27_list = []
    for _ in range(num_rows):
        s = "".join(np.random.choice(chars, 10))
        f_27_list.append(s)
    data["f_27"] = f_27_list

    # Target (only for train/val)
    if not is_test:
        data["target"] = np.random.randint(0, 2, size=num_rows)

    # Source path (required by metadata schema, though not strictly used by loader logic provided)
    data["source_path"] = "dummy.csv"

    return pd.DataFrame(data)


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Directories and Config
    demo_dir = os.path.join("working", "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Patch Config to use demo directory and small settings
    # This ensures the library code uses our dummy data and runs fast
    Config.WORKING_DIR = demo_dir
    Config.INPUT_DIR = demo_dir  # Not strictly used by loader if we override paths
    Config.METADATA_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    Config.TRAIN_PATH = os.path.join(demo_dir, "train.csv")
    Config.VAL_PATH = os.path.join(demo_dir, "val.csv")
    Config.TEST_PATH = os.path.join(demo_dir, "test.csv")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller model for speed
    Config.EMBEDDING_DIM = 8

    print(f"Config patched. Working dir: {Config.WORKING_DIR}")

    # 2. Generate Dummy Data
    print("Generating synthetic data...")
    df_train = generate_dummy_data(500, is_test=False)
    df_val = generate_dummy_data(100, is_test=False)
    df_test = generate_dummy_data(100, is_test=True)

    df_train.to_csv(Config.TRAIN_PATH, index=False)
    df_val.to_csv(Config.VAL_PATH, index=False)
    df_test.to_csv(Config.TEST_PATH, index=False)

    print("Synthetic data saved.")

    # 3. Verify Utils
    print("\n--- Verifying Utils ---")
    seed_everything(42)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = compute_auc(y_true, y_pred)
    print(f"Computed AUC: {auc}")
    assert 0 <= auc <= 1, "AUC calculation is out of bounds"

    # 4. Verify Data Processing (GlobalPreprocessor)
    print("\n--- Verifying GlobalPreprocessor ---")
    preprocessor = GlobalPreprocessor()
    # Force process from scratch
    data, meta = preprocessor.process_data(load_cached_data=False)

    print("Data keys:", data.keys())
    print("Meta info:", meta)

    # Assertions for data shapes
    assert data["train_cont"].shape[0] == 500
    assert data["val_cont"].shape[0] == 100
    assert data["test_cont"].shape[0] == 100

    # Check f_27 decomposition:
    # Original columns: 28 continuous + 2 categorical (f29, f30) + 1 string (f27)
    # Processed:
    #   Categorical: 10 chars from f27 + f29 + f30 = 12 categorical features
    #   Continuous: 28 original continuous + 1 unique_char_count = 29 continuous features
    expected_cat_feats = 12
    expected_cont_feats = 29

    assert (
        data["train_cat"].shape[1] == expected_cat_feats
    ), f"Expected {expected_cat_feats} cat features, got {data['train_cat'].shape[1]}"
    assert (
        data["train_cont"].shape[1] == expected_cont_feats
    ), f"Expected {expected_cont_feats} cont features, got {data['train_cont'].shape[1]}"
    assert meta["num_cont"] == expected_cont_feats
    assert len(meta["cat_cardinalities"]) == expected_cat_feats

    # 5. Verify Dataset and Model
    print("\n--- Verifying Model & Dataset ---")
    dataset = ManufacturingDataset(
        data["train_cont"], data["train_cat"], data["train_y"]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Get one batch
    x_cont_batch, x_cat_batch, y_batch = next(iter(loader))

    model = LNGatedFunnelNet(
        num_cont=meta["num_cont"],
        cat_cardinalities=meta["cat_cardinalities"],
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout=Config.DROPOUT_RATE,
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_cont_batch, x_cat_batch)

    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model forward pass successful.")

    # 6. Verify Engine (End-to-End Run)
    print("\n--- Verifying Engine (Run Pipeline) ---")
    # We use the patched Config settings (2 epochs, small batch)
    run_engine(
        load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Load submission and check shape
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    assert (
        len(sub_df) == 100
    ), "Submission should have 100 rows (matching dummy test set)"
    assert "id" in sub_df.columns and "target" in sub_df.columns

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
