import os
import sys
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.data_processing import process_data, get_dataloaders
from library.training import train_model, generate_submission
from library.model import SNNModel


def create_mini_datasets(n_samples=2000):
    """
    Creates smaller versions of the datasets for rapid demonstration purposes.
    """
    print(f"Creating mini-datasets with {n_samples} samples each...")

    # Define paths for mini datasets
    mini_dir = os.path.join(Config.WORKING_DIR, "mini_data")
    os.makedirs(mini_dir, exist_ok=True)

    mini_train_path = os.path.join(mini_dir, "train.csv")
    mini_val_path = os.path.join(mini_dir, "val.csv")
    mini_test_path = os.path.join(mini_dir, "test.csv")

    # Read original metadata (first n_samples only)
    # We use the paths defined in the original Config before we overwrite them
    train_df = pd.read_csv(Config.TRAIN_META, nrows=n_samples)
    val_df = pd.read_csv(Config.VAL_META, nrows=n_samples)
    test_df = pd.read_csv(Config.TEST_META, nrows=n_samples)

    # Save mini datasets
    train_df.to_csv(mini_train_path, index=False)
    val_df.to_csv(mini_val_path, index=False)
    test_df.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def run_demo():
    print("Initializing demonstration...")

    # 1. Configuration Override for Speed and Isolation
    # We modify the Config class attributes directly to control the library behavior
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Create and point to mini datasets
    mini_train, mini_val, mini_test = create_mini_datasets(n_samples=2000)
    Config.TRAIN_META = mini_train
    Config.VAL_META = mini_val
    Config.TEST_META = mini_test

    # Update output paths
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 512
    Config.HIDDEN_LAYERS = [64, 32]  # Smaller model
    Config.LOAD_CACHED_DATA = False  # Force processing to demonstrate logic

    # 2. Data Processing
    print("\n--- Step 1: Data Processing ---")
    train_df, val_df, test_df, meta = process_data(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # Validation of Data Processing
    print("Validating processed data...")
    assert len(train_df) == 2000, f"Expected 2000 train samples, got {len(train_df)}"
    assert len(test_df) == 2000, f"Expected 2000 test samples, got {len(test_df)}"

    # Check meta dictionary
    assert "cont_cols" in meta
    assert "cat_cols" in meta
    assert "vocab_sizes" in meta

    # Check feature engineering (f_27 decomposition)
    # f_27 is split into 10 chars + unique_count, plus original f_27 might be dropped or kept depending on implementation
    # The library implementation keeps f_27 but adds f_27_0...f_27_9 and unique_character_count
    # The cont_cols list in meta excludes f_27, cat_cols includes f_27_i
    expected_cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]
    assert meta["cat_cols"] == expected_cat_cols, "Categorical columns mismatch"
    assert (
        "unique_character_count" in meta["cont_cols"]
    ), "Feature engineering missing unique_character_count"

    print("Data processing validation passed.")

    # 3. Data Loaders
    print("\n--- Step 2: Data Loaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df,
        val_df,
        test_df,
        meta,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple script execution to avoid multiprocessing overhead in demo
    )

    # Validate Loader Output
    print("Validating DataLoader batch...")
    x_cont, x_cat, y = next(iter(train_loader))

    # Check shapes
    # x_cont: (batch_size, num_cont)
    # x_cat: (batch_size, num_cat)
    # y: (batch_size,)
    assert x_cont.shape[0] == Config.BATCH_SIZE
    assert x_cat.shape[0] == Config.BATCH_SIZE
    assert x_cont.shape[1] == len(meta["cont_cols"])
    assert x_cat.shape[1] == len(meta["cat_cols"])

    print(f"Batch shapes verified: Cont={x_cont.shape}, Cat={x_cat.shape}")

    # 4. Model Instantiation (Manual Check)
    print("\n--- Step 3: Model Instantiation ---")
    model = SNNModel(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    )
    print(model)
    # Pass a dummy batch to verify forward pass
    with torch.no_grad():
        logits = model(x_cont, x_cat)
        assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("Model forward pass successful.")

    # 5. Training
    print("\n--- Step 4: Training ---")
    # train_model handles the loop, validation, and saving best model
    best_auc = train_model(train_loader, val_loader, meta)

    print(f"Training complete. Best AUC: {best_auc}")
    assert 0.0 <= best_auc <= 1.0, "AUC score out of range"
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved"

    # 6. Inference / Submission
    print("\n--- Step 5: Inference ---")
    generate_submission(test_loader, meta)

    # Validate Submission File
    print("Validating submission file...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape (should match mini test set size)
    assert sub_df.shape == (
        2000,
        2,
    ), f"Submission shape mismatch. Expected (2000, 2), got {sub_df.shape}"

    # Check columns
    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch"

    # Check value ranges
    assert sub_df["target"].min() >= 0.0, "Probabilities should be >= 0"
    assert sub_df["target"].max() <= 1.0, "Probabilities should be <= 1"

    print("Submission validation passed.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n[FAIL] Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
