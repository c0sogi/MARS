import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
import library.data_utils as data_utils
import library.model as model_utils
import library.train_eval as train_eval


def setup_demo_environment():
    """
    Creates a subset of the data in a working directory to ensure the
    demonstration runs quickly (seconds instead of minutes/hours).
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create subdirectories
    submission_dir = os.path.join(demo_dir, "submission")
    os.makedirs(submission_dir, exist_ok=True)

    # Load a small subset of the metadata
    # We use the existing metadata files which are guaranteed to exist
    train_full = pd.read_csv("./metadata/train.csv", nrows=1001)
    val_full = pd.read_csv("./metadata/val.csv", nrows=1001)
    test_full = pd.read_csv("./metadata/test.csv", nrows=1001)

    # Save subsets to the demo directory
    train_path = os.path.join(demo_dir, "train.csv")
    val_path = os.path.join(demo_dir, "val.csv")
    test_path = os.path.join(demo_dir, "test.csv")

    train_full.to_csv(train_path, index=False)
    val_full.to_csv(val_path, index=False)
    test_full.to_csv(test_path, index=False)

    print(f"Created subset data at {demo_dir}")
    return demo_dir, train_path, val_path, test_path, submission_dir


def patch_config(demo_dir, train_path, val_path, test_path, submission_dir):
    """
    Updates the Config class to point to the demo environment.
    Since Config attributes are static, we modify them directly.
    """
    print("Patching Config for demo execution...")

    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = submission_dir

    # Input paths
    Config.TRAIN_PATH = train_path
    Config.VAL_PATH = val_path
    Config.TEST_PATH = test_path

    # Output paths
    Config.MODEL_PATH = os.path.join(demo_dir, "best_classifier.pth")
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Cache paths (must update these as they were initialized with the old WORKING_DIR)
    Config.TRAIN_PROCESSED_PATH = os.path.join(demo_dir, "train_processed.parquet")
    Config.VAL_PROCESSED_PATH = os.path.join(demo_dir, "val_processed.parquet")
    Config.TEST_PROCESSED_PATH = os.path.join(demo_dir, "test_processed.parquet")
    Config.ENCODERS_PATH = os.path.join(demo_dir, "encoders.npy")

    # Reduce model complexity for speed
    Config.HIDDEN_LAYERS = [32, 16]
    Config.EMBEDDING_DIM = 4
    Config.BATCH_SIZE = 32
    Config.EPOCHS = 2  # Minimum 2 to test loop and potentially early stopping logic
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data


def verify_data_utils():
    """
    Verifies that data loading and processing logic works correctly.
    """
    print("\n=== Verifying Data Utils ===")

    # Force reload to test processing logic
    train_loader, val_loader, test_loader, vocab_sizes = data_utils.get_dataloaders(
        load_cached_data=False
    )

    # Assertions
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Val loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    # Check batch structure
    batch = next(iter(train_loader))
    assert "cont_features" in batch
    assert "cat_features" in batch
    assert "target" in batch
    assert "id" in batch

    # Check shapes
    # Continuous features: 27 original + 1 (f_28) + 1 (unique_char_count) = 29
    expected_cont_dim = 29
    # Categorical features: f_29, f_30 + 10 chars = 12
    expected_cat_dim = 12

    assert (
        batch["cont_features"].shape[1] == expected_cont_dim
    ), f"Expected cont_dim {expected_cont_dim}, got {batch['cont_features'].shape[1]}"
    assert (
        batch["cat_features"].shape[1] == expected_cat_dim
    ), f"Expected cat_dim {expected_cat_dim}, got {batch['cat_features'].shape[1]}"

    print("Data Utils verification passed.")
    return train_loader, val_loader, test_loader, vocab_sizes, expected_cont_dim


def verify_model_logic(vocab_sizes, cont_dim):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n=== Verifying Model Logic ===")

    device = Config.DEVICE
    model = model_utils.GatedFunnelMLP(vocab_sizes, cont_dim).to(device)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_cont = torch.randn(batch_size, cont_dim).to(device)
    # Create dummy categorical input (indices within range)
    dummy_cat = []
    for col in Config.CAT_FEATURES:
        # Max index is vocab_sizes[col] - 1
        limit = vocab_sizes[col]
        dummy_cat.append(torch.randint(0, limit, (batch_size, 1)).to(device))
    dummy_cat = torch.cat(dummy_cat, dim=1)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Check output shape (batch_size, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"

    print("Model logic verification passed.")


def run_full_pipeline_test():
    """
    Runs the high-level training and prediction pipeline provided by library.train_eval.
    """
    print("\n=== Running Full Pipeline Test ===")

    # Run training (we already patched Config, so it uses the demo data)
    # load_cached_data=True allows it to pick up the parquet files generated
    # during verify_data_utils(), saving time.
    train_eval.run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Verify artifacts
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (
        1001,
        2,
    ), f"Expected submission shape (1001, 2), got {df_sub.shape}"
    assert list(df_sub.columns) == ["id", "target"], "Submission columns incorrect."

    # Check probability range
    assert df_sub["target"].min() >= 0.0, "Probabilities must be >= 0"
    assert df_sub["target"].max() <= 1.0, "Probabilities must be <= 1"

    print("Full pipeline test passed.")


if __name__ == "__main__":
    # 1. Setup
    demo_dir, t_path, v_path, te_path, sub_dir = setup_demo_environment()

    # 2. Configure
    patch_config(demo_dir, t_path, v_path, te_path, sub_dir)

    # 3. Verify Data Utils (and generate processed cache)
    train_loader, val_loader, test_loader, vocab_sizes, cont_dim = verify_data_utils()

    # 4. Verify Model
    verify_model_logic(vocab_sizes, cont_dim)

    # 5. Run Pipeline
    run_full_pipeline_test()

    print("\nAll demonstrations and verifications completed successfully.")
