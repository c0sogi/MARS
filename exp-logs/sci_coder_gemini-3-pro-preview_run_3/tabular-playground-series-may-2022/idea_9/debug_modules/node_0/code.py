import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config, set_seed
from library.data_utils import get_data_loaders
from library.model import InputInjectedFunnelMLP
from library.train_eval import run_training


def setup_demo_config():
    """
    Overrides Config parameters for a quick demonstration run.
    """
    print("Setting up demo configuration...")

    # Use a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 512
    Config.HIDDEN_DIMS = [64, 32, 16]  # Smaller model for speed
    Config.EMBEDDING_DIM = 4

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")


def verify_data_pipeline():
    """
    Verifies the data loading and processing pipeline.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Use a small subset for debugging/verification
    debug_limit = 2000

    # 1. Get Data Loaders
    print(f"Loading data with debug_limit={debug_limit}...")
    train_loader, val_loader, test_loader, vocab_sizes, cont_dim = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing from scratch for demo
        debug_limit=debug_limit,
    )

    # 2. Assertions
    print("Running assertions on DataLoaders...")

    # Check if loaders are not empty
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Check batch structure
    x_cont, x_cat, y = next(iter(train_loader))

    # Check dimensions
    assert x_cont.dim() == 2, f"Expected 2D continuous input, got {x_cont.dim()}"
    assert x_cat.dim() == 2, f"Expected 2D categorical input, got {x_cat.dim()}"
    assert y.dim() == 2, f"Expected 2D target, got {y.dim()}"
    assert (
        x_cont.shape[1] == cont_dim
    ), f"Mismatch in continuous features: {x_cont.shape[1]} vs {cont_dim}"
    assert x_cat.shape[1] == len(
        vocab_sizes
    ), f"Mismatch in categorical features: {x_cat.shape[1]} vs {len(vocab_sizes)}"

    print(f"Continuous Dim: {cont_dim}")
    print(f"Vocab Sizes: {vocab_sizes}")
    print("Data Pipeline verification passed.")

    return vocab_sizes, cont_dim


def verify_model_architecture(vocab_sizes, cont_dim):
    """
    Verifies the model instantiation and forward pass.
    """
    print("\n=== Verifying Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = InputInjectedFunnelMLP(
        cont_dim=cont_dim,
        vocab_sizes=vocab_sizes,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT,
    ).to(device)

    print("Model instantiated successfully.")

    # Create dummy input
    batch_size = 4
    dummy_cont = torch.randn(batch_size, cont_dim).to(device)
    dummy_cat = torch.zeros(batch_size, len(vocab_sizes), dtype=torch.long).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Check output
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    print(f"Forward pass successful. Output shape: {output.shape}")
    print("Model Architecture verification passed.")


def verify_full_training_cycle():
    """
    Runs the full training loop using the library function and validates output.
    """
    print("\n=== Verifying Full Training Cycle ===")

    # Limit samples for speed
    debug_limit = 5000

    # Run training
    # Note: run_training handles data loading, model init, training loop, and submission generation
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Use cache if available (likely from verify_data_pipeline step)
        debug_limit=debug_limit,
    )

    # Verify Outputs
    print("\nChecking generated files...")

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    submission_path = Config.SUBMISSION_PATH

    # Check Model File
    if os.path.exists(best_model_path):
        print(f"[OK] Model file found at: {best_model_path}")
        file_size = os.path.getsize(best_model_path)
        print(f"     Model size: {file_size / 1024:.2f} KB")
        assert file_size > 0, "Model file is empty"
    else:
        raise FileNotFoundError(f"Model file not found at {best_model_path}")

    # Check Submission File
    if os.path.exists(submission_path):
        print(f"[OK] Submission file found at: {submission_path}")

        df_sub = pd.read_csv(submission_path)
        print(f"     Submission shape: {df_sub.shape}")

        # Validate Submission Content
        assert (
            "id" in df_sub.columns and "target" in df_sub.columns
        ), "Submission missing required columns"
        assert (
            df_sub.shape[0] == 100000
        ), f"Expected 100,000 predictions, got {df_sub.shape[0]}"
        assert (
            df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
        ), "Predictions out of probability range [0, 1]"
        assert df_sub["id"].nunique() == 100000, "Duplicate IDs found in submission"

        print("Submission file content valid.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("Full Training Cycle verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Verify Data Pipeline
        vocab_sizes, cont_dim = verify_data_pipeline()

        # 3. Verify Model Logic
        verify_model_architecture(vocab_sizes, cont_dim)

        # 4. Verify Integration (Training & Inference)
        verify_full_training_cycle()

        print("\nAll demonstrations and verifications completed successfully.")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
