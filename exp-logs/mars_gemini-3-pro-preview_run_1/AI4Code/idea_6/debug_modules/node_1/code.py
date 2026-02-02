import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.data_preprocessor import DataPreprocessor
from library.dataset import NotebookEmbeddingDataset
from library.model import DualContextAnchorNetwork
from library.train import train_model
from library.inference import predict_and_sort


def setup_demo_config():
    """
    Overrides Config parameters to ensure the demo runs quickly and
    uses a separate working directory.
    """
    print(">>> Setting up Demo Configuration...")

    # Use a specific directory for this demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.setup()  # Create the directory

    # Update paths to point to the demo working directory
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_features.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_features.parquet")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set Debug parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small sample for demonstration
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")


def run_preprocessing_demo():
    """
    Demonstrates data preprocessing and feature caching.
    """
    print("\n>>> Running Data Preprocessing...")

    # Initialize Preprocessor
    preprocessor = DataPreprocessor()

    # Run processing (force reload to ensure we generate new debug files)
    preprocessor.run(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.TRAIN_CACHE_PATH), "Train parquet not created."
    assert os.path.exists(Config.VAL_CACHE_PATH), "Val parquet not created."
    assert os.path.exists(Config.TEST_CACHE_PATH), "Test parquet not created."

    # Check content of one file
    df = pd.read_parquet(Config.TRAIN_CACHE_PATH)
    print(f"Train features shape: {df.shape}")
    assert (
        len(df) <= Config.DEBUG_SAMPLE_SIZE
    ), "Preprocessing did not respect debug sample size."
    assert "code_embeddings" in df.columns, "Missing code_embeddings column."
    assert "markdown_embeddings" in df.columns, "Missing markdown_embeddings column."
    print("Preprocessing verification passed.")


def run_dataset_and_model_demo():
    """
    Demonstrates dataset loading and model forward pass logic.
    """
    print("\n>>> Verifying Dataset and Model Logic...")

    # 1. Dataset Loading
    train_dataset = NotebookEmbeddingDataset(
        split="train", max_size=Config.DEBUG_SAMPLE_SIZE
    )
    assert len(train_dataset) > 0, "Dataset is empty."

    # Get a single item
    item = train_dataset[0]
    print(f"Sample Item Keys: {list(item.keys())}")

    # Check shapes
    code_dim = item["code_embeddings"].shape[1]
    md_dim = item["markdown_embeddings"].shape[1]
    assert code_dim == Config.INPUT_DIM, f"Incorrect code embedding dim: {code_dim}"
    assert md_dim == Config.INPUT_DIM, f"Incorrect md embedding dim: {md_dim}"

    # 2. Collate Function
    batch_list = [
        train_dataset[i] for i in range(min(len(train_dataset), Config.BATCH_SIZE))
    ]
    batch = NotebookEmbeddingDataset.collate_fn(batch_list)

    # 3. Model Forward Pass
    device = Config.DEVICE
    model = DualContextAnchorNetwork().to(device)
    model.eval()

    # Move batch to device
    code_embs = batch["code_embeddings"].to(device)
    md_embs = batch["markdown_embeddings"].to(device)
    code_mask = batch["code_mask"].to(device)
    md_mask = batch["markdown_mask"].to(device)

    with torch.no_grad():
        logits = model(code_embs, md_embs, code_mask, md_mask)

    # Logits shape should be (Batch, N_md, N_code + 1)
    # N_code + 1 accounts for the End-of-Notebook token
    B, N_md, N_classes = logits.shape
    B_in, N_code_in, _ = code_embs.shape

    print(f"Logits Shape: {logits.shape}")

    assert B == len(batch_list), "Batch size mismatch in output."
    assert (
        N_classes == N_code_in + 1
    ), f"Expected {N_code_in + 1} classes, got {N_classes}"

    print("Dataset and Model logic verification passed.")


def run_training_demo():
    """
    Demonstrates the training loop.
    """
    print("\n>>> Running Training Loop...")

    # Ensure no previous model exists
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)

    # Run training
    train_model()

    # Verify model artifact
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print(f"Training finished. Model saved at {Config.MODEL_PATH}")


def run_inference_demo():
    """
    Demonstrates inference and submission generation.
    """
    print("\n>>> Running Inference...")

    # Ensure no previous submission exists
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Run inference
    predict_and_sort()

    # Verify submission artifact
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Incorrect submission columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check format of cell_order (space delimited string)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order is not a string."
    assert len(sample_order.split()) > 0, "cell_order seems empty."

    print("Inference verification passed.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Preprocessing
        run_preprocessing_demo()

        # 3. Logic Verification
        run_dataset_and_model_demo()

        # 4. Training
        run_training_demo()

        # 5. Inference
        run_inference_demo()

        print("\n>>> All demo steps completed successfully!")

    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
