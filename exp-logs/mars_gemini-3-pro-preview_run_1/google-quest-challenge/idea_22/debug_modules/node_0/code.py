import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader
import transformers

# Suppress warnings and progress bars for clean output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.utils import seed_everything
from library.dataset import StackExchangeDataset, collate_fn
from library.model import PartitionedPoolingDualEncoder
from library.trainer import Trainer


def check_dataset_and_loader():
    """
    Demonstrates and verifies Dataset loading and Collate function.
    """
    print("\n=== 1. Verifying Dataset and DataLoader ===")

    # Initialize dataset in debug mode
    # num_debug_samples defaults to 100 in the class definition
    cache_dir = "./working/demo_check_data"
    ds = StackExchangeDataset(
        split="train",
        debug=True,
        cache_dir=cache_dir,
        load_cached_data=False,  # Force reload to test processing logic
    )

    # 1. Check single item structure
    item = ds[0]
    expected_keys = [
        "qa_id",
        "input_ids_q",
        "attention_mask_q",
        "title_mask",
        "body_mask",
        "input_ids_a",
        "attention_mask_a",
        "labels",
    ]

    for key in expected_keys:
        assert key in item, f"Missing key {key} in dataset item"

    print(f"Dataset item keys verified: {list(item.keys())}")

    # Verify label shape (should be 30 targets)
    assert item["labels"].shape == (
        30,
    ), f"Expected label shape (30,), got {item['labels'].shape}"

    # 2. Check DataLoader and Collate
    batch_size = 4
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    batch = next(iter(loader))

    # Verify batch sizes
    assert len(batch["qa_id"]) == batch_size, "Incorrect batch size for qa_id"
    assert (
        batch["input_ids_q"].shape[0] == batch_size
    ), "Incorrect batch dim for input_ids_q"
    assert batch["labels"].shape == (batch_size, 30), "Incorrect batch dim for labels"

    # Verify padding logic (masks should be 0 or 1)
    # attention_mask should be binary
    unique_mask_vals = torch.unique(batch["attention_mask_q"])
    assert all(
        val in [0, 1] for val in unique_mask_vals
    ), "Attention mask contains non-binary values"

    print("DataLoader and Collate function verified successfully.")
    return batch


def check_model_forward(batch):
    """
    Demonstrates and verifies Model initialization and forward pass.
    """
    print("\n=== 2. Verifying Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartitionedPoolingDualEncoder(model_name="roberta-base")
    model.to(device)
    model.eval()

    # Move batch to device
    inputs = {
        "input_ids_q": batch["input_ids_q"].to(device),
        "attention_mask_q": batch["attention_mask_q"].to(device),
        "title_mask": batch["title_mask"].to(device),
        "body_mask": batch["body_mask"].to(device),
        "input_ids_a": batch["input_ids_a"].to(device),
        "attention_mask_a": batch["attention_mask_a"].to(device),
    }

    with torch.no_grad():
        logits = model(**inputs)

    # Verify output shape
    batch_size = inputs["input_ids_q"].shape[0]
    expected_shape = (batch_size, 30)
    assert (
        logits.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {logits.shape}"

    # Verify values are finite (no NaNs)
    assert torch.isfinite(logits).all(), "Model output contains NaNs or Infs"

    print(f"Model forward pass successful. Output shape: {logits.shape}")


def check_training_pipeline():
    """
    Demonstrates and verifies the Trainer class and full training loop.
    """
    print("\n=== 3. Verifying Training Pipeline (Trainer) ===")

    working_dir = "./working/demo_trainer_run"

    # Clean up previous run if exists
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    # Initialize Trainer with minimal settings for speed
    trainer = Trainer(
        model_name="roberta-base",
        epochs=1,  # Phantom scheduling epochs
        stop_epoch=1,  # Stop after 1 epoch
        batch_size=4,  # Small batch size
        accum_steps=1,
        working_dir=working_dir,
        debug=True,  # Use debug subset (100 samples)
    )

    print("Starting training run...")
    trainer.train()

    # Verify outputs
    best_model_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    assert os.path.exists(best_model_path), "best_model.pth was not created"
    assert os.path.exists(submission_path), "submission.csv was not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    assert "qa_id" in sub_df.columns, "qa_id column missing in submission"
    assert (
        sub_df.shape[1] == 31
    ), f"Expected 31 columns in submission, got {sub_df.shape[1]}"

    # Check if predictions are in [0, 1]
    target_cols = [c for c in sub_df.columns if c != "qa_id"]
    preds = sub_df[target_cols].values
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Predictions out of range [0, 1]"

    print("Training pipeline verified successfully. Files generated.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Dataset Check
        batch = check_dataset_and_loader()

        # 2. Model Check
        check_model_forward(batch)

        # 3. Trainer Check
        check_training_pipeline()

        print("\nAll demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
