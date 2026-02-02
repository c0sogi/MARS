import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import warnings
from contextlib import contextmanager

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_loaders
from library.model import get_model
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import predict_fn

# Suppress warnings
warnings.filterwarnings("ignore")


# Context manager to suppress stderr (used to hide tqdm progress bars from library)
@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def main():
    print("=== Starting Cassava Leaf Disease Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1/5] Initializing Configuration...")
    # Initialize config in debug mode for speed
    config = Config(debug=True)

    # Override parameters for an extremely fast demonstration
    config.subset_size = 32  # Use only 32 images
    config.batch_size = 8  # Small batch size
    config.n_folds = 2  # Setup for 2 folds, but we will only run fold 0
    config.phase1_epochs = 1  # Only 1 epoch
    config.phase2_epochs = 0  # Skip phase 2
    config.working_dir = "./working/demo_run"
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)

    print(f"    Working Directory: {config.working_dir}")
    print(f"    Device: {config.device}")
    print(f"    Subset Size: {config.subset_size}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Data Loading...")

    # Get loaders for Phase 1, Fold 0
    train_loader, val_loader, _ = get_loaders(config, phase=1, fold_idx=0)

    # Fetch a single batch to verify shapes
    try:
        images, labels = next(iter(train_loader))

        # Verify Image Shape: (Batch_Size, 3, Height, Width)
        expected_img_shape = (
            config.batch_size,
            3,
            config.phase1_image_size,
            config.phase1_image_size,
        )
        assert (
            images.shape == expected_img_shape
        ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

        # Verify Label Shape: (Batch_Size,)
        expected_lbl_shape = (config.batch_size,)
        assert (
            labels.shape == expected_lbl_shape
        ), f"Label shape mismatch. Expected {expected_lbl_shape}, got {labels.shape}"

        print("    Data shapes verified successfully.")

    except StopIteration:
        raise AssertionError("DataLoader is empty!")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture...")

    model = get_model(config)
    model.to(config.device)

    # Verify model is a valid PyTorch module
    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"

    # Test Forward Pass with dummy data
    with torch.no_grad():
        dummy_input = torch.randn(
            2, 3, config.phase1_image_size, config.phase1_image_size
        ).to(config.device)
        output = model(dummy_input)

        # Verify Output Shape: (Batch_Size, Num_Classes)
        expected_out_shape = (2, config.num_classes)
        assert (
            output.shape == expected_out_shape
        ), f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"

    print("    Model forward pass verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Train for one epoch
    # Note: Using suppress_stderr might hide errors if they print to stderr,
    # but we want to see errors, just not progress bars.
    # Since train_one_epoch doesn't use tqdm, we run it directly.
    train_loss = train_one_epoch(
        model, train_loader, optimizer, config.device, epoch=0, config=config
    )

    print(f"    Training Loss: {train_loss:.4f}")
    assert not pd.isna(train_loss), "Training loss resulted in NaN"

    # Validate
    val_loss, val_acc = valid_one_epoch(model, val_loader, config.device, config)

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Accuracy: {val_acc:.2f}%")
    assert not pd.isna(val_loss), "Validation loss resulted in NaN"

    # Save the model weights to simulate a completed fold training
    # The inference engine expects 'fold_0_best.pth'
    model_save_path = os.path.join(config.working_dir, "fold_0_best.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"    Model weights saved to {model_save_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[5/5] Running Inference Pipeline...")

    # Configure for inference
    # We only trained fold 0, so we limit n_folds to 1 for the inference loop
    config.n_folds = 1

    # Run inference (suppressing tqdm output from library/inference.py)
    print("    Generating predictions...")
    with suppress_stderr():
        predict_fn(config)

    # Verify Submission File
    if not os.path.exists(config.submission_path):
        raise FileNotFoundError(
            f"Submission file was not created at {config.submission_path}"
        )

    df_submission = pd.read_csv(config.submission_path)

    # Check content
    print(f"    Submission file created with {len(df_submission)} rows.")

    # Verify row count matches subset size (since debug=True truncates test set too)
    assert (
        len(df_submission) == config.subset_size
    ), f"Submission row count mismatch. Expected {config.subset_size}, got {len(df_submission)}"

    # Verify columns
    expected_cols = ["image_id", "label"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_submission.columns)}"

    print("    Submission format verified.")
    print("\n=== Demo Completed Successfully ===")
    print("Sample Submission:")
    print(df_submission.head())


if __name__ == "__main__":
    main()
