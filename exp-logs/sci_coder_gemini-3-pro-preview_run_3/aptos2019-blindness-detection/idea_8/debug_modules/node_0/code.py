import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DRModel
from library.engine import train_one_epoch, evaluate


def run_demo():
    print("Starting Diabetic Retinopathy Classification Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration
    Config.seed = 42
    Config.debug = True  # Uses a small subset (100 train, 50 val, 20 test)
    Config.stage1_image_size = 224  # Smaller image size for speed
    Config.stage1_batch_size = 8
    Config.num_workers = 2

    # Update paths to a temporary demo directory
    Config.working_dir = "./working/demo_execution"
    Config.models_dir = os.path.join(Config.working_dir, "models")
    Config.predictions_dir = os.path.join(Config.working_dir, "predictions")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")

    # Re-run setup to create these new directories
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.seed)

    print(f"Device: {Config.device}")
    print(f"Debug Mode: {Config.debug}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[Step 2] Initializing DataLoaders...")
    # Force load_cached_data=False to demonstrate processing logic
    train_loader, val_loader, test_loader = get_dataloaders(
        image_size=Config.stage1_image_size,
        batch_size=Config.stage1_batch_size,
        load_cached_data=False,
    )

    # Verification: Check batch structure
    try:
        images, labels = next(iter(train_loader))
        print(f"  Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions
        expected_shape = (
            Config.stage1_batch_size,
            3,
            Config.stage1_image_size,
            Config.stage1_image_size,
        )
        if images.shape != expected_shape:
            raise AssertionError(
                f"Image batch shape mismatch. Expected {expected_shape}, got {images.shape}"
            )

        if labels.shape != (Config.stage1_batch_size,):
            raise AssertionError(
                f"Label batch shape mismatch. Expected {(Config.stage1_batch_size,)}, got {labels.shape}"
            )

        print("  Data Loading Verification: PASSED")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # ==========================================
    # 3. Model Instantiation
    # ==========================================
    print("\n[Step 3] Initializing Model...")
    # Use resnet18 and pretrained=False for speed/no-download
    model = DRModel(model_name="resnet18", pretrained=False)
    model.to(Config.device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_input = images.to(Config.device)
        output = model(dummy_input)

        print(f"  Output Shape: {output.shape}")
        if output.shape != (Config.stage1_batch_size, 1):
            raise AssertionError(
                f"Model output shape mismatch. Expected {(Config.stage1_batch_size, 1)}, got {output.shape}"
            )

    print("  Model Verification: PASSED")

    # ==========================================
    # 4. Training Loop (1 Epoch)
    # ==========================================
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for one epoch
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=Config.device,
        epoch=1,
        accum_iter=1,
    )

    print(f"  Training Loss: {train_loss:.4f}")
    if not np.isfinite(train_loss):
        raise AssertionError("Training loss is NaN or Infinite.")
    print("  Training Verification: PASSED")

    # ==========================================
    # 5. Evaluation
    # ==========================================
    print("\n[Step 5] Running Evaluation...")
    val_loss, val_qwk = evaluate(model, val_loader, Config.device)

    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation QWK: {val_qwk:.4f}")

    # QWK should be between -1 and 1 (usually > 0 for a trained model, but random init might be low)
    if not (-1.0 <= val_qwk <= 1.0):
        raise AssertionError(
            f"QWK score {val_qwk} is out of theoretical bounds [-1, 1]."
        )
    print("  Evaluation Verification: PASSED")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[Step 6] Generating Submission...")
    model.eval()
    test_preds = []
    test_ids = []

    # We need to get IDs. In the provided library, the dataset doesn't return IDs,
    # so we read the metadata directly to map predictions back to IDs.
    df_test = pd.read_csv(Config.test_metadata_path)
    if Config.debug:
        df_test = df_test.head(20)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.device)
            outputs = model(images)
            outputs = outputs.view(-1).cpu().numpy()

            # Post-process: Clip and Round
            preds = np.round(np.clip(outputs, 0, 4)).astype(int)
            test_preds.extend(preds)

    # Verify length matches
    if len(test_preds) != len(df_test):
        raise AssertionError(
            f"Number of predictions ({len(test_preds)}) does not match test set size ({len(df_test)})."
        )

    # Create submission DataFrame
    submission = pd.DataFrame({"id_code": df_test["id_code"], "diagnosis": test_preds})

    # Save submission
    submission_path = os.path.join(Config.working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"  Submission saved to {submission_path}")
    print(f"  Submission Head:\n{submission.head()}")
    print("  Inference Verification: PASSED")

    # ==========================================
    # 7. Cleanup
    # ==========================================
    print("\n[Step 7] Cleaning up...")
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
        print(f"  Removed temporary directory: {Config.working_dir}")

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
