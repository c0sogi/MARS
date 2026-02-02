import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import get_model
from library.training import run_training
from library.inference import predict_and_submit, generate_pseudo_labels, run_inference


def main():
    print("Initializing Demo...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config defaults for a fast, offline demonstration
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Speed optimizations
    Config.PRETRAINED = False  # Avoid downloading weights
    Config.EPOCHS = 2  # Minimal epochs to test loop
    Config.SWA_START_EPOCH = 1  # Trigger SWA logic immediately in 2nd epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce overhead

    set_seed(Config.SEED)
    print("Configuration configured for fast demo run.")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch to verify structure
    images, labels, rec_ids = next(iter(train_loader))

    print(f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels (RGB)"
    assert images.shape[2] == Config.IMG_HEIGHT, f"Height must be {Config.IMG_HEIGHT}"
    assert images.shape[3] == Config.IMG_WIDTH, f"Width must be {Config.IMG_WIDTH}"
    assert labels.shape[1] == Config.NUM_CLASSES, "Labels must match num_classes"
    assert rec_ids.shape[0] == Config.BATCH_SIZE, "rec_ids count must match batch size"

    print("Data Loading verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Model Initialization ---")
    model = get_model(pretrained=False, num_classes=Config.NUM_CLASSES)

    # Test Forward Pass
    model.eval()
    with torch.no_grad():
        # Move sample to device
        sample_imgs = images.to(Config.DEVICE)
        outputs = model(sample_imgs)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model initialization and forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Training Loop (2 Epochs) ---")
    # This runs the full training pipeline defined in library.training
    best_auc, swa_auc = run_training(
        checkpoint_dir=Config.CHECKPOINT_DIR,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
    )

    print(f"Training Complete. Best AUC: {best_auc:.4f}, SWA AUC: {swa_auc:.4f}")

    # Assertions
    assert 0 <= best_auc <= 1, "Best AUC out of range"
    assert 0 <= swa_auc <= 1, "SWA AUC out of range"

    # Verify Checkpoints exist
    expected_files = ["model_last.pth", "model_swa.pth"]
    # model_best.pth might not exist if validation AUC never improved from 0.0 (unlikely but possible)
    # However, with initialized weights, random chance usually gives AUC ~0.5

    for fname in expected_files:
        fpath = os.path.join(Config.CHECKPOINT_DIR, fname)
        assert os.path.exists(fpath), f"Checkpoint file missing: {fname}"

    print("Training loop and checkpointing verified.")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Inference and Submission ---")

    # Use the SWA model for inference
    swa_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_swa.pth")

    # Run the high-level predict_and_submit function
    predict_and_submit(
        checkpoint_path=swa_checkpoint_path,
        output_path=Config.SUBMISSION_PATH,
        use_tta=True,  # Enable TTA to test that logic too
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Expected rows = num_test_samples * num_classes
    # Test set size is 64 (from metadata info in prompt)
    num_test_samples = len(test_loader.dataset)
    expected_rows = num_test_samples * Config.NUM_CLASSES

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"
    assert (
        df_sub["Probability"].min() >= 0 and df_sub["Probability"].max() <= 1
    ), "Probabilities out of [0, 1] range"

    print("Submission generation verified.")

    # --------------------------------------------------------------------------
    # 6. Pseudo-Label Generation Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Testing Pseudo-Label Generation ---")

    # Manually run inference to get raw arrays for pseudo-label function
    model_inf = get_model(pretrained=False)
    load_checkpoint(model_inf, swa_checkpoint_path)

    rec_ids, probs = run_inference(model_inf, test_loader)

    # Generate pseudo labels (soft)
    df_pseudo = generate_pseudo_labels(rec_ids, probs, threshold=None)

    # Assertions
    assert len(df_pseudo) == num_test_samples, "Pseudo-label DF length mismatch"
    assert "rec_id" in df_pseudo.columns, "rec_id column missing in pseudo-labels"
    assert (
        len(df_pseudo.columns) == Config.NUM_CLASSES + 1
    ), "Incorrect column count in pseudo-labels"

    # Save pseudo labels to verify file I/O
    pseudo_path = os.path.join(Config.WORKING_DIR, "demo_pseudo_labels.parquet")
    df_pseudo.to_parquet(pseudo_path)
    assert os.path.exists(pseudo_path), "Pseudo-label parquet file not saved"

    print("Pseudo-label generation verified.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
