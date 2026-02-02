import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, create_model_soup
from library.dataset import get_dataloaders, get_test_loader
from library.model import get_model
from library.engine import train_fold, predict_with_tta


def run_demo():
    print("=== Starting Dog Breed Classification Library Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demo execution...")

    # Modify Config attributes to run a minimal version of the pipeline
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.EPOCHS = 2  # 1 epoch won't trigger soup if start > 0
    Config.BATCH_SIZE = 8  # Small batch for the small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.SOUP_EPOCH_START = 0  # Save checkpoints from the start
    Config.OUTPUT_DIR = "./working/demo_run"

    # Ensure output directory exists
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=2, BATCH_SIZE=8")

    # ---------------------------------------------------------
    # 2. Dataset and DataLoader Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying DataLoaders...")

    # Get loaders for Fold 0
    train_loader, val_loader, classes = get_dataloaders(
        fold_idx=0, load_cached_data=False
    )
    test_loader = get_test_loader()

    # Assertions
    assert len(classes) == 120, f"Expected 120 classes, got {len(classes)}"

    # Check Train Batch
    images, labels = next(iter(train_loader))
    print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")
    assert images.shape == (Config.BATCH_SIZE, 3, 224, 224)
    assert labels.shape == (Config.BATCH_SIZE,)

    # Check Test Batch
    test_images, test_ids = next(iter(test_loader))
    print(f"Test Batch Shape: Images {test_images.shape}, IDs length {len(test_ids)}")
    assert test_images.shape[1:] == (3, 224, 224)

    print("DataLoaders verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Architecture and Freezing Logic Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture and Freezing Logic...")

    device = Config.DEVICE
    model = get_model(
        device=device, pretrained=False
    )  # Pretrained=False for speed in initialization

    # Test Forward Pass
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, 224, 224).to(device)
    output = model(dummy_input)
    assert output.shape == (
        Config.BATCH_SIZE,
        120,
    ), f"Output shape mismatch: {output.shape}"

    # Test Freezing
    model.freeze_backbone()
    # Check that backbone is frozen (requires_grad=False) but head is not
    # Note: implementation of freeze_backbone depends on model structure.
    # We check if at least some params are frozen and some are not.
    requires_grad_flags = [p.requires_grad for p in model.parameters()]
    assert any(requires_grad_flags) and not all(
        requires_grad_flags
    ), "Freezing logic failed: All params are either frozen or unfrozen."

    # Test Unfreezing
    model.unfreeze_backbone()
    assert all(
        [p.requires_grad for p in model.parameters()]
    ), "Unfreezing logic failed: Not all params are trainable."

    print("Model logic verified successfully.")

    # ---------------------------------------------------------
    # 4. Training Engine Simulation (Fold 0)
    # ---------------------------------------------------------
    print("\n[4] Simulating Training for Fold 0...")

    # train_fold runs the loop and saves checkpoints
    # It returns a list of paths to the saved checkpoints
    soup_checkpoints = train_fold(fold_idx=0)

    print(f"Training complete. Checkpoints generated: {len(soup_checkpoints)}")
    assert len(soup_checkpoints) > 0, "No checkpoints were generated during training."
    for ckpt in soup_checkpoints:
        assert os.path.exists(ckpt), f"Checkpoint file missing: {ckpt}"

    # ---------------------------------------------------------
    # 5. Model Soup Creation
    # ---------------------------------------------------------
    print("\n[5] Creating Model Soup from checkpoints...")

    soup_state_dict = create_model_soup(soup_checkpoints, device=device)

    # Verify the soup state dict structure matches the model
    model_keys = set(model.state_dict().keys())
    soup_keys = set(soup_state_dict.keys())

    # Check key overlap (exact match expected)
    assert (
        model_keys == soup_keys
    ), "Model Soup state_dict keys do not match model architecture."
    print("Model Soup created and verified.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Running Inference with Model Soup...")

    # Load soup weights
    model.load_state_dict(soup_state_dict)
    model.to(device)

    # Predict
    preds = predict_with_tta(model, test_loader, device)

    # Verify predictions
    n_test_samples = len(pd.read_csv(Config.TEST_CSV).head(Config.DEBUG_SAMPLE_SIZE))
    assert preds.shape == (
        n_test_samples,
        120,
    ), f"Prediction shape mismatch. Expected ({n_test_samples}, 120), got {preds.shape}"

    # Create Submission
    df_test = pd.read_csv(Config.TEST_CSV).head(Config.DEBUG_SAMPLE_SIZE)
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, "id", df_test["id"])

    submission_path = os.path.join(Config.OUTPUT_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Inference complete. Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
