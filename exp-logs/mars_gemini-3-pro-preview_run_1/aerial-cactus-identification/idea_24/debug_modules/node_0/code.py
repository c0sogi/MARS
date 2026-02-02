import sys
import os
import torch
import pandas as pd
import numpy as np
import glob

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

from library.config import Config
from library.data import get_dataloaders
from library.model import MultiHeadRepVGG, RepVGGBlock
from library.engine import train_fold_swa, predict_tta
from library.utils import set_seed


def main():
    print("Starting demonstration script...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("Configuring environment for fast demonstration...")

    # Enable Debug mode to use a small subset of data (1000 samples)
    Config.DEBUG = True

    # Reduce epochs to minimum for demonstration
    Config.EPOCHS_CONVERGENCE = 1
    Config.EPOCHS_SWA = 1

    # Adjust batch size for the small subset if necessary, though 128 is fine
    Config.BATCH_SIZE = 32

    # Setup directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Loading & Validation
    # ==========================================
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Fetch a single batch to validate shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions
    assert len(images.shape) == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == 32 and images.shape[3] == 32, "Images should be 32x32"
    assert len(targets.shape) == 1, "Targets should be 1D tensors"
    assert images.dtype == torch.float32, "Images should be float32"

    print("Data loading verification passed.")

    # ==========================================
    # 3. Model Initialization & Check
    # ==========================================
    print("\nInitializing model...")
    device = Config.DEVICE
    model = MultiHeadRepVGG(num_classes=1, deploy=False).to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    tex_out, sem_out = model(dummy_input)

    print(f"Model Output Shapes - Texture: {tex_out.shape}, Semantic: {sem_out.shape}")

    # Assertions
    assert tex_out.shape == (2, 1), "Texture head output shape mismatch"
    assert sem_out.shape == (2, 1), "Semantic head output shape mismatch"

    print("Model initialization verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\nStarting training loop (Fold 0)...")

    # Train for one fold (runs convergence phase + SWA phase)
    # This returns the averaged SWA model (nn.Module)
    trained_model = train_fold_swa(
        model, train_loader, val_loader, fold_idx=0, device=device
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "swa_fold0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    print(f"Training complete. Checkpoint saved at {checkpoint_path}")

    # ==========================================
    # 5. Inference & Optimization (RepVGG Deploy)
    # ==========================================
    print("\nOptimizing model for inference (RepVGG Deploy Mode)...")

    # Check structure before deploy switch (should have rbr_dense, rbr_1x1)
    has_branches = False
    for m in trained_model.modules():
        if isinstance(m, RepVGGBlock):
            if hasattr(m, "rbr_dense"):
                has_branches = True
                break
    assert has_branches, "Model should have multi-branch blocks before deploy switch"

    # Switch to deploy (fuses layers)
    trained_model.switch_to_deploy()
    trained_model.eval()

    # Check structure after deploy switch (should NOT have rbr_dense)
    for m in trained_model.modules():
        if isinstance(m, RepVGGBlock):
            assert not hasattr(
                m, "rbr_dense"
            ), "RepVGGBlock still has rbr_dense after deploy switch"
            assert hasattr(
                m, "rbr_reparam"
            ), "RepVGGBlock missing rbr_reparam after deploy switch"

    print("Model successfully switched to deploy mode.")

    print("Running inference on test set (with TTA)...")
    predictions = predict_tta(trained_model, test_loader, device)

    # Validate predictions
    assert len(predictions) == len(
        test_loader.dataset
    ), "Number of predictions does not match test set size"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions should be probabilities [0, 1]"

    print(f"Generated {len(predictions)} predictions.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nGenerating submission file...")

    # Load sample submission to get IDs
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # If we are in debug mode, the test loader only loaded a subset.
    # We need to match predictions to the correct IDs.
    if Config.DEBUG:
        # In debug mode, get_dataloaders loads the first N samples.
        # We slice the dataframe to match.
        sub_df = sub_df.iloc[: len(predictions)].copy()

    # Assign predictions
    sub_df["has_cactus"] = predictions

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print("First 5 rows:")
    print(sub_df.head())

    print("\nDemonstration script completed successfully.")


if __name__ == "__main__":
    main()
