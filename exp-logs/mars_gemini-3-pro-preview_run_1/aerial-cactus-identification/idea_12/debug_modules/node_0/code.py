import os
import torch
import numpy as np
import pandas as pd
import sys

# Import the library modules
# We import Config first to patch it before other modules consume it
from library.config import Config

# --- 1. Configuration Override for Speed and Demo ---
print("--- Configuring Environment for Demonstration ---")
# Enable debug mode to use a small subset of data (50 samples)
Config.DEBUG = True
Config.DEBUG_SAMPLES = 50
# Reduce batch size for the small subset
Config.BATCH_SIZE = 8
# Set minimal epochs to verify the loop works without waiting for convergence
Config.EPOCHS_CONVERGENCE = 1
Config.EPOCHS_SWA = 1
Config.TOTAL_EPOCHS = Config.EPOCHS_CONVERGENCE + Config.EPOCHS_SWA
# Ensure we use a clean working directory for the demo if needed,
# but we stick to the default provided in Config to ensure path consistency.
print(f"Debug Mode: {Config.DEBUG}")
print(f"Total Epochs: {Config.TOTAL_EPOCHS}")

# Now import the rest of the library
from library.utils import set_seed, mixup_data
from library.data import get_dataloaders
from library.model import MetadataGatedRepVGG
from library.train import Trainer


def run_demo():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # --- 2. Verify Utility Functions ---
    print("\n--- Verifying Utility Functions ---")
    # Test Mixup
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 32, 32)
    dummy_lbls = torch.tensor([0, 1, 0, 1])

    mixed_imgs, y_a, y_b, lam = mixup_data(dummy_imgs, dummy_lbls, alpha=1.0)

    assert mixed_imgs.shape == dummy_imgs.shape, "Mixup output image shape mismatch"
    assert y_a.shape == dummy_lbls.shape, "Mixup target A shape mismatch"
    assert y_b.shape == dummy_lbls.shape, "Mixup target B shape mismatch"
    print("Mixup logic verified.")

    # --- 3. Verify Data Pipeline ---
    print("\n--- Verifying Data Pipeline ---")
    # We force load_cached_data=False to ensure the processing logic runs on the debug subset
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch
    images, metadata, labels = next(iter(train_loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Metadata: {metadata.shape}, Labels: {labels.shape}"
    )

    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert metadata.shape == (Config.BATCH_SIZE, 1), "Incorrect metadata batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    print("Data Loaders verified.")

    # --- 4. Verify Model Architecture ---
    print("\n--- Verifying Model Architecture ---")
    model = MetadataGatedRepVGG()

    # Check Forward Pass (Training Mode)
    model.train()
    outputs = model(images, metadata)
    assert outputs.shape == (Config.BATCH_SIZE, 2), "Model output shape mismatch"
    print("Forward pass (Training) successful.")

    # Check Structural Re-parameterization (RepVGG)
    # The 'stem' is a RepVGGBlock. In training mode, it should have 'rbr_dense' (3x3 branch).
    assert hasattr(
        model.stem, "rbr_dense"
    ), "Model should have multi-branch structure in training mode"
    assert not hasattr(
        model.stem, "rbr_reparam"
    ), "Model should not have re-param structure yet"

    # Switch to Deploy
    print("Switching model to deploy mode...")
    model.switch_to_deploy()

    # Verify branches are fused
    assert not hasattr(
        model.stem, "rbr_dense"
    ), "Multi-branch structure should be removed after deploy switch"
    assert hasattr(
        model.stem, "rbr_reparam"
    ), "Fused re-param structure should exist after deploy switch"

    # Check Forward Pass (Inference Mode)
    model.eval()
    with torch.no_grad():
        outputs_deploy = model(images, metadata)
    assert outputs_deploy.shape == (
        Config.BATCH_SIZE,
        2,
    ), "Deploy model output shape mismatch"
    print("RepVGG structural re-parameterization verified.")

    # --- 5. Verify Training Pipeline (Trainer) ---
    print("\n--- Verifying Training Pipeline ---")
    # Instantiate Trainer
    # Note: Trainer re-initializes dataloaders inside __init__, but since we patched Config,
    # it will use the debug settings.
    trainer = Trainer()

    # Run the full training loop (Convergence -> SWA -> BN Update)
    print("Running training loop (this may take a few seconds)...")
    final_model_path = trainer.run()

    assert os.path.exists(
        final_model_path
    ), f"Final model not found at {final_model_path}"
    print(f"Training complete. Model saved to {final_model_path}")

    # --- 6. Verify Inference and Submission ---
    print("\n--- Verifying Inference ---")
    trainer.predict(final_model_path)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")

    # In debug mode, test_loader is also truncated to DEBUG_SAMPLES (50).
    # So we expect 50 rows.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions, found {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission columns missing"

    print("Inference and submission generation verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
