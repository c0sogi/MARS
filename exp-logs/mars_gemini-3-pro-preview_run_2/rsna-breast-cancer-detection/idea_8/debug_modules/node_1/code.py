import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
# We import Config first to modify it before other components use it
from library.config import Config

# --- 1. Configure for Fast Demonstration ---
print("Configuring for fast demonstration...")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
Config.NUM_EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 8  # Increased for stability
Config.GRAD_ACCUMULATION_STEPS = 1
Config.IMAGE_SIZE = (256, 256)  # Reduced resolution for speed
Config.LEARNING_RATE = 1e-4  # Lower LR for stability with random init
Config.WORKING_DIR = "./working/demo_execution"
Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Clean up stale model from previous runs to avoid architecture mismatch errors
if os.path.exists(Config.MODEL_PATH):
    os.remove(Config.MODEL_PATH)

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# --- 2. Monkey-Patch Model to avoid Downloads ---
# The Trainer class initializes the model with pretrained=True.
# We patch this to False to ensure the demo runs without internet access.
import library.model

original_init = library.model.MultiTaskEfficientNet.__init__


def patched_init(self, pretrained=True):
    # Force pretrained=False regardless of input
    original_init(self, pretrained=False)


library.model.MultiTaskEfficientNet.__init__ = patched_init

# Import remaining components after config setup
from library.dataset import get_dataloaders
from library.model import MultiTaskEfficientNet
from library.loss import StableFocalLoss
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    # Setup directories
    Config.setup()

    # --- 3. Demonstrate Dataset & DataLoader ---
    print("\n--- Demonstrating Dataset & DataLoader ---")
    # Force load_cached_data=False to ensure we test the preprocessing logic
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify Train Loader
    if train_loader is None:
        raise RuntimeError("Train loader is None. Check metadata availability.")

    # Fetch one batch
    images, meta, targets = next(iter(train_loader))

    print(f"Batch Shapes -> Images: {images.shape}, Meta: {meta.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Unexpected image shape: {images.shape}"
    assert meta.shape == (
        Config.BATCH_SIZE,
        4,
    ), f"Unexpected metadata shape: {meta.shape}"
    assert "cancer" in targets, "Targets dictionary missing 'cancer' key"
    assert targets["cancer"].shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected target shape: {targets['cancer'].shape}"

    print("Dataset verification passed.")

    # --- 4. Demonstrate Model Architecture ---
    print("\n--- Demonstrating Model Architecture ---")
    model = MultiTaskEfficientNet(pretrained=False)
    model.eval()

    with torch.no_grad():
        outputs = model(images, meta)

    print(f"Output Keys: {list(outputs.keys())}")

    # Assertions
    assert "cancer" in outputs
    assert outputs["cancer"].shape == (Config.BATCH_SIZE, 1)
    if Config.USE_AUX_HEADS:
        assert "birads" in outputs
        assert "density" in outputs

    print("Model forward pass verification passed.")

    # --- 5. Demonstrate Loss Function ---
    print("\n--- Demonstrating Loss Function ---")
    criterion = StableFocalLoss()

    # Create dummy logits and targets
    logits = torch.randn(Config.BATCH_SIZE, 1)
    target_labels = torch.randint(0, 2, (Config.BATCH_SIZE, 1)).float()

    loss = criterion(logits, target_labels)
    print(f"Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print("Loss function verification passed.")

    # --- 6. Demonstrate Training Loop ---
    print("\n--- Demonstrating Training Loop ---")
    # Initialize Trainer (uses patched model init)
    trainer = Trainer()

    # Run training (1 epoch on debug subset)
    trainer.train()

    # Verify model checkpoint creation
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print(f"Training loop completed. Model saved to {Config.MODEL_PATH}")

    # --- 7. Demonstrate Inference ---
    print("\n--- Demonstrating Inference ---")
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{sub_df.head()}")

    # Assertions on submission format
    assert "prediction_id" in sub_df.columns, "Missing 'prediction_id' column"
    assert "cancer" in sub_df.columns, "Missing 'cancer' column"
    assert len(sub_df) > 0, "Submission file is empty"

    print("Inference verification passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
