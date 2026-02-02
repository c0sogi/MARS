import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import seed_everything, calculate_weights
from library.dataset import get_dataloaders, WhaleDataset
from library.model import EfficientNetGeM, GeM
from library.trainer import Trainer


def run_demonstration():
    print("Starting Right Whale Detection Library Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # We modify the Config class attributes at runtime to run a small subset
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Very small subset for speed
    Config.EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.EARLY_STOPPING_PATIENCE = 1

    # Clean up any previous debug artifacts in working directory to ensure fresh run
    if os.path.exists(Config.IDEA_DIR):
        shutil.rmtree(Config.IDEA_DIR)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    Config.print_config()

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("\n[2] Verifying 'library.utils'...")

    # Test Seed
    seed_everything(Config.SEED)
    r1 = torch.rand(1).item()
    seed_everything(Config.SEED)
    r2 = torch.rand(1).item()
    assert r1 == r2, "Seed setting did not produce deterministic results."
    print(" - seed_everything: OK")

    # Test Weight Calculation
    # Create a dummy dataframe: 4 negatives, 1 positive -> weight should be 4/1 = 4.0
    dummy_df = pd.DataFrame({"label": [0, 0, 0, 0, 1]})
    weight = calculate_weights(dummy_df, label_col="label")
    assert abs(weight - 4.0) < 1e-6, f"Expected weight 4.0, got {weight}"
    print(" - calculate_weights: OK")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # --------------------------------------------------------------------------
    print("\n[3] Verifying 'library.dataset'...")

    # This will generate debug cache files since Config.DEBUG is True
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check DataLoader lengths
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."

    # Fetch one batch to verify shapes and processing
    images, labels = next(iter(train_loader))

    # Expected Shape: (Batch, Channels, Freq, Time)
    # Freq = 128 (N_MELS), Time approx 201 (Duration * SR / Hop)
    print(f" - Batch Input Shape: {images.shape}")
    print(f" - Batch Label Shape: {labels.shape}")

    assert images.dim() == 4, "Images should be 4D tensors (B, C, F, T)"
    assert images.size(1) == 1, "Input channel should be 1 (Spectrogram)"
    assert images.size(2) == Config.N_MELS, f"Height should be {Config.N_MELS}"
    assert labels.dim() == 1, "Labels should be 1D tensors"
    assert labels.size(0) == Config.BATCH_SIZE, "Batch size mismatch"

    print(" - Data Loading and Preprocessing: OK")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[4] Verifying 'library.model'...")

    model = EfficientNetGeM(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Check GeM layer existence
    assert isinstance(model.pooling, GeM), "Model should use GeM pooling"

    # Run forward pass with the batch fetched earlier
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        outputs = model(images)

    print(f" - Model Output Shape: {outputs.shape}")

    # Expected Output: (Batch, Num_Classes) -> (B, 1)
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print(" - Model Forward Pass: OK")

    # --------------------------------------------------------------------------
    # 5. Verify Training Loop (Trainer)
    # --------------------------------------------------------------------------
    print("\n[5] Verifying 'library.trainer'...")

    # Initialize Trainer
    # Note: Trainer will re-initialize model and loaders.
    # Since we already generated the cache in step 3, it will load quickly.
    trainer = Trainer()

    # Run Training (Fit)
    # We set EPOCHS=1, so this runs a single epoch loop + validation
    print(" - Starting training loop (1 Epoch)...")
    trainer.fit()

    # Check if best model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print(" - Training loop completed and model saved: OK")

    # Run Inference (Predict)
    print(" - Starting inference...")
    trainer.predict()

    # Check Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f" - Submission Shape: {df_sub.shape}")

    # Verify Submission Format
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Submission columns missing."
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission rows {len(df_sub)} != subset size {Config.DEBUG_SUBSET_SIZE}"
    assert df_sub["probability"].dtype == float, "Probability column should be float."

    print(" - Inference and Submission generation: OK")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demonstration()
