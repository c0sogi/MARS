import os
import shutil
import torch
import numpy as np
import pandas as pd
import random
import warnings

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ShallowCNN
from library.trainer import ModelTrainer
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # -------------------------------------------------------------------------
    # 1. Configure for Speed/Debug
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config parameters to run a fast, small-scale test
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples per split
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect outputs to a temp working directory to avoid polluting real workspace
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean state for the demo directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Dataset Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Dataset Loading...")

    # Force reload (load_cached_data=False) to demonstrate raw audio processing logic
    # This will read .aif files, compute spectrograms, and cache them
    dataloaders = get_dataloaders(load_cached_data=False)

    # Verify DataLoader keys
    assert "train" in dataloaders, "Train loader missing"
    assert "val" in dataloaders, "Val loader missing"
    assert "test" in dataloaders, "Test loader missing"

    train_loader = dataloaders["train"]
    print(f"  Train loader batches: {len(train_loader)}")

    # Fetch a single batch to verify shapes
    inputs, labels = next(iter(train_loader))
    print(f"  Batch Input Shape: {inputs.shape}")  # Expected: (B, 1, F, T)
    print(f"  Batch Label Shape: {labels.shape}")  # Expected: (B,)

    # Assertions
    assert inputs.shape[0] == Config.BATCH_SIZE, "Incorrect batch size"
    assert inputs.shape[1] == 1, "Incorrect channel dimension (should be 1)"
    assert (
        inputs.shape[2] == Config.N_MELS
    ), f"Incorrect frequency bins (should be {Config.N_MELS})"
    # Time dimension depends on duration/hop_length, roughly 32 for 2s @ 2000Hz
    assert inputs.shape[3] > 0, "Time dimension is zero"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("  Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Architecture...")

    model = ShallowCNN()

    # Move inputs to CPU for this quick check (model is init on CPU by default here)
    inputs = inputs.to("cpu")

    # Forward pass
    outputs = model(inputs)
    print(f"  Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert outputs.min() >= 0.0, "Output probability < 0"
    assert outputs.max() <= 1.0, "Output probability > 1"

    print("  Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop...")

    # Initialize Trainer (it will reload dataloaders, using the cache generated in step 2)
    trainer = ModelTrainer(epochs=Config.EPOCHS, debug=True)

    # Run training
    best_model_path = trainer.train()
    print(f"  Training complete. Best model saved at: {best_model_path}")

    # Verify file exists
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model file was not created.")

    print("  Training verification passed.")

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference...")

    # Run inference using the trained model
    generate_submission(best_model_path, output_path=Config.SUBMISSION_PATH, debug=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission shape: {df_sub.shape}")
    print(f"  Submission columns: {df_sub.columns.tolist()}")

    # Assertions
    assert "clip" in df_sub.columns, "Column 'clip' missing in submission"
    assert "probability" in df_sub.columns, "Column 'probability' missing in submission"
    assert len(df_sub) > 0, "Submission file is empty"
    # Length should match debug subset size
    assert (
        len(df_sub) <= Config.DEBUG_SUBSET_SIZE
    ), "Submission contains more rows than debug subset"

    print("  Inference verification passed.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    set_seed(42)
    run_demo()
