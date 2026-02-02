import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, kl_divergence_score
from library.data import get_loaders
from library.models import TriViewNet
from library.train import train
from library.infer import inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Set a fixed seed for reproducibility
    set_seed(42)

    # Override Config for a fast demo run
    # We use a specific subdirectory in ./working to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Configuring environment... Working Directory: {demo_dir}")

    # Update Config paths dynamically
    Config.WORKING_DIR = demo_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Create necessary directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Adjust hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset of data
    Config.NUM_WORKERS = 2

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loading (library.data) ---")

    # Initialize loaders with debug=True to use the small sample size
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, load_cached_data=False
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Extract components
    micro = batch["micro"]
    meso = batch["meso"]
    macro = batch["macro"]
    targets = batch["target"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Micro (EEG) shape: {micro.shape}")
    print(f"Meso (Local Spec) shape: {meso.shape}")
    print(f"Macro (Global Spec) shape: {macro.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions to ensure data integrity
    # Micro: (Batch, Channels=20, Time=5000)
    assert micro.shape == (Config.BATCH_SIZE, 20, 5000), "Incorrect Micro-View shape"
    # Meso: (Batch, Channels=3, H=224, W=224)
    assert meso.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Meso-View shape"
    # Macro: (Batch, Channels=3, H=512, W=512)
    assert macro.shape == (Config.BATCH_SIZE, 3, 512, 512), "Incorrect Macro-View shape"
    # Targets: (Batch, Classes=6)
    assert targets.shape == (Config.BATCH_SIZE, 6), "Incorrect Target shape"

    print("Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Logic (library.models) ---")

    # Instantiate model (pretrained=False for speed/offline safety in demo)
    model = TriViewNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    micro = micro.to(Config.DEVICE)
    meso = meso.to(Config.DEVICE)
    macro = macro.to(Config.DEVICE)

    # Perform forward pass
    with torch.no_grad():
        logits = model(micro, meso, macro)

    print(f"Logits shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 6), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN values"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n--- Testing Metric Calculation (library.utils) ---")

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Calculate KL Divergence
    score = kl_divergence_score(targets, probs)
    print(f"Calculated KL Score: {score:.4f}")

    # Assertion
    assert isinstance(score, float), "KL Score should be a float"
    assert score >= 0, "KL Divergence cannot be negative"

    print("Metric verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Loop (library.train) ---")

    # Run training for 1 epoch on the debug subset
    # This will save the best model to Config.CHECKPOINT_DIR
    train(debug=True, epochs=1)

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"

    print(f"Training complete. Checkpoint saved to {expected_checkpoint}")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Inference Pipeline (library.infer) ---")

    # Run inference using the newly trained checkpoint
    inference(
        checkpoint_path=expected_checkpoint,
        output_path=Config.SUBMISSION_PATH,
        device=str(Config.DEVICE),
        debug=True,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print("First 2 rows of submission:")
    print(df_sub.head(2))

    # Verify columns
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Verify probability sum constraint
    vote_cols = expected_cols[1:]
    row_sums = df_sub[vote_cols].sum(axis=1)
    # Allow small float tolerance
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("Inference verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
