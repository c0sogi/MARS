import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HierarchicalCRNN
from library.engine import train_model


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Modify Config for a fast demonstration run
    print("Configuring for demo execution...")
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = f"./working/{Config.PROJECT_NAME}"
    Config.OUTPUT_DIR = Config.WORKING_DIR

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Sufficient for demo
    Config.EPOCHS = 2  # Minimal epochs to demonstrate training loop

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[Step 1/4] Loading Data...")
    # We use a small debug_size to ensure the script runs quickly (e.g., < 5 mins)
    # load_cached_data=False forces the preprocessing logic to run, verifying it works.
    DEBUG_SIZE = 100

    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # Verify Data Shapes
    try:
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"  Train Batch Shape: {sample_batch.shape}")
        print(f"  Train Label Shape: {sample_labels.shape}")

        # Assertions
        assert sample_batch.dim() == 4, "Batch must be 4D (N, C, F, T)"
        assert sample_batch.shape[1] == 1, "Input must have 1 channel"
        assert (
            sample_batch.shape[2] == Config.N_MELS
        ), f"Freq dim must be {Config.N_MELS}"
        # Time dimension depends on preprocessing, usually fixed around 200 for 2s @ 2kHz with hop 20

    except StopIteration:
        raise RuntimeError("Train loader is empty. Check data loading logic.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Step 2/4] Initializing Model...")
    model = HierarchicalCRNN().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Use the sample batch from data loading
        dummy_input = sample_batch.to(device)
        dummy_output = model(dummy_input)

        print(f"  Output Shape: {dummy_output.shape}")
        assert dummy_output.shape == (
            dummy_input.size(0),
            1,
        ), "Model output shape mismatch"
        assert not torch.isnan(dummy_output).any(), "Model produced NaN outputs"

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n[Step 3/4] Starting Training...")

    # Define Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Run Training
    # train_model handles the loop, validation, and checkpointing
    best_auc = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    print(f"  Training finished. Best AUC: {best_auc:.4f}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 4/4] Running Inference & Generating Submission...")

    # Load best model weights
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("  Loaded best model checkpoint.")
    else:
        print("  Warning: Best model checkpoint not found. Using current weights.")

    model.eval()
    all_probs = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)

    # Create Submission DataFrame
    # Note: test_ids corresponds to the order in test_loader (shuffle=False)
    submission_df = pd.DataFrame({"clip": test_ids, "probability": all_probs})

    # Save Submission
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    # Verify Submission
    print(f"  Submission saved to: {submission_path}")
    print("  First 5 rows:")
    print(submission_df.head())

    assert os.path.exists(submission_path), "Submission file was not created."
    assert len(submission_df) == len(test_ids), "Submission row count mismatch."
    assert (
        submission_df["probability"].between(0, 1).all()
    ), "Probabilities must be in [0, 1]."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
