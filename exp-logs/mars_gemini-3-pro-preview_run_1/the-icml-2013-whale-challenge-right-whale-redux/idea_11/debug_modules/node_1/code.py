import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.dataset import get_dataloaders, WhaleDataset
from library.model import HierarchicalCRNN
from library.train import Trainer, predict


def main():
    print("=== Starting Whale Detection Demo ===")

    # ---------------------------------------------------------
    # 1. Setup Directories and Create Data Subset (Speed Optimization)
    # ---------------------------------------------------------
    # We create a temporary working directory to store subset metadata and cache
    WORK_DIR = "./working/demo_execution"
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    METADATA_SUBSET_DIR = os.path.join(WORK_DIR, "metadata")
    os.makedirs(METADATA_SUBSET_DIR)

    print(f"Creating metadata subsets in {METADATA_SUBSET_DIR}...")

    # Create subsets of the original metadata to speed up data loading/processing
    # We take 20 samples for train/val/test
    for split in ["train", "val", "test"]:
        original_csv = f"./metadata/{split}.csv"
        target_csv = os.path.join(METADATA_SUBSET_DIR, f"{split}.csv")

        if os.path.exists(original_csv):
            df = pd.read_csv(original_csv)
            # Sample a small subset. Use random_state for reproducibility.
            # We ensure we don't exceed the dataframe length.
            n_samples = min(20, len(df))
            df_subset = df.sample(n=n_samples, random_state=Config.SEED)
            df_subset.to_csv(target_csv, index=False)
        else:
            raise FileNotFoundError(f"Original metadata not found at {original_csv}")

    # ---------------------------------------------------------
    # 2. Override Configuration
    # ---------------------------------------------------------
    print("Overriding Config parameters for demo...")

    # Point to our subset metadata
    Config.METADATA_DIR = METADATA_SUBSET_DIR

    # Use a separate cache directory for this run
    Config.CACHE_DIR = os.path.join(WORK_DIR, "cache")

    # Set output directories
    Config.SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute requirements
    Config.BATCH_SIZE = 4
    Config.N_EPOCHS = 1
    Config.PATIENCE = 1

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 3. Data Loading & Verification
    # ---------------------------------------------------------
    print("Initializing DataLoaders...")
    # load_cached_data=False forces processing of our new subset metadata
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print("Verifying Data Shapes...")
    # Fetch one batch from train_loader
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty. Check metadata subset creation.")

    # Expected Shape: (Batch, Channels, Freq, Time)
    # Channels = 1
    # Freq = Config.N_MELS (128)
    # Time ~ 200 (depends on duration/hop_length)
    print(f"Batch Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")

    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {images.shape[0]}"
    assert images.shape[1] == 1, "Expected 1 channel (mono)"
    assert images.shape[2] == Config.N_MELS, f"Expected {Config.N_MELS} mel bands"

    # ---------------------------------------------------------
    # 4. Model Instantiation & Forward Pass Verification
    # ---------------------------------------------------------
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HierarchicalCRNN().to(device)

    print("Verifying Forward Pass...")
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Output Shape: {outputs.shape}")
    # Model returns logits of shape (Batch, 1)
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("Starting Training (1 Epoch)...")
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run training
    trainer.fit(n_epochs=Config.N_EPOCHS)

    # Check if model was saved
    if not os.path.exists(trainer.best_model_path):
        # If validation AUC didn't improve (possible with random init and tiny data),
        # force save for the sake of the demo flow, or handle gracefully.
        # The Trainer logic saves if val_auc > best_auc (init 0.0).
        # If val_auc is 0.0, it might not save.
        print(
            "Note: Best model might not have been saved if AUC was 0. Saving current state manually for demo."
        )
        torch.save(model.state_dict(), trainer.best_model_path)

    assert os.path.exists(trainer.best_model_path), "Model file missing after training"

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("Running Inference on Test Set...")
    # Load best model
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))

    predictions = predict(model, test_loader, device)

    print(f"Generated {len(predictions)} predictions.")
    assert len(predictions) == len(test_loader.dataset), "Prediction count mismatch"

    print("Generating Submission CSV...")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Retrieve IDs from dataset
    test_ids = test_loader.dataset.ids
    submission_df = pd.DataFrame({"clip": test_ids, "probability": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head().to_string())

    # Final Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
