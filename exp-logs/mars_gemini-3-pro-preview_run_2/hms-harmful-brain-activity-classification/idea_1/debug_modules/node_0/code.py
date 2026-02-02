import os
import torch
import pandas as pd
import numpy as np
import random
import warnings
import shutil
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.signal_processing import EEGProcessor
from library.dataset import EEGDataset
from library.model import SpecEfficientNet
from library.engine import Trainer

# -----------------------------------------------------------------------------
# Setup & Configuration
# -----------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting EEG Classification Pipeline Demo ===")

    # 1. Override Configuration for Speed and Demo
    # We modify the Config class attributes directly to run a minimal version.
    print("\n[1] Configuring environment...")

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Ensure working directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify EEGProcessor (Signal Processing)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying EEGProcessor...")

    # Load metadata to find a valid file path
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    sample_row = train_meta.iloc[0]
    eeg_rel_path = sample_row["eeg_path"]

    processor = EEGProcessor()

    # Test: Load and Process a single file
    # We use the load_and_process method which handles loading, montage, spec, and resizing
    try:
        img_tensor = processor.load_and_process(
            eeg_path=eeg_rel_path,
            offset_seconds=sample_row["eeg_label_offset_seconds"],
            load_cached_data=False,  # Force computation
            cache_id="debug_test",
        )

        # Assertions
        assert isinstance(img_tensor, torch.Tensor), "Output must be a torch Tensor"
        assert (
            img_tensor.ndim == 3
        ), f"Expected 3 dimensions (C, H, W), got {img_tensor.ndim}"
        assert img_tensor.shape == (
            3,
            *Config.IMG_SIZE,
        ), f"Expected shape (3, {Config.IMG_SIZE[0]}, {Config.IMG_SIZE[1]}), got {img_tensor.shape}"
        assert img_tensor.dtype == torch.float32, "Expected float32 dtype"

        print("    EEGProcessor output shape verified:", img_tensor.shape)

    except Exception as e:
        raise RuntimeError(f"EEGProcessor verification failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Verify EEGDataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[3] Verifying EEGDataset & DataLoader...")

    # Create a small subset of data for training simulation (e.g., 16 samples)
    subset_df = train_meta.head(16).copy()

    # Instantiate Dataset
    dataset = EEGDataset(data=subset_df, mode="train")

    # Assertions
    assert len(dataset) == 16, "Dataset length mismatch"

    # Test __getitem__
    img, target = dataset[0]
    assert img.shape == (3, *Config.IMG_SIZE), "Dataset image shape incorrect"
    assert target.shape == (6,), "Dataset target shape incorrect"
    assert torch.isclose(
        target.sum(), torch.tensor(1.0)
    ), "Target probabilities must sum to 1"

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    batch_imgs, batch_targets = next(iter(loader))
    print(f"    Batch Image Shape: {batch_imgs.shape}")
    print(f"    Batch Target Shape: {batch_targets.shape}")

    assert batch_imgs.shape[0] == Config.BATCH_SIZE
    assert batch_targets.shape[1] == 6

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying SpecEfficientNet Model...")

    model = SpecEfficientNet(config=Config, pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    # Forward pass with the batch from previous step
    batch_imgs = batch_imgs.to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(batch_imgs)

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 6), "Model output shape incorrect"
    # Check Softmax (sums to 1)
    sums = outputs.sum(dim=1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ), "Model output is not a valid probability distribution"

    print("    Model forward pass successful. Output shape:", outputs.shape)

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Simulation...")

    # Split subset into train/val
    train_subset = subset_df.iloc[:12]
    val_subset = subset_df.iloc[12:]

    train_ds = EEGDataset(train_subset, mode="train")
    val_ds = EEGDataset(val_subset, mode="val")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Setup Training Components
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    trainer = Trainer(model, optimizer, Config.DEVICE, scheduler)

    # Run Fit
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("    Training simulation complete. Model saved.")

    # -------------------------------------------------------------------------
    # 6. Verify Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference & Generating Submission...")

    # Load Test Metadata (Using provided metadata file)
    test_meta = pd.read_csv(Config.TEST_CSV)

    # For demo speed, we only predict on the first few rows of the test set
    test_subset = test_meta.head(8).copy()

    test_ds = EEGDataset(test_subset, mode="test")
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Predict
    # Note: Trainer uses the model currently in memory.
    # In a real run, we might load the best state_dict from disk.
    preds = trainer.predict(test_loader)

    assert preds.shape == (len(test_subset), 6), "Prediction shape mismatch"

    # Create Submission DataFrame
    submission = pd.DataFrame(preds, columns=Config.VOTE_COLS)
    submission.insert(0, "eeg_id", test_subset["eeg_id"])

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print("    Sample Predictions:")
    print(submission.head().to_string())
    print(f"\n    Submission saved to: {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
