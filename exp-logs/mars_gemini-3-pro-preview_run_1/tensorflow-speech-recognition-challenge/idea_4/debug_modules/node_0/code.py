import os
import sys
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import SpeechCommandDataset
from library.model import DilatedEfficientNet
from library.trainer import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Configuration...")
    # Initialize Config with debug settings for speed
    # subset_size=200 limits the dataset to 200 samples
    # epochs=2 ensures the training loop finishes quickly
    config = Config(debug=True, subset_size=200, epochs=2)

    # Override working directory to isolate this demo execution
    config.working_dir = "./working/demo_execution"
    config.output_dir = config.working_dir
    os.makedirs(config.working_dir, exist_ok=True)

    # Ensure reproducibility
    set_seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Working Directory: {config.working_dir}")

    # -------------------------------------------------------------------------
    # 1. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Datasets...")

    # Initialize Datasets
    # Note: The first run might take a few seconds to process/cache the balanced dataframe
    train_dataset = SpeechCommandDataset(mode="train", config=config)
    val_dataset = SpeechCommandDataset(mode="val", config=config)
    test_dataset = SpeechCommandDataset(mode="test", config=config)

    # Initialize DataLoaders
    # Using a small batch size for the demo
    demo_batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=demo_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=demo_batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=demo_batch_size, shuffle=False, num_workers=2
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size:   {len(val_dataset)}")
    print(f"Test Dataset Size:  {len(test_dataset)}")

    # Verification: Check Batch Shapes
    print("Verifying data shapes...")
    sample_batch, sample_labels = next(iter(train_loader))

    # Expected Spectrogram Shape: [Batch, 1, n_mels, time_steps]
    # time_steps = (sample_rate * duration) // hop_length + 1
    # 16000 * 1.0 // 160 + 1 = 101
    expected_shape = (demo_batch_size, 1, config.n_mels, 101)

    assert (
        sample_batch.shape == expected_shape
    ), f"Batch shape mismatch. Expected {expected_shape}, got {sample_batch.shape}"

    assert sample_labels.shape == (
        demo_batch_size,
    ), f"Label shape mismatch. Expected {(demo_batch_size,)}, got {sample_labels.shape}"

    print("Data verification passed.")

    # -------------------------------------------------------------------------
    # 2. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model...")

    model = DilatedEfficientNet(config)

    # Move to device (CPU or CUDA)
    model.to(config.device)

    # Verification: Dummy Forward Pass
    print("Verifying model forward pass...")
    dummy_input = torch.randn(demo_batch_size, 1, config.n_mels, 101).to(config.device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected Output Shape: [Batch, Num_Classes]
    expected_out_shape = (demo_batch_size, config.num_classes)

    assert (
        output.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 3] Starting Training...")

    trainer = Trainer(model, train_loader, val_loader, config)

    # Run training
    trainer.fit()

    # Verification: Check if best model is saved
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training failed to save model checkpoint at {best_model_path}"
        )

    print(f"Training completed. Model saved to {best_model_path}")

    # -------------------------------------------------------------------------
    # 4. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Submission...")

    trainer.generate_submission(test_loader)

    # Verification: Check Submission File
    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check columns
    required_cols = {"fname", "label"}
    if not required_cols.issubset(df_sub.columns):
        raise ValueError(
            f"Submission file missing required columns. Found: {df_sub.columns}"
        )

    # Check row count (should match test dataset size)
    if len(df_sub) != len(test_dataset):
        raise ValueError(
            f"Submission row count mismatch. Expected {len(test_dataset)}, got {len(df_sub)}"
        )

    print(f"Submission verified. File saved to {submission_path}")
    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    run_demo()
