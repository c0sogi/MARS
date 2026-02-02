import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import ToxicityDataset
from library.model import ToxicityModel
from library.trainer import Trainer


def main():
    print("=" * 50)
    print("Starting Toxicity Classification Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config parameters for speed and isolation
    Config.debug = True
    Config.debug_subset_size = 64  # Small subset for quick execution
    Config.epochs = 1
    Config.train_batch_size = 16
    Config.valid_batch_size = 16

    # Redirect outputs to a specific demo directory
    Config.working_dir = "./working/demo_execution"
    Config.submission_dir = "./working/demo_execution"

    # Update dependent paths manually since they were defined at class level
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Create the directories
    Config.create_dirs()

    # Set seeds
    seed_everything(Config.seed)
    print(f"Debug Mode: {Config.debug}")
    print(f"Working Directory: {Config.working_dir}")

    # ---------------------------------------------------------
    # 2. Dataset Demonstration
    # ---------------------------------------------------------
    print("\n[2] Demonstrating ToxicityDataset...")

    # Initialize dataset (force processing from scratch to verify logic)
    print("Initializing Train Dataset...")
    train_dataset = ToxicityDataset(
        split="train",
        load_cached_data=False,
        debug=Config.debug,
        debug_subset_size=Config.debug_subset_size,
    )

    # Verification
    print(f"Dataset length: {len(train_dataset)}")
    if len(train_dataset) != Config.debug_subset_size:
        raise AssertionError(
            f"Expected dataset size {Config.debug_subset_size}, got {len(train_dataset)}"
        )

    # Check sample structure
    sample = train_dataset[0]
    print("Sample keys:", list(sample.keys()))

    required_keys = ["input_ids", "attention_mask", "labels"]
    for key in required_keys:
        if key not in sample:
            raise AssertionError(f"Missing key in dataset sample: {key}")

    # Check tensor shapes
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    print(f"Input shape: {input_ids.shape}")
    print(f"Label shape: {labels.shape}")

    if input_ids.shape[0] != Config.max_len:
        raise AssertionError(
            f"Input sequence length mismatch. Expected {Config.max_len}, got {input_ids.shape[0]}"
        )

    if labels.shape[0] != Config.num_classes:
        raise AssertionError(
            f"Label dimension mismatch. Expected {Config.num_classes}, got {labels.shape[0]}"
        )

    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model Demonstration
    # ---------------------------------------------------------
    print("\n[3] Demonstrating ToxicityModel...")

    model = ToxicityModel()
    model.to(Config.device)
    model.eval()

    # Prepare a dummy batch
    batch_input_ids = input_ids.unsqueeze(0).to(Config.device)
    batch_mask = sample["attention_mask"].unsqueeze(0).to(Config.device)

    print("Running forward pass...")
    with torch.no_grad():
        output = model(batch_input_ids, batch_mask)

    print(f"Output shape: {output.shape}")

    expected_shape = (1, Config.num_classes)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print("Model verification passed.")

    # ---------------------------------------------------------
    # 4. Trainer Demonstration (Training)
    # ---------------------------------------------------------
    print("\n[4] Demonstrating Trainer.fit()...")

    trainer = Trainer()

    # Run training loop
    # We use load_cached_data=True because we just processed the data in step 2 (Dataset Demo).
    # The Dataset class saves to cache automatically.
    trainer.fit(load_cached_data=True)

    # Verify model file creation
    if not os.path.exists(trainer.model_path):
        raise AssertionError(f"Model file was not saved at {trainer.model_path}")

    print("Trainer fit verification passed.")

    # ---------------------------------------------------------
    # 5. Trainer Demonstration (Inference)
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Trainer.predict()...")

    # Run inference
    # This will process the test set (debug subset) and generate submission.csv
    trainer.predict(load_cached_data=False)

    # Verify submission file creation
    if not os.path.exists(Config.submission_path):
        raise AssertionError(
            f"Submission file was not created at {Config.submission_path}"
        )

    # Verify submission content
    submission_df = pd.read_csv(Config.submission_path)
    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission columns: {submission_df.columns.tolist()}")

    expected_cols = ["id"] + Config.target_cols
    if list(submission_df.columns) != expected_cols:
        raise AssertionError("Submission columns do not match requirements.")

    if len(submission_df) != Config.debug_subset_size:
        raise AssertionError(
            f"Submission row count {len(submission_df)} does not match debug subset size {Config.debug_subset_size}."
        )

    print("Trainer predict verification passed.")

    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
