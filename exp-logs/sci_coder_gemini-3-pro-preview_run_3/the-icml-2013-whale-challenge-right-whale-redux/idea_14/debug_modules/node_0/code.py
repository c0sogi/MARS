import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from provided library files
from library.config import AudioConfig, ModelConfig, TrainConfig, PathConfig
from library.utils import set_seed
from library.dataset import get_dataloaders, compute_spectrogram
from library.model import WhaleEfficientNet
from library.trainer import Trainer


def main():
    print("=== Starting Right Whale Detection Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    print("--- Step 1: Configuration & Setup ---")
    set_seed(42)

    # Override Configs for Speed/Demo purposes
    # We use class attributes to ensure these changes propagate
    TrainConfig.epochs = 1
    TrainConfig.batch_size = 4
    TrainConfig.debug = True
    TrainConfig.debug_sample_size = 20  # Use only 20 samples for speed
    TrainConfig.num_workers = 0  # Avoid multiprocessing overhead for tiny demo

    # Update PathConfig to use a demo directory
    PathConfig.working_dir = "./working/demo_execution"
    PathConfig.cache_dir = os.path.join(PathConfig.working_dir, "cache")
    PathConfig.checkpoint_dir = os.path.join(PathConfig.working_dir, "checkpoints")
    PathConfig.submission_dir = os.path.join(PathConfig.working_dir, "submission")
    PathConfig.submission_path = os.path.join(
        PathConfig.submission_dir, "submission.csv"
    )
    PathConfig.teacher_checkpoint = os.path.join(
        PathConfig.checkpoint_dir, "demo_best.pth"
    )

    # Clean up previous demo run if exists to ensure a fresh start
    if os.path.exists(PathConfig.working_dir):
        shutil.rmtree(PathConfig.working_dir)

    # Create necessary directories
    PathConfig.create_dirs()
    print(f"Working directory set to: {PathConfig.working_dir}")
    print("Configuration updated for rapid execution.")

    # ---------------------------------------------------------
    # 2. Data Processing & Loading
    # ---------------------------------------------------------
    print("\n--- Step 2: Data Processing & Loading ---")

    # Test Spectrogram Computation on a single real file
    train_meta = pd.read_csv(PathConfig.train_meta)
    sample_file_rel = train_meta.iloc[0]["file_path"]
    sample_file_full = os.path.join(PathConfig.input_dir, sample_file_rel)

    print(f"Testing spectrogram generation on: {sample_file_rel}")
    spec = compute_spectrogram(sample_file_full, AudioConfig())
    print(f"Spectrogram shape: {spec.shape}")

    # Verify Spectrogram Shape: (1, n_mels, time)
    assert spec.dim() == 3, "Spectrogram must be 3D (C, F, T)"
    assert spec.shape[0] == 1, "Channel dimension should be 1"
    assert (
        spec.shape[1] == AudioConfig.n_mels
    ), f"Mel bins should be {AudioConfig.n_mels}"
    assert spec.shape[2] > 0, "Time dimension should be positive"

    # Get DataLoaders (Debug Mode)
    print("Initializing DataLoaders...")
    loaders = get_dataloaders(debug=True, load_cached_data=False)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"Train batches: {len(train_loader)}")

    # Fetch one batch to verify
    images, targets = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch targets shape: {targets.shape}")

    assert images.shape[0] == TrainConfig.batch_size
    assert images.shape[1] == 1
    assert targets.shape[0] == TrainConfig.batch_size

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n--- Step 3: Model Initialization ---")
    model = WhaleEfficientNet(ModelConfig())
    device = TrainConfig.device
    model.to(device)
    print(f"Model initialized on {device}")

    # Test Forward Pass
    dummy_input = images.to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (TrainConfig.batch_size, ModelConfig.num_classes)

    # ---------------------------------------------------------
    # 4. Training Loop (Teacher Phase)
    # ---------------------------------------------------------
    print("\n--- Step 4: Training (Teacher Phase) ---")
    trainer = Trainer(model, train_loader, val_loader, TrainConfig())

    # Train for 1 epoch
    save_path = PathConfig.teacher_checkpoint
    trainer.train(save_path)

    assert os.path.exists(save_path), "Checkpoint file was not saved."
    print("Training complete. Checkpoint verified.")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n--- Step 5: Inference & Submission ---")
    # Generate submission for the test set (debug subset)
    trainer.generate_submission(test_loader, PathConfig.submission_path)

    assert os.path.exists(PathConfig.submission_path), "Submission file not found."

    submission_df = pd.read_csv(PathConfig.submission_path)
    print(f"Submission rows: {len(submission_df)}")
    print(submission_df.head(3))

    # Verify submission format
    assert "clip" in submission_df.columns
    assert "probability" in submission_df.columns
    assert len(submission_df) > 0

    # ---------------------------------------------------------
    # 6. Student Training (Pseudo-labeling)
    # ---------------------------------------------------------
    print("\n--- Step 6: Student Training (Pseudo-labeling) ---")
    # Use the generated submission as pseudo-labels for the next training phase
    print("Re-initializing loaders with pseudo-labels...")

    # We load cached data this time to speed things up
    student_loaders = get_dataloaders(
        debug=True, load_cached_data=True, pseudo_labels=submission_df
    )
    student_train_loader = student_loaders["train"]

    # Check size: Should include both train and test samples now
    print(f"Student train batches: {len(student_train_loader)}")

    # Verify that we can iterate and get targets
    st_images, st_targets = next(iter(student_train_loader))
    print(f"Student batch targets: {st_targets}")

    # Targets should be floats (probabilities)
    assert st_targets.dtype == torch.float32

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
