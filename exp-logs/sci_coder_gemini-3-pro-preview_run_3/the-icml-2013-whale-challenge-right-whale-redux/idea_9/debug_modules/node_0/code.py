import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import AudioConfig, TrainConfig, ModelConfig
from library.utils import set_seed, calculate_metrics
from library.dataset import AudioPreprocessor, get_dataloaders
from library.model import WhaleConvNeXt
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Whale Detection Pipeline Demo ===\n")

    # 1. Configuration Setup for Demo
    # We modify the configuration at runtime to ensure speed and isolation
    print("Step 1: Configuring runtime parameters...")
    TrainConfig.seed = 42
    TrainConfig.epochs = 1
    TrainConfig.batch_size = 4  # Small batch size for demo
    TrainConfig.num_workers = 2
    TrainConfig.working_dir = "./working/demo_execution"
    TrainConfig.debug = True  # Limits data processing to 100 samples

    # Ensure working directory exists and is clean
    if os.path.exists(TrainConfig.working_dir):
        shutil.rmtree(TrainConfig.working_dir)
    os.makedirs(TrainConfig.working_dir, exist_ok=True)

    # Set global seeds
    set_seed(TrainConfig.seed)
    print(f"Working directory set to: {TrainConfig.working_dir}")
    print(f"Debug mode: {TrainConfig.debug}")
    print("Configuration complete.\n")

    # 2. Verify Audio Preprocessing
    print("Step 2: Verifying Audio Preprocessor...")
    # Get a sample file path from metadata
    train_meta_path = os.path.join(TrainConfig.metadata_dir, "train.csv")
    df_train = pd.read_csv(train_meta_path)
    sample_rel_path = df_train.iloc[0]["file_path"]
    sample_full_path = os.path.join(TrainConfig.input_dir, sample_rel_path)

    preprocessor = AudioPreprocessor()

    # Process the file
    spec = preprocessor.process_path(sample_full_path)

    # Expected shape: (1, n_mels, time_steps)
    # Time steps = (duration * sample_rate) / hop_length
    # 2.0 * 2000 / 20 = 200 frames.
    # Note: STFT output size depends on padding/centering, usually N/hop + 1.
    # Let's verify dimensions roughly match config.
    expected_mels = AudioConfig.n_mels
    print(f"Spectrogram shape: {spec.shape}")

    assert spec.ndim == 3, f"Expected 3 dimensions (C, F, T), got {spec.ndim}"
    assert spec.shape[0] == 1, f"Expected 1 channel, got {spec.shape[0]}"
    assert (
        spec.shape[1] == expected_mels
    ), f"Expected {expected_mels} mels, got {spec.shape[1]}"
    print("Audio Preprocessor verification passed.\n")

    # 3. Verify Data Loading
    print("Step 3: Verifying DataLoaders...")
    # This will trigger process_and_cache_data with debug=True (100 samples)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    images, labels, clips = next(iter(train_loader))

    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")

    assert images.shape[0] == TrainConfig.batch_size, "Batch size mismatch"
    assert images.shape[1] == 1, "Channel dimension mismatch"
    assert images.shape[2] == AudioConfig.n_mels, "Frequency dimension mismatch"
    assert len(labels) == TrainConfig.batch_size, "Label count mismatch"
    print("DataLoaders verification passed.\n")

    # 4. Verify Model Architecture
    print("Step 4: Verifying Model Architecture...")
    model = WhaleConvNeXt()
    model.eval()

    # Run the fetched batch through the model
    with torch.no_grad():
        outputs = model(images)

    print(f"Model output shape: {outputs.shape}")

    # Expected output: (Batch_Size, Num_Classes) -> (4, 1)
    assert outputs.shape == (
        TrainConfig.batch_size,
        ModelConfig.num_classes,
    ), f"Expected output shape {(TrainConfig.batch_size, ModelConfig.num_classes)}, got {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"
    print("Model architecture verification passed.\n")

    # 5. Verify Metric Calculation
    print("Step 5: Verifying Metric Calculation...")
    y_true = [0, 0, 1, 1]
    y_pred_good = [0.1, 0.2, 0.8, 0.9]
    y_pred_bad = [0.9, 0.8, 0.2, 0.1]

    score_good = calculate_metrics(y_true, y_pred_good)
    score_bad = calculate_metrics(y_true, y_pred_bad)

    print(f"Score (Good predictions): {score_good}")
    print(f"Score (Bad predictions): {score_bad}")

    assert score_good == 1.0, "Metric calculation failed for perfect predictions"
    assert score_bad == 0.0, "Metric calculation failed for inverted predictions"
    print("Metric verification passed.\n")

    # 6. Run Full Training Pipeline (Trainer)
    print("Step 6: Running Trainer (Fit & Predict)...")
    trainer = Trainer()

    # Run fit (trains for 1 epoch on debug subset, validates, saves checkpoint, predicts on test)
    trainer.fit(debug=True)

    # Verify artifacts
    best_model_path = os.path.join(TrainConfig.working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    print(f"Checking for best model at: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."

    print(f"Checking for submission file at: {submission_path}")
    assert os.path.exists(submission_path), "Submission file was not generated."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    print("Trainer execution verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
