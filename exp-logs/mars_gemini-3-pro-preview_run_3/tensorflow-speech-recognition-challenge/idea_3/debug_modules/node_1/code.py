import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work if not already set
sys.path.append(os.getcwd())

# Import from the provided library files
import library.config as config
from library.utils import set_seed, calculate_accuracy
from library.dataset import (
    LogMelSpectrogram,
    SpecAugment,
    SpeechCommandsDataset,
    get_balanced_dataframes,
)
from library.model import ResNet34BiGRU
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Speech Command Recognition Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Safety
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Enable debug mode to use a small subset of data
    config.TRAIN_CONFIG["debug"] = True
    config.TRAIN_CONFIG["debug_samples"] = 50  # Small sample size for speed

    # Reduce training parameters
    config.TRAIN_CONFIG["batch_size"] = 4
    config.TRAIN_CONFIG["num_epochs"] = 2
    config.TRAIN_CONFIG["num_workers"] = 0  # Avoid multiprocessing overhead in demo
    config.TRAIN_CONFIG["early_stopping_patience"] = 2

    # Disable pretrained weights to avoid potential internet connection errors
    config.MODEL_CONFIG["pretrained"] = False

    # Set seed for reproducibility
    set_seed(config.TRAIN_CONFIG["seed"])
    print(
        "    Configuration updated: Debug=True, Epochs=2, Batch=4, Pretrained=False.\n"
    )

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("[2] Verifying utility functions...")

    # Test calculate_accuracy
    # Logits: Class 1 is higher in first row, Class 0 is higher in second row
    dummy_outputs = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    dummy_targets = torch.tensor([1, 0])  # Both match
    acc = calculate_accuracy(dummy_outputs, dummy_targets)

    assert acc == 1.0, f"Accuracy calculation failed. Expected 1.0, got {acc}"
    print("    calculate_accuracy() logic verified.\n")

    # ---------------------------------------------------------
    # 3. Verify Dataset and Transforms
    # ---------------------------------------------------------
    print("[3] Verifying Dataset and Transforms...")

    # Test LogMelSpectrogram
    spec_transform = LogMelSpectrogram()
    # Create a dummy waveform: (1, 16000)
    dummy_waveform = torch.randn(1, config.AUDIO_CONFIG["num_samples"])
    spec = spec_transform(dummy_waveform)

    # Check shape: (1, n_mels, time)
    # n_mels = 64. Time depends on hop_length. 16000/160 = 100 frames + padding ~ 101
    assert spec.dim() == 3, "Spectrogram must be 3D tensor (1, F, T)"
    assert spec.size(1) == config.AUDIO_CONFIG["n_mels"], "Incorrect Mel dimension"
    print(f"    LogMelSpectrogram output shape: {tuple(spec.shape)}")

    # Test SpecAugment
    augment = SpecAugment(freq_mask_param=5, time_mask_param=5)
    aug_spec = augment(spec.clone())
    assert aug_spec.shape == spec.shape, "Augmentation should not change tensor shape"
    print("    SpecAugment shape preservation verified.")

    # Test DataFrame Loading
    df_train, df_val, df_test = get_balanced_dataframes(load_cached_data=False)
    assert not df_train.empty, "Training dataframe is empty"
    print(f"    Loaded DataFrames (Full) - Train: {len(df_train)}, Val: {len(df_val)}")

    # Test Dataset Class (using a small slice of df_train)
    dataset = SpeechCommandsDataset(
        df_train.head(10), config.PATHS["train_audio_dir"], is_training=True
    )
    sample_spec, sample_label = dataset[0]

    assert torch.is_tensor(sample_spec), "Dataset item must be a tensor"
    assert isinstance(sample_label, (int, np.integer)), "Label must be an integer"
    print("    SpeechCommandsDataset item retrieval verified.\n")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("[4] Verifying Model Architecture...")

    model = ResNet34BiGRU()
    model.eval()

    # Create a dummy batch: (Batch, 1, n_mels, time)
    # Using the shape obtained from the dataset verification
    dummy_input = torch.randn(2, 1, spec.size(1), spec.size(2))

    with torch.no_grad():
        output = model(dummy_input)

    # Output should be (Batch, NumClasses)
    assert output.shape == (
        2,
        config.MODEL_CONFIG["num_classes"],
    ), f"Model output shape mismatch. Expected (2, 12), got {output.shape}"

    print(f"    Model forward pass successful. Output shape: {tuple(output.shape)}\n")

    # ---------------------------------------------------------
    # 5. Run Trainer (Fit and Predict)
    # ---------------------------------------------------------
    print("[5] Running Training Loop (Debug Mode)...")

    trainer = Trainer()

    # Verify Trainer loaded the debug subset
    print(f"    Trainer Train Loader size: {len(trainer.train_loader)} batches")

    # Run training
    trainer.fit()

    # Check if model was saved
    assert os.path.exists(
        config.PATHS["model_save_path"]
    ), "Best model file was not saved."
    print("    Model training complete and checkpoint saved.")

    # Run prediction
    print("    Generating predictions...")
    trainer.predict()

    # Verify submission file
    submission_path = config.PATHS["submission_path"]
    assert os.path.exists(submission_path), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == ["fname", "label"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    print(f"    Submission generated at {submission_path}")
    print(f"    First 3 rows:\n{sub_df.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
