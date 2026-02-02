import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import (
    PathConfig,
    AudioConfig,
    MelConfig,
    ModelConfig,
    TrainConfig,
)
from library.utils import set_seed
from library.transforms import (
    MultiResMelSpectrogram,
    GPUSpecAugment,
    GPUNoiseInjector,
)
from library.model import ResNeStCRNN
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_metadata(source_dir, dest_dir, num_samples=50):
    """
    Creates small subset CSVs for rapid demonstration.
    """
    os.makedirs(dest_dir, exist_ok=True)

    files = {
        "train.csv": "demo_train.csv",
        "val.csv": "demo_val.csv",
        "test.csv": "demo_test.csv",
    }

    generated_paths = {}

    for src_name, dest_name in files.items():
        src_path = os.path.join(source_dir, src_name)
        dest_path = os.path.join(dest_dir, dest_name)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Sample subset, ensuring we don't exceed available rows
            n = min(num_samples, len(df))
            df_subset = df.sample(n=n, random_state=42).reset_index(drop=True)
            df_subset.to_csv(dest_path, index=False)
            generated_paths[src_name] = dest_path
            print(f"Created {dest_name} with {len(df_subset)} samples.")
        else:
            raise FileNotFoundError(f"Source metadata not found: {src_path}")

    return generated_paths


def verify_transforms_and_model(
    audio_config, mel_config, model_config, train_config, path_config
):
    """
    Verifies the shapes and logic of Transforms and Model.
    """
    print("\n=== Verifying Transforms and Model ===")

    device = torch.device(train_config.device)
    batch_size = 4
    seq_len = audio_config.n_samples

    # 1. Generate Dummy Audio (Batch, 1, Time)
    dummy_audio = torch.randn(batch_size, 1, seq_len).to(device)
    print(f"Input Audio Shape: {dummy_audio.shape}")

    # 2. Test Noise Injector
    # Note: This might not modify signal if noise files aren't loaded, but should run without error
    noise_injector = GPUNoiseInjector(path_config, audio_config, train_config).to(
        device
    )
    noise_injector.train()
    noisy_audio = noise_injector(dummy_audio)
    assert (
        noisy_audio.shape == dummy_audio.shape
    ), "Noise injector changed output shape."
    print("GPUNoiseInjector: OK")

    # 3. Test Mel Spectrogram
    mel_transform = MultiResMelSpectrogram(mel_config, audio_config).to(device)
    specs = mel_transform(noisy_audio)
    # Expected: (Batch, 3, n_mels, TimeFrames)
    # 3 channels because MultiResMelSpectrogram stacks Short, Medium, Long windows
    print(f"Spectrogram Shape: {specs.shape}")
    assert specs.dim() == 4, "Spectrogram should be 4D (B, C, F, T)"
    assert specs.shape[1] == 3, "Spectrogram should have 3 channels (Multi-Resolution)"
    assert (
        specs.shape[2] == mel_config.n_mels
    ), f"Freq dim should be {mel_config.n_mels}"
    print("MultiResMelSpectrogram: OK")

    # 4. Test SpecAugment
    spec_aug = GPUSpecAugment(train_config).to(device)
    spec_aug.train()
    aug_specs = spec_aug(specs)
    assert aug_specs.shape == specs.shape, "SpecAugment changed output shape."
    print("GPUSpecAugment: OK")

    # 5. Test Model
    model = ResNeStCRNN(model_config).to(device)
    logits = model(aug_specs)
    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (
        batch_size,
        model_config.num_classes,
    ), f"Expected logits shape ({batch_size}, {model_config.num_classes}), got {logits.shape}"
    print("ResNeStCRNN: OK")


def run_demo():
    # 1. Setup
    set_seed(42)

    # Define directories
    original_metadata_dir = "./metadata"
    working_dir = "./working/demo_execution"
    submission_path = os.path.join(working_dir, "submission.csv")

    # Create demo metadata (subset of real data)
    print("=== Preparing Demo Data ===")
    demo_paths = create_demo_metadata(
        original_metadata_dir, working_dir, num_samples=100
    )

    # 2. Configure
    # We override paths to point to our demo CSVs and demo working dir
    path_config = PathConfig(
        metadata_dir=working_dir,
        train_csv=demo_paths["train.csv"],
        val_csv=demo_paths["val.csv"],
        test_csv=demo_paths["test.csv"],
        working_dir=working_dir,
        submission_path=submission_path,
    )

    audio_config = AudioConfig()
    mel_config = MelConfig()
    model_config = ModelConfig()

    # Optimize training config for speed
    train_config = TrainConfig(
        batch_size=16,
        num_epochs=2,  # Run just 2 epochs
        num_workers=2,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=True,  # Enable debug mode if supported
    )

    print(f"Running on device: {train_config.device}")

    # 3. Verify Components
    verify_transforms_and_model(
        audio_config, mel_config, model_config, train_config, path_config
    )

    # 4. Initialize Trainer
    print("\n=== Initializing Trainer ===")
    trainer = Trainer(path_config, audio_config, mel_config, model_config, train_config)

    # 5. Run Training
    print("\n=== Starting Training Loop (Demo) ===")
    # load_cached_data=False ensures we process the new demo CSVs and don't load old cache
    trainer.fit(load_cached_data=False)

    # Check if best model was saved
    best_model_path = os.path.join(working_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Success: Best model saved at {best_model_path}")
    else:
        raise FileNotFoundError("Training finished but best_model.pth was not found.")

    # 6. Run Prediction
    print("\n=== Starting Inference (Demo) ===")
    trainer.predict(load_cached_data=False)

    # 7. Validate Submission
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Success: Submission file created at {submission_path}")
        print(sub_df.head())

        # Check format
        assert (
            "fname" in sub_df.columns and "label" in sub_df.columns
        ), "Submission missing required columns"
        assert len(sub_df) > 0, "Submission file is empty"
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Prediction finished but submission.csv was not found.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
